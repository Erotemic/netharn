import pandas as pd
import numpy as np
import ubelt as ub
from netharn import util
from netharn.util import profiler
import warnings


class DetectionMetrics(object):
    """
    Attributes:
        true (nh.data.CocoAPI): ground truth dataset
        pred (nh.data.CocoAPI): predictions dataset
    """
    def __init__(dmet, true=None):
        import netharn as nh
        if true is None:
            true = nh.data.coco_api.CocoDataset()
        dmet.true = true
        dmet.pred = nh.data.coco_api.CocoDataset()

    @classmethod
    def demo(cls, **kw):
        """
        Creates random true boxes and predicted boxes that have some noisy
        offset from the truth.

        Example:
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=100, nboxes=(0, 3), n_fp=(0, 1))
            >>> #print(dmet.score_coco()['mAP'])
            >>> print(dmet.score_netharn(bias=0)['mAP'])
            >>> print(dmet.score_voc(bias=0)['mAP'])
        """
        import netharn as nh

        rng = nh.util.ensure_rng(kw.get('rng', 0))

        class RV(object):
            def __init__(self, val):
                self.rng = rng
                if isinstance(val, tuple):
                    self.low = val[0]
                    self.high = val[1] + 1
                else:
                    self.low = val
                    self.high = val + 1

            def __call__(self):
                return self.rng.randint(self.low, self.high)

        nclasses = kw.get('nclasses', 1)
        nimgs = kw.get('nimgs', 1)

        classes = list(range(nclasses))

        nboxes = RV(kw.get('nboxes', 1))
        n_fp = RV(kw.get('n_fp', 0))
        n_fn = RV(kw.get('n_fn', 0))

        box_noise = kw.get('box_noise', 0)
        cls_noise = kw.get('cls_noise', 0)

        anchors = kw.get('anchors', None)

        scale = 100

        true = nh.data.coco_api.CocoDataset()
        pred = nh.data.coco_api.CocoDataset()

        for cid in classes:
            true.add_category('cat_{}'.format(cid), cid=cid)
            pred.add_category('cat_{}'.format(cid), cid=cid)

        for gid in range(nimgs):
            nboxes_ = nboxes()
            n_fp_ = n_fp()
            n_fn_ = n_fn()

            imgname = 'img_{}'.format(gid)
            gid = true.add_image(imgname, gid=gid)
            gid = pred.add_image(imgname, gid=gid)

            # Generate random ground truth detections
            true_boxes = nh.util.Boxes.random(num=nboxes_, scale=scale,
                                              anchors=anchors, rng=rng,
                                              format='cxywh')
            # Prevent 0 sized boxes
            true_boxes.data[2:4] += 1
            true_cids = rng.choice(classes, size=len(true_boxes))
            true_weights = np.ones(len(true_boxes))

            # Initialize predicted detections as a copy of truth
            pred_boxes = true_boxes.copy()
            pred_cids = true_cids.copy()

            # Perterb box coordinates
            pred_boxes.data = np.abs(pred_boxes.data.astype(np.float) + (
                rng.randn() * box_noise))

            # Perterb class predictions
            change = rng.rand(len(pred_cids)) < cls_noise
            pred_cids_swap = rng.choice(classes, size=len(pred_cids))
            pred_cids[change] = pred_cids_swap[change]

            # Drop true positive boxes
            if n_fn_:
                pred_boxes.data = pred_boxes.data[n_fn_:]
                pred_cids = pred_cids[n_fn_:]

            # Add false positive boxes
            if n_fp_:
                pred_boxes.data = np.vstack([
                    pred_boxes.data,
                    nh.util.Boxes.random(num=n_fp_, scale=scale, rng=rng,
                                         format='cxywh').data])
                pred_cids = np.hstack([pred_cids,
                                      rng.choice(classes, size=n_fp_)])

            # Create netharn style confusion data
            pred_scores = np.linspace(0.01, 10, len(pred_boxes))

            for bbox, cid, weight in zip(true_boxes.to_xywh(), true_cids,
                                         true_weights):
                true.add_annotation(gid, cid, bbox=bbox, weight=weight)
            for bbox, cid, score in zip(pred_boxes.to_xywh(), pred_cids,
                                        pred_scores):
                pred.add_annotation(gid, cid, bbox=bbox, score=score)

        dmet = cls()
        dmet.true = true
        dmet.pred = pred
        return dmet

    def add_predictions(dmet, imgname, pred_boxes, pred_cids, pred_scores):
        pred_gid = dmet.pred.add_image(imgname)
        for bbox, cid, score in zip(pred_boxes.to_xywh(), pred_cids, pred_scores):
            dmet.pred.add_annotation(pred_gid, cid, bbox=bbox, score=score)

    def add_truth(dmet, imgname, true_boxes, true_cids, true_weights):
        true_gid = dmet.true.add_image(imgname)
        for bbox, cid, weight in zip(true_boxes.to_xywh(), true_cids, true_weights):
            dmet.true.add_annotation(true_gid, cid, bbox=bbox, weight=weight)

    def score_netharn(dmet, ovthresh=0.5, bias=0, method='voc2012', gids=None):
        y_accum = ub.ddict(list)
        # confusions = []
        if gids is None:
            gids = dmet.pred.imgs.keys()
        for gid in gids:
            pred_annots = dmet.pred.annots(gid=gid)
            true_annots = dmet.true.annots(gid=gid)

            true_boxes = true_annots.boxes
            true_cxs = true_annots.cids
            true_weights = true_annots._lookup('weight')

            pred_boxes = pred_annots.boxes
            pred_cxs = pred_annots.cids
            pred_scores = pred_annots._lookup('score')

            y = detection_confusions(true_boxes, true_cxs, true_weights,
                                     pred_boxes, pred_scores, pred_cxs,
                                     bg_weight=1.0, ovthresh=ovthresh,
                                     bg_cls=-1, bias=bias)
            y['gid'] = [gid] * len(y['pred'])
            for k, v in y.items():
                y_accum[k].extend(v)

        y_df = pd.DataFrame(y_accum)

        # class agnostic score
        ap, prec, rec = pr_curves(y_df)
        peritem = {
            'ap': ap,
            'pr': (prec, rec),
        }

        # perclass scores
        perclass = {}
        cx_to_group = dict(iter(y_df.groupby('cx')))
        for cx in cx_to_group:
            # for cx, group in cx_to_group.items():
            group = cx_to_group.get(cx, None)
            ap, prec, rec = pr_curves(group, method=method)
            perclass[cx] = {
                'ap': ap,
                'pr': (prec, rec),
            }

        mAP = np.nanmean([d['ap'] for d in perclass.values()])
        nh_scores = {
            'mAP': mAP,
            'perclass': perclass,
            'peritem': peritem
        }
        return nh_scores

    def score_voc(dmet, ovthresh=0.5, bias=1, method='voc2012', gids=None):
        recs = {}
        cx_to_lines = ub.ddict(list)
        # confusions = []
        if gids is None:
            gids = dmet.pred.imgs.keys()
        for gid in gids:
            pred_annots = dmet.pred.annots(gid=gid)
            true_annots = dmet.true.annots(gid=gid)

            true_boxes = true_annots.boxes
            true_cxs = true_annots.cids
            true_weights = true_annots._lookup('weight')

            pred_boxes = pred_annots.boxes
            pred_cxs = pred_annots.cids
            pred_scores = pred_annots._lookup('score')

            recs[gid] = []
            for bbox, cx, weight in zip(true_boxes.to_tlbr().data,
                                         true_cxs, true_weights):
                recs[gid].append({
                    'bbox': bbox,
                    'difficult': weight < .5,
                    'name': cx
                })

            for bbox, cx, score in zip(pred_boxes.to_tlbr().data,
                                         pred_cxs, pred_scores):
                cx_to_lines[cx].append([gid, score] + list(bbox))

        perclass = ub.ddict(dict)
        for cx in cx_to_lines.keys():
            lines = cx_to_lines[cx]
            classname = cx
            rec, prec, ap = voc_eval(lines, recs, classname, ovthresh=ovthresh,
                                     bias=bias, method=method)
            perclass[cx]['pr'] = (rec, prec)
            perclass[cx]['ap'] = ap

        mAP = np.nanmean([d['ap'] for d in perclass.values()])
        voc_scores = {
            'mAP': mAP,
            'perclass': perclass,
        }
        return voc_scores

    def score_coco(dmet):
        from pycocotools import coco
        from pycocotools import cocoeval
        # The original pycoco-api prints to much, supress it
        with util.SupressPrint(coco, cocoeval):
            cocoGt = dmet.true._aspycoco()
            cocoDt = dmet.pred._aspycoco()

            for ann in cocoGt.dataset['annotations']:
                w, h = ann['bbox'][-2:]
                ann['ignore'] = ann['weight'] < .5
                ann['area'] = w * h
                ann['iscrowd'] = False

            for ann in cocoDt.dataset['annotations']:
                w, h = ann['bbox'][-2:]
                ann['area'] = w * h

            evaler = cocoeval.COCOeval(cocoGt, cocoDt, iouType='bbox')
            evaler.evaluate()
            evaler.accumulate()
            evaler.summarize()
            coco_ap = evaler.stats[1]
            coco_scores = {
                'mAP': coco_ap,
            }
        return coco_scores


@profiler.profile
def detection_confusions(true_boxes, true_cxs, true_weights, pred_boxes, pred_scores, pred_cxs, bg_weight=1.0, ovthresh=0.5, bg_cls=-1, bias=0.0):
    """ Classify detections by assigning to groundtruth boxes.

    Given predictions and truth for an image return (y_pred, y_true,
    y_score), which is suitable for sklearn classification metrics

    Args:
        true_boxes (ndarray): boxes in tlbr format
        true_cxs (ndarray): classes of each box
        true_weights (ndarray): weight of this each groundtruth item
        pred_boxes (ndarray): predicted boxes in tlbr format
        pred_scores (ndarray): scores for each prediction
        pred_cxs (ndarray): class predictions
        ovthresh (float): overlap threshold

        bg_weight (ndarray): weight of background predictions
          (default=1)

        single_class(): if True, considers this to be a binary problem
        bias : for computing overlap either 1 or 0

    Returns:
        dict: with relevant clf information

    Ignore:
        from xinspect.dynamic_kwargs import get_func_kwargs
        globals().update(get_func_kwargs(detection_confusions))

    Example:
        >>> from netharn.metrics.detections import *
        >>> from netharn.metrics.detections import _ave_precision, pr_curves
        >>> true_boxes = np.array([[ 0,  0, 10, 10],
        >>>                        [10,  0, 20, 10],
        >>>                        [10,  0, 20, 10],
        >>>                        [20,  0, 30, 10]])
        >>> true_weights = np.array([1, 0, .9, 1])
        >>> bg_weight = 1.0
        >>> true_cxs = np.array([0, 0, 1, 1])
        >>> pred_boxes = np.array([[6, 2, 20, 10],
        >>>                        [3,  2, 9, 7],
        >>>                        [20,  0, 30, 10]])
        >>> pred_scores = np.array([.5, .5, .5])
        >>> pred_cxs = np.array([0, 0, 1])
        >>> y = detection_confusions(true_boxes, true_cxs, true_weights,
        >>>                          pred_boxes, pred_scores, pred_cxs,
        >>>                          bg_weight=bg_weight, ovthresh=.5)
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
           pred  true  score  weight  cx  txs  pxs
        0     1     1 0.5000  1.0000   1    3    2
        1     0    -1 0.5000  1.0000   0   -1    1
        2     0     0 0.5000  0.0000   0    1    0
        3    -1     0 0.0000  1.0000   0    0   -1
        4    -1     1 0.0000  0.9000   1    2   -1

    Example:
        >>> true_boxes = np.array([[ 0,  0, 10, 10],
        >>>                        [10,  0, 20, 10],
        >>>                        [10,  0, 20, 10],
        >>>                        [20,  0, 30, 10]])
        >>> true_weights = np.array([1, 0.0, 1, 1.0])
        >>> bg_weight = 1.0
        >>> true_cxs = np.array([0, 0, 1, 1])
        >>> pred_boxes = np.array([[6, 2, 20, 10],
        >>>                        [3,  2, 9, 7],
        >>>                        [20,  0, 30, 10]])
        >>> pred_scores = np.array([.5, .6, .7])
        >>> pred_cxs = np.array([0, 0, 1])
        >>> y = detection_confusions(true_boxes, true_cxs, true_weights,
        >>>                          pred_boxes, pred_scores, pred_cxs,
        >>>                          bg_weight=bg_weight, ovthresh=.5)
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
    """
    y_pred = []
    y_true = []
    y_score = []
    y_weight = []
    cxs = []

    y_pxs = []
    y_txs = []

    if bg_weight is None:
        bg_weight = 1.0

    if False:
        if isinstance(true_boxes, util.Boxes):
            true_boxes = true_boxes.data
        if isinstance(pred_boxes, util.Boxes):
            pred_boxes = pred_boxes.data
    else:
        if not isinstance(true_boxes, util.Boxes):
            true_boxes = util.Boxes(true_boxes, 'tlbr')
        if not isinstance(pred_boxes, util.Boxes):
            pred_boxes = util.Boxes(pred_boxes, 'tlbr')

    # Keep track of which true items have been used
    true_unused = np.ones(len(true_cxs), dtype=np.bool)
    if true_weights is None:
        true_weights = np.ones(len(true_cxs))
    else:
        true_weights = np.array(true_weights)
    pred_scores = np.array(pred_scores)
    pred_cxs = np.array(pred_cxs)
    true_cxs = np.array(true_cxs)

    # Group true boxes by class
    # Keep track which true boxes are unused / not assigned
    cx_to_idxs = ub.group_items(range(len(true_cxs)), true_cxs)
    cx_to_tboxes = util.group_items(true_boxes, true_cxs, axis=0)
    cx_to_tweight = util.group_items(true_weights, true_cxs, axis=0)

    # cx_to_boxes = ub.group_items(true_boxes, true_cxs)
    # cx_to_boxes = ub.map_vals(np.array, cx_to_boxes)

    # sort predictions by descending score
    _pred_sortx = pred_scores.argsort()[::-1]
    _pred_boxes = pred_boxes.take(_pred_sortx, axis=0)
    _pred_cxs = pred_cxs.take(_pred_sortx, axis=0)
    _pred_scores = pred_scores.take(_pred_sortx, axis=0)

    # For each predicted detection box
    # Allow it to match the truth of a particular class
    for px, cx, box, score in zip(_pred_sortx, _pred_cxs, _pred_boxes, _pred_scores):
        cls_true_idxs = cx_to_idxs.get(cx, [])

        ovmax = -np.inf
        ovidx = None
        weight = bg_weight
        tx = None  # we will set this to the index of the assignd gt

        if len(cls_true_idxs):
            cls_true_boxes = cx_to_tboxes[cx]
            cls_true_weights = cx_to_tweight[cx]
            # cls_true_boxes = true_boxes.take(cls_true_idxs, axis=0)
            # cls_true_weights = true_weights.take(cls_true_idxs, axis=0)

            overlaps = cls_true_boxes.ious(box, bias=bias)

            # choose best score by default
            ovidx = overlaps.argsort()[-1]
            ovmax = overlaps[ovidx]
            weight = cls_true_weights[ovidx]
            tx = cls_true_idxs[ovidx]

        if ovmax > ovthresh and true_unused[tx]:
            # Assign this prediction to a groundtruth object
            # Mark this prediction as a true positive
            y_pred.append(cx)
            y_true.append(cx)
            y_score.append(score)
            y_weight.append(weight)
            cxs.append(cx)
            # cls_unused[ovidx] = False

            tx = cls_true_idxs[ovidx]
            true_unused[tx] = False

            y_pxs.append(px)
            y_txs.append(tx)
        else:
            # Assign this prediction to a the background
            # Mark this prediction as a false positive
            y_pred.append(cx)
            y_true.append(bg_cls)  # use -1 as background ignore class
            y_score.append(score)
            y_weight.append(bg_weight)
            cxs.append(cx)

            tx = -1
            y_pxs.append(px)
            y_txs.append(tx)

    # All pred boxes have been assigned to a truth box or the background.
    # Mark unused true boxes we failed to predict as false negatives
    for tx in np.where(true_unused)[0]:
        # Mark each unmatched truth as a false negative
        y_pred.append(-1)
        y_true.append(true_cxs[tx])
        y_score.append(0.0)
        y_weight.append(true_weights[tx])
        cxs.append(true_cxs[tx])

        px = -1
        y_pxs.append(px)
        y_txs.append(tx)

    y = {
        'pred': y_pred,
        'true': y_true,
        'score': y_score,
        'weight': y_weight,
        'cx': cxs,
        'txs': y_txs,  # index into the original true box for this row
        'pxs': y_pxs,  # index into the original pred box for this row
    }
    # print('y = {}'.format(ub.repr2(y, nl=1)))
    # y = pd.DataFrame(y)
    return y


def _ave_precision(rec, prec, method='voc2012'):
    """ Compute AP from precision and recall

    Returns:
        float

    ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If method == voc2007, uses the VOC 07 11 point method (default:False).
    """
    if method == 'voc2007':
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    elif method == 'voc2012':
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    else:
        raise KeyError(method)

    if False:
        # sklearn metric
        ap = -np.sum(np.diff(rec[::-1]) * np.array(prec[::-1])[:-1])
    return ap


def score_detection_assignment(y, labels=None, method='voc2012'):
    """ Measures scores of predicted detections assigned to groundtruth objects

    Args:
        y (pd.DataFrame): pre-measured frames of predictions, truth,
            weight and class.
        method (str): either voc2007 voc2012 or sklearn

    Returns:
        pd.DataFrame

    Example:
        >>> # xdoc: +IGNORE_WANT
        >>> y_true = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,-1,-1,-1]
        >>> y_pred = [1, 3,-1, 2,-1, 1, 2, 1,-1, 3, 3, 3, 3, 3, 4, 4, 3, 1, 1]
        >>> y = pd.DataFrame.from_dict({
        >>>     'true': y_true,
        >>>     'pred': y_pred,
        >>> })
        >>> y['score'] = 0.99
        >>> y['weight'] = 1.0
        >>> ave_precs = ave_precisions(y, method='voc2007')
        >>> print(ave_precs)
           cx     ap
        0   1 0.2727
        1   2 0.1364
        2   3 1.0000
        3   4 1.0000
        >>> mAP = np.nanmean(ave_precs['ap'])
        >>> print('mAP = {:.4f}'.format(mAP))
        mAP = 0.6023
        >>> # -----------------
        >>> ave_precs = ave_precisions(y, method='voc2012')
        >>> print(ave_precs)
           cx     ap
        0   1 0.2500
        1   2 0.1000
        2   3 1.0000
        3   4 1.0000
        >>> mAP = np.nanmean(ave_precs['ap'])
        >>> print('mAP = {:.4f}'.format(mAP))
        mAP = 0.5875
    """
    if method not in ['sklearn', 'voc2007', 'voc2012']:
        raise KeyError(method)

    if 'cx' not in y:
        cx = y['true'].copy()
        flags = cx == -1
        cx[flags] = y['pred'][flags]
        y['cx'] = cx

    if labels is None:
        labels = pd.unique(y['cx'])

    # because we use -1 to indicate a wrong prediction we can use max to
    # determine the class groupings.
    cx_to_group = dict(iter(y.groupby('cx')))
    class_aps = []
    for cx in labels:
        # for cx, group in cx_to_group.items():
        group = cx_to_group.get(cx, None)
        ap, prec, rec = pr_curves(group, method)
        class_aps.append((cx, ap))

    ave_precs = pd.DataFrame(class_aps, columns=['cx', 'ap'])
    return ave_precs


def pr_curves(y, method='voc2012'):  # -> Tuple[float, ndarray, ndarray]:
    """ Compute a PR curve from a method

    Args:
        y (pd.DataFrame): output of detection_confusions
    """
    if method not in ['sklearn', 'voc2007', 'voc2012']:
        raise KeyError(method)

    # compute metrics on a per class basis
    if y is None:
        return np.nan, [], []

    # References [Manning2008] and [Everingham2010] present alternative
    # variants of AP that interpolate the precision-recall curve. Currently,
    # average_precision_score does not implement any interpolated variant
    # http://scikit-learn.org/stable/modules/model_evaluation.html

    # g2 = y[y.weight > 0]
    # prec2, rec2, thresh = sklearn.metrics.precision_recall_curve(
    #     (g2.true > -1).values,
    #     g2.score.values,
    #     sample_weight=g2.weight.values)
    # eav_ap = _ave_precision(rec2, prec2, method=method)
    # print('eav_ap = {!r}'.format(eav_ap))

    if method == 'sklearn':
        # In the future, we should simply use the sklearn version
        # which gives nice easy to reproduce results.
        import sklearn.metrics
        df = y
        ap = sklearn.metrics.average_precision_score(
            y_true=(df['true'].values == df['pred'].values).astype(np.int),
            y_score=df['score'].values,
            sample_weight=df['weight'].values,
        )
        raise NotImplementedError('todo: return pr curves')
        return ap, [], []
    elif method == 'voc2007' or method == 'voc2012':
        y = y.sort_values('score', ascending=False)
        # if True:
        #     # ignore "difficult" matches
        #     y = y[y.weight > 0]

        # npos = sum(y.true >= 0)
        npos = y[y.true >= 0].weight.sum()
        dets = y[y.pred > -1]
        if npos > 0 and len(dets) > 0:
            tp = (dets.pred == dets.true).values.astype(np.int)
            fp = 1 - tp
            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)

            eps = np.finfo(np.float64).eps
            rec = 1 if npos == 0 else tp_cum / npos
            prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)

            ap = _ave_precision(rec, prec, method=method)
        else:
            prec, rec = None, None
            if npos == 0:
                ap = np.nan
            if len(dets) == 0:
                if npos == 0:
                    ap = np.nan
                ap = 0.0
    else:
        raise KeyError(method)

    return ap, prec, rec


def voc_eval(lines, recs, classname, ovthresh=0.5, method='voc2012', bias=1):
    import copy
    # imagenames = ([x.strip().split(' ')[0] for x in lines])
    imagenames = ([x[0] for x in lines])
    recs2 = copy.deepcopy(recs)

    # BUGFIX: need to score images with no predictions / no truth
    imagenames += list(recs.keys())

    # BUGFIX: the original code did not cast this to a set
    imagenames = sorted(set(imagenames))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs2[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    splitlines = lines
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([x[1] for x in splitlines])
    BB = np.array([[z for z in x[2:]] for x in splitlines])

    # splitlines = [x.strip().split(' ') for x in lines]
    # confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)  #
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + bias, 0.)
                ih = np.maximum(iymax - iymin + bias, 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + bias) * (bb[3] - bb[1] + bias) +
                       (BBGT[:, 2] - BBGT[:, 0] + bias) *
                       (BBGT[:, 3] - BBGT[:, 1] + bias) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = _ave_precision(rec=rec, prec=prec, method=method)
    return rec, prec, ap


ave_precisions = score_detection_assignment


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.metrics.detections all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
