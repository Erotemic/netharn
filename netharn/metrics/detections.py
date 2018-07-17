import pandas as pd
import numpy as np
import ubelt as ub
from netharn import util
from netharn.util import profiler


def _devcheck_voc_consistency():
    """
    # CHECK FOR ISSUES WITH MY MAP COMPUTATION
    """
    def voc_eval(lines, recs, classname, ovthresh=0.5, method=False, bias=1):
        import copy
        imagenames = ([x.strip().split(' ')[0] for x in lines])
        # BUGFIX: the original code did not cast this to a set
        imagenames = set(imagenames)
        recs2 = copy.deepcopy(recs)

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

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        # sorted_scores = np.sort(-confidence)  #
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
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

        def voc_ap(rec, prec, use_07_metric=False):
            """ ap = voc_ap(rec, prec, [use_07_metric])
            Compute VOC AP given precision and recall.
            If use_07_metric is true, uses the
            VOC 07 11 point method (default:False).
            """
            if use_07_metric:
                # 11 point metric
                ap = 0.
                for t in np.arange(0., 1.1, 0.1):
                    if np.sum(rec >= t) == 0:
                        p = 0
                    else:
                        p = np.max(prec[rec >= t])
                    ap = ap + p / 11.
            else:
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
            return ap

        ap2 = voc_ap(rec, prec, use_07_metric=method ==  'voc2007')
        ap = _ave_precision(rec, prec, method)

        assert ap == ap2
        return rec, prec, ap
    import netharn as nh
    method = 'voc2012'

    xdata = []
    ydatas = ub.ddict(list)

    for noise in np.linspace(0, 5, 10):
        recs = {}
        lines = []
        confusions = []
        rng = np.random.RandomState(0)

        classname = 0
        nimgs = 5
        nboxes = 2
        for imgname in range(nimgs):
            imgname = str(imgname)

            true_boxes = nh.util.Boxes.random(num=nboxes, scale=100, rng=rng, format='cxywh')
            pred_boxes = true_boxes.copy()
            pred_boxes.data = pred_boxes.data.astype(np.float) + (rng.rand() * noise)

            true_boxes = true_boxes.to_tlbr().data
            pred_boxes = pred_boxes.to_tlbr().data
            # pred_boxes = nh.util.Boxes.random(num=10, scale=100, rng=rng, format='tlbr')

            recs[imgname] = []
            for bbox in true_boxes:
                recs[imgname].append({
                    'bbox': bbox,
                    'difficult': False,
                    'name': classname
                })

            for bbox, score in zip(pred_boxes, np.arange(len(pred_boxes))):
                lines.append('{} {} {} {} {} {}'.format(imgname, score, *bbox))

            # Create netharn style confusion data
            true_cxs = np.array([0] * len(true_boxes))
            pred_cxs = np.array([0] * len(true_boxes))
            true_weights = np.array([1] * len(true_boxes))
            pred_scores = np.arange(len(pred_boxes))

            y = pd.DataFrame(detection_confusions(true_boxes, true_cxs, true_weights,
                                                  pred_boxes, pred_scores, pred_cxs,
                                                  bg_weight=1.0, ovthresh=0.5, bg_cls=-1,
                                                  bias=0.0, PREFER_WEIGHTED_TRUTH=False))
            y['gx'] = int(imgname)
            y = (y)
            confusions.append(y)
        y = pd.concat(confusions)

        ap3 = ave_precisions(y, method=method)['ap']
        rec, prec, ap = voc_eval(lines, recs, classname, ovthresh=0.5,
                                 method=method, bias=0.0)
        prec2, rec2, ap2 = _multiclass_ap(y)
        ap2 = _ave_precision(rec2, prec2, method)
        print('noise = {!r}'.format(noise))
        print('ap3 = {!r}'.format(ap3.values.mean()))
        print('ap = {!r}'.format(ap))
        print('ap2 = {!r}'.format(ap2))
        print('---')
        xdata.append(noise)
        ydatas['orig'].append(ap)
        ydatas['eav'].append(ap2)
        ydatas['mine'].append(ap3)
    nh.util.autompl()
    nh.util.multi_plot(xdata=xdata, ydata=ydatas)


def _multiclass_ap(y):
    """ computes pr like lightnet from netharn confusions """
    y = y.sort_values('score', ascending=False)

    num_annotations = y[y.true >= 0].weight.sum()
    positives = []

    for rx, row in y.iterrows():
        if row.pred >= 0:
            if row.true == row.pred:
                positives.append((row.score, True))
            else:
                positives.append((row.score, False))

    # sort matches by confidence from high to low
    positives = sorted(positives, key=lambda d: (d[0], -d[1]), reverse=True)

    tps = []
    fps = []
    tp_counter = 0
    fp_counter = 0

    # all matches in dataset
    for pos in positives:
        if pos[1]:
            tp_counter += 1
        else:
            fp_counter += 1
        tps.append(tp_counter)
        fps.append(fp_counter)

    precision = []
    recall = []
    for tp, fp in zip(tps, fps):
        recall.append(tp / num_annotations)
        precision.append(tp / np.maximum(fp + tp, np.finfo(np.float64).eps))

    import scipy
    num_of_samples = 100

    if len(precision) > 1 and len(recall) > 1:
        p = np.array(precision)
        r = np.array(recall)
        p_start = p[np.argmin(r)]
        samples = np.arange(0., 1., 1.0 / num_of_samples)
        interpolated = scipy.interpolate.interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
        avg = sum(interpolated) / len(interpolated)
    elif len(precision) > 0 and len(recall) > 0:
        # 1 point on PR: AP is box between (0,0) and (p,r)
        avg = precision[0] * recall[0]
    else:
        avg = float('nan')

    return precision, recall, avg


@profiler.profile
def detection_confusions(true_boxes, true_cxs, true_weights, pred_boxes,
                         pred_scores, pred_cxs, bg_weight=1.0, ovthresh=0.5,
                         bg_cls=-1, bias=0.0, PREFER_WEIGHTED_TRUTH=False):
    """
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
        pd.DataFrame: with relevant clf information

    Ignore:
        from xinspect.dynamic_kwargs import get_func_kwargs
        globals().update(get_func_kwargs(detection_confusions))

    Example:
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
           cx  pred  score  true  weight
        0   1     1 0.5000     1  1.0000
        1   0     0 0.5000    -1  1.0000
        2   0     0 0.5000     0  0.0000
        3   0    -1 0.0000     0  1.0000
        4   1    -1 0.0000     1  0.9000

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

    if isinstance(true_boxes, util.Boxes):
        true_boxes = true_boxes.data
    if isinstance(pred_boxes, util.Boxes):
        pred_boxes = pred_boxes.data

    # Group true boxes by class
    # Keep track which true boxes are unused / not assigned
    cx_to_idxs = ub.group_items(range(len(true_cxs)), true_cxs)
    cx_to_unused = {cx: np.array([True] * len(idxs))
                    for cx, idxs in cx_to_idxs.items()}

    # cx_to_boxes = ub.group_items(true_boxes, true_cxs)
    # cx_to_boxes = ub.map_vals(np.array, cx_to_boxes)

    # sort predictions by descending score
    sortx = pred_scores.argsort()[::-1]
    pred_boxes = pred_boxes.take(sortx, axis=0)
    pred_cxs = pred_cxs.take(sortx, axis=0)
    pred_scores = pred_scores.take(sortx, axis=0)

    for px, cx, box, score in zip(sortx, pred_cxs, pred_boxes, pred_scores):
        # For each predicted detection box
        cls_true_idxs = cx_to_idxs.get(cx, [])
        cls_unused = cx_to_unused.get(cx, [])

        ovmax = -np.inf
        ovidx = None
        weight = bg_weight

        if len(cls_true_idxs):
            cls_true_boxes = true_boxes.take(cls_true_idxs, axis=0)
            if true_weights is None:
                cls_true_weights = np.ones(len(cls_true_boxes))
            else:
                cls_true_weights = true_weights.take(cls_true_idxs, axis=0)

            # TODO: make this more efficient
            cls_true_boxes_ = util.Boxes(cls_true_boxes, 'tlbr')
            box_ = util.Boxes(box, 'tlbr')
            overlaps = cls_true_boxes_.ious(box_, bias=bias)

            sortx = overlaps.argsort()[::-1]

            # choose best score by default
            ovidx = sortx[0]
            ovmax = overlaps[ovidx]
            weight = cls_true_weights[ovidx]

            NOT_HACK = True
            # True
            # False
            if NOT_HACK:
                # Only allowed to select matches over a thresh
                is_valid = overlaps > ovthresh
                # Only allowed to select unused annotations
                is_valid = is_valid * cls_unused
                reweighted = overlaps * is_valid.astype(np.float)
                # settings can modify this
                if PREFER_WEIGHTED_TRUTH:
                    # PREFER_WEIGHTED_TRUTH is bugged, doesn't work.
                    # should be trying to ignore difficult gt boxes
                    reweighted = reweighted * cls_true_weights

            if NOT_HACK:
                if np.any(reweighted > 0):
                    # Prefer truth with more weight
                    sortx = reweighted.argsort()[::-1]
                    ovidx = sortx[0]
                    ovmax = overlaps[ovidx]
                else:
                    ovmax = 0
                    weight = bg_weight

        if ovmax > ovthresh and cls_unused[ovidx]:
            # Mark this prediction as a true positive
            y_pred.append(cx)
            y_true.append(cx)
            y_score.append(score)
            y_weight.append(weight)
            cxs.append(cx)
            cls_unused[ovidx] = False

            tx = cls_true_idxs[ovidx]
            y_pxs.append(px)
            y_txs.append(tx)
        else:
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
    for cx, cls_unused in cx_to_unused.items():
        cls_true_idxs = cx_to_idxs.get(cx, [])
        if true_weights is None:
            cls_true_weights = np.ones(len(cls_true_boxes))
        else:
            cls_true_weights = true_weights.take(cls_true_idxs, axis=0)
        for ovidx, flag in enumerate(cls_unused):
            if flag:
                weight = cls_true_weights[ovidx]
                # Mark this prediction as a false negative
                y_pred.append(-1)
                y_true.append(cx)
                y_score.append(0.0)
                y_weight.append(weight)
                cxs.append(cx)

                tx = cls_true_idxs[ovidx]
                px = -1
                y_pxs.append(px)
                y_txs.append(tx)

    y = {
        'pred': y_pred,
        'true': y_true,
        'score': y_score,
        'weight': y_weight,
        'cx': cxs,
        'y_txs': y_txs,  # index into the original true box for this row
        'y_pxs': y_pxs,  # index into the original pred box for this row
    }
    # print('y = {}'.format(ub.repr2(y, nl=1)))
    # y = pd.DataFrame(y)
    return y


def _ave_precision(rec, prec, method='voc2012'):
    """ ap = voc_ap(rec, prec, [use_07_metric])
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

    if False:
        # sklearn metric
        ap = -np.sum(np.diff(rec[::-1]) * np.array(prec[::-1])[:-1])
    return ap


def ave_precisions(y, labels=None, method='voc2012'):
    """

    Args:
        y (pd.DataFrame): pre-measured frames of predictions, truth,
            weight and class.
        method (str): either voc2007 voc2012 or sklearn

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
        ap = _group_metrics(group, method)
        class_aps.append((cx, ap))

    ave_precs = pd.DataFrame(class_aps, columns=['cx', 'ap'])
    return ave_precs


def _group_metrics(group, method):
    # compute metrics on a per class basis
    if group is None:
        return np.nan

    # References [Manning2008] and [Everingham2010] present alternative
    # variants of AP that interpolate the precision-recall curve. Currently,
    # average_precision_score does not implement any interpolated variant
    # http://scikit-learn.org/stable/modules/model_evaluation.html

    # g2 = group[group.weight > 0]
    # prec2, rec2, thresh = sklearn.metrics.precision_recall_curve(
    #     (g2.true > -1).values,
    #     g2.score.values,
    #     sample_weight=g2.weight.values)
    # ap2 = _ave_precision(rec2, prec2, method=method)
    # print('ap2 = {!r}'.format(ap2))

    if method == 'sklearn':
        # In the future, we should simply use the sklearn version
        # which gives nice easy to reproduce results.
        import sklearn.metrics
        df = group
        ap = sklearn.metrics.average_precision_score(
            y_true=(df['true'].values == df['pred'].values).astype(np.int),
            y_score=df['score'].values,
            sample_weight=df['weight'].values,
        )
        return ap

    if False and method == 'voc2007':
        import sklearn.metrics

        # The VOC scoring is weird, and does not conform to sklearn We
        # overcount the number of trues and use unweighted PR curves instead of
        # simply using the weighted variant (that albiet gives lower scores)
        npos = group[group.true >= 0].weight.sum()
        dets = group[group.pred > -1]

        fps, tps, thresholds = sklearn.metrics.ranking._binary_clf_curve(
            (dets.pred == dets.true).values, dets.score.values,
            sample_weight=None)

        precision = tps / (tps + fps)
        recall = tps / npos

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)
        prec3 = np.r_[precision[sl], 1]
        rec3 = np.r_[recall[sl], 0]
        # thre3 = thresholds[sl]

        ap = _ave_precision(rec3, prec3, method=method)
        return ap

    if method == 'voc2007' or method == 'voc2012':
        group = group.sort_values('score', ascending=False)
        # if True:
        #     # ignore "difficult" matches
        #     group = group[group.weight > 0]

        # npos = sum(group.true >= 0)
        npos = group[group.true >= 0].weight.sum()
        dets = group[group.pred > -1]
        if npos > 0 and len(dets) > 0:
            tp = (dets.pred == dets.true).values.astype(np.int)
            fp = 1 - tp
            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)

            eps = np.finfo(np.float64).eps
            rec = 1 if npos == 0 else tp_cum / npos
            prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)

            ap = _ave_precision(rec, prec, method=method)
            return ap
        else:
            if npos == 0:
                return np.nan
            if len(dets) == 0:
                if npos == 0:
                    return np.nan
                return 0.0


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.metrics.detections all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
