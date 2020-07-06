"""
DEPRECATED

USE kwcoco.metrics instead!
"""
import warnings
import numpy as np
import ubelt as ub


class VOC_Metrics(ub.NiceRepr):
    """
    API to compute object detection scores using Pascal VOC evaluation method.

    To use, add true and predicted detections for each image and then run the
    `score` function.

    Attributes:
        recs (Dict[int, List[dict]): true boxes for each image.
            maps image ids to a list of records within that image.
            Each record is a tlbr bbox, a difficult flag, and a class name.

        cx_to_lines (Dict[int, List]): VOC formatted prediction preditions.
            mapping from class index to all predictions for that category.
            Each "line" is a list of [
                [<imgid>, <score>, <tl_x>, <tl_y>, <br_x>, <br_y>]].
    """
    def __init__(self, classes=None):
        self.recs = {}
        self.cx_to_lines = ub.ddict(list)
        self.classes = classes

    def __nice__(self):
        info = {
            'n_true_imgs': len(self.recs),
            'n_true_anns': sum(map(len, self.recs.values())),
            'n_pred_anns': sum(map(len, self.cx_to_lines.values())),
            'n_pred_cats': len(self.cx_to_lines),
        }
        return ub.repr2(info)

    def add_truth(self, true_dets, gid):
        self.recs[gid] = []
        true_weights = true_dets.data.get('weights', None)
        if true_weights is None:
            true_weights = [1.0] * len(true_dets)
        for bbox, cx, weight in zip(true_dets.boxes.to_tlbr().data,
                                    true_dets.class_idxs,
                                    true_weights):
            self.recs[gid].append({
                'bbox': bbox,
                'difficult': weight < .5,
                'name': cx
            })

    def add_predictions(self, pred_dets, gid):
        pred_scores = pred_dets.data.get('scores', None)
        if pred_scores is None:
            pred_scores = [1.0] * len(pred_dets)
        for bbox, cx, score in zip(pred_dets.boxes.to_tlbr().data,
                                   pred_dets.class_idxs,
                                   pred_scores):
            voc_line = [gid, score] + list(bbox)
            self.cx_to_lines[cx].append(voc_line)

    def score(self, ovthresh=0.5, bias=1, method='voc2012'):
        """
        Compute VOC scores for every category

        Example:
            >>> from netharn.metrics.detect_metrics import DetectionMetrics
            >>> from netharn.metrics.voc_metrics import *  # NOQA
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=1, nboxes=(0, 100), n_fp=(0, 30), n_fn=(0, 30), nclasses=2, score_noise=0.9)
            >>> self = VOC_Metrics(classes=dmet.classes)
            >>> self.add_truth(dmet.true_detections(0), 0)
            >>> self.add_predictions(dmet.pred_detections(0), 0)
            >>> voc_scores = self.score()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> voc_scores['perclass'].draw()

            kwplot.figure(fnum=2)
            dmet.true_detections(0).draw(color='green', labels=None)
            dmet.pred_detections(0).draw(color='blue', labels=None)
            kwplot.autoplt().gca().set_xlim(0, 100)
            kwplot.autoplt().gca().set_ylim(0, 100)
        """
        from netharn.metrics.confusion_vectors import PR_Result
        from netharn.metrics.confusion_vectors import PerClass_PR_Result
        perclass = {}
        for cx in self.cx_to_lines.keys():
            lines = self.cx_to_lines[cx]
            classname = cx
            roc_info = _voc_eval(lines, self.recs, classname,
                                 ovthresh=ovthresh, bias=bias, method=method)
            roc_info['cx'] = cx
            if self.classes is not None:
                catname = self.classes[cx]
                roc_info.update({
                    'node': catname,
                })
                perclass[catname] = PR_Result(roc_info)
            else:
                perclass[cx] = PR_Result(roc_info)

        perclass = PerClass_PR_Result(perclass)

        mAP = np.nanmean([d['ap'] for d in perclass.values()])
        voc_scores = {
            'mAP': mAP,
            'perclass': perclass,
        }
        return voc_scores


def _pr_curves(y, method='voc2012'):
    """
    Compute a PR curve from a method

    Args:
        y (pd.DataFrame | DataFrameArray): output of detection_confusions

    Returns:
        Tuple[float, ndarray, ndarray]

    Example:
        >>> import pandas as pd
        >>> y1 = pd.DataFrame.from_records([
        >>>     {'pred': 0, 'score': 10.00, 'true': -1, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  1.65, 'true':  0, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  8.64, 'true': -1, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  3.97, 'true':  0, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  1.68, 'true':  0, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  5.06, 'true':  0, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  0.25, 'true':  0, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  1.75, 'true':  0, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  8.52, 'true':  0, 'weight': 1.00},
        >>>     {'pred': 0, 'score':  5.20, 'true':  0, 'weight': 1.00},
        >>> ])
        >>> import kwarray
        >>> y2 = kwarray.DataFrameArray(y1)
        >>> _pr_curves(y2)
        >>> _pr_curves(y1)
    """
    import pandas as pd
    IS_PANDAS = isinstance(y, pd.DataFrame)

    if method not in ['sklearn', 'voc2007', 'voc2012']:
        raise KeyError(method)

    # compute metrics on a per class basis
    if y is None:
        return np.nan, [], []

    # References [Manning2008] and [Everingham2010] present alternative
    # variants of AP that interpolate the precision-recall curve. Currently,
    # average_precision_score does not implement any interpolated variant
    # http://scikit-learn.org/stable/modules/model_evaluation.html
    if method in {'sklearn', 'scikit-learn'}:
        import sklearn
        # In the future, we should simply use the sklearn version
        # which gives nice easy to reproduce results.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid .* true_divide')
            is_correct = (y['true'] == y['pred']).astype(np.int)
            ap = sklearn.metrics.average_precision_score(
                y_true=is_correct, y_score=y['score'],
                sample_weight=y['weight'],
            )
            prec, rec, thresholds = sklearn.metrics.precision_recall_curve(
                is_correct, y['score'], sample_weight=y['weight'],
            )
            return ap, prec, rec
    elif method == 'voc2007' or method == 'voc2012':
        try:
            y = y.sort_values('score', ascending=False)
            # if True:
            #     # ignore "difficult" matches
            #     y = y[y.weight > 0]
            # npos = sum(y.true >= 0)
        except KeyError:
            npos = 0
            dets = []
        else:
            if IS_PANDAS:
                npos = y[y['true'] >= 0].weight.sum()
                dets = y[y['pred'] > -1]
            else:
                npos = y.compress(y['true'] >= 0)['weight'].sum()
                dets = y.compress(y['pred'] > -1)

        if npos > 0 and len(dets) > 0:
            tp = (dets['pred'] == dets['true'])
            fp = 1 - tp
            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)

            eps = np.finfo(np.float64).eps
            rec = 1 if npos == 0 else tp_cum / npos
            prec = tp_cum / np.maximum(tp_cum + fp_cum, eps)

            ap = _voc_ave_precision(rec, prec, method=method)
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


def _voc_eval(lines, recs, classname, ovthresh=0.5, method='voc2012',
              bias=1.0):
    """
    VOC AP evaluation for a single category.

    Args:
        lines (List[list]): VOC formatted predictions.  Each "line" is a list
            of [[<imgid>, <score>, <tl_x>, <tl_y>, <br_x>, <br_y>]].

        recs (Dict[int, List[dict]): true boxes for each image.
            maps image ids to a list of records within that image.
            Each record is a tlbr bbox, a difficult flag, and a class name.

        classname (str): the category to evaluate.

        method (str): code for how the AP is computed.

        bias (float): either 1.0 or 0.0.

    Returns:
        Dict: info about the evaluation containing AP. Contains fp, tp, prec,
            rec,

    Notes:
        Raw replication of matlab implementation of creating assignments and
        the resulting PR-curves and AP. Based on MATLAB code [1].

    References:
        [1] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
    """
    import copy
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

    # Unlike the original implementation our input is presplit
    splitlines = lines
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([x[1] for x in splitlines])
    BB = np.array([[z for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)  #
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    is_tp = np.zeros(nd)
    is_fp = np.zeros(nd)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid .* true_divide')

        # For each prediction
        for d in range(nd):

            # Check if it overlaps any true box.
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
                        # Mark that this true box has been used.
                        is_tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        is_fp[d] = 1.
            else:
                is_fp[d] = 1.

        thresholds = confidence[sorted_ind]
        # compute precision recall
        fp = np.cumsum(is_fp)
        tp = np.cumsum(is_tp)
        fn = npos - tp

        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = _voc_ave_precision(rec=rec, prec=prec, method=method)

    # number of supports is the number of real positives + unassigned preds
    realneg_total = fp[-1]  # number of unassigned predictions
    realpos_total = npos  # number of truth predictions
    nsupport = realneg_total + realpos_total

    info = {
        'fp_count': fp,
        'tp_count': tp,
        'fn_count': fn,
        'tpr': rec,    # (true positive rate) == (recall)
        'ppv': prec,  # (positive predictive value) == (precision)
        'thresholds': thresholds,
        'npos': npos,
        'nsupport': nsupport,
        'realpos_total': realpos_total,
        'realneg_total': realneg_total,
        'ap': ap,
    }
    return info


def _voc_ave_precision(rec, prec, method='voc2012'):
    """
    Compute AP from precision and recall
    Based on MATLAB code in [1]_, [2]_, and [3]_.

    Args:
        rec (ndarray): recall
        prec (ndarray): precision
        method (str): either voc2012 or voc2007

    Returns:
        float: ap: average precision

    References:
        .. [1] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
        .. [2] https://github.com/rbgirshick/voc-dpm/blob/master/test/pascal_eval.m
        .. [3] https://github.com/rbgirshick/voc-dpm/blob/c0b88564bd668bcc6216bbffe96cb061613be768/utils/bootstrap/VOCevaldet_bootstrap.m
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
    elif method == 'sklearn':
        # sklearn metric
        # Note: the voc rec, prec dont extend all the way to 1, so this AUC
        # might not be accurate.
        from sklearn.metrics import auc
        ap = auc(rec, prec)
        # ap = -np.sum(np.diff(rec[::-1]) * np.array(prec[::-1])[:-1])
    else:
        raise KeyError(method)
    return ap
