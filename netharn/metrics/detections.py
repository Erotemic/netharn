import pandas as pd
import numpy as np
import ubelt as ub
from netharn import util
from netharn.util import profiler


class ScoreDets:
    def __init__(self, true_dets, pred_dets, bg_weight=1.0, ovthresh=0.5,
                 bg_cls=-1, bias=0.0):
        self.true_dets = true_dets
        self.pred_dets = pred_dets

    def score_netharn():
        pass

    def score_voc():
        pass

    def score_coco():
        pass


@profiler.profile
def assign_dets(true_boxes, true_cxs, true_weights, pred_boxes, pred_scores, pred_cxs, bg_weight=1.0, ovthresh=0.5, bg_cls=-1, bias=0.0) -> dict:
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
        pd.DataFrame: with relevant clf information

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
           pred  true  score  weight  cx  y_txs  y_pxs
        0     1     1 0.5000  1.0000   1      3      2
        1     0    -1 0.5000  1.0000   0     -1      1
        2     0     0 0.5000  0.0000   0      1      0
        3    -1     0 0.0000  1.0000   0      0     -1
        4    -1     1 0.0000  0.9000   1      2     -1

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

    # Group true boxes by class
    # Keep track which true boxes are unused / not assigned
    cx_to_idxs = ub.group_items(range(len(true_cxs)), true_cxs)

    # cx_to_boxes = ub.group_items(true_boxes, true_cxs)
    # cx_to_boxes = ub.map_vals(np.array, cx_to_boxes)

    # sort predictions by descending score
    spred_sortx = pred_scores.argsort()[::-1]
    spred_boxes = pred_boxes.take(spred_sortx, axis=0)
    spred_cxs = pred_cxs.take(spred_sortx, axis=0)
    spred_scores = pred_scores.take(spred_sortx, axis=0)

    # For each predicted detection box
    # Allow it to match the truth of a particular class
    for px, cx, box, score in zip(spred_sortx, spred_cxs, spred_boxes, spred_scores):
        cls_true_idxs = cx_to_idxs.get(cx, [])

        ovmax = -np.inf
        ovidx = None
        weight = bg_weight
        tx = None  # we will set this to the index of the assignd gt

        if len(cls_true_idxs):
            cls_true_boxes = true_boxes.take(cls_true_idxs, axis=0)
            cls_true_weights = true_weights.take(cls_true_idxs, axis=0)

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
        'y_txs': y_txs,  # index into the original true box for this row
        'y_pxs': y_pxs,  # index into the original pred box for this row
    }
    # print('y = {}'.format(ub.repr2(y, nl=1)))
    # y = pd.DataFrame(y)
    return y


def _ave_precision(rec, prec, method='voc2012') -> float:
    """ Compute AP from precision and recall


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

    if False:
        # sklearn metric
        ap = -np.sum(np.diff(rec[::-1]) * np.array(prec[::-1])[:-1])
    return ap


def score_detection_assignment(y, labels=None, method='voc2012') -> pd.DataFrame:
    """ Measures scores of predicted detections assigned to groundtruth objects

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
        ap = pr_curves(group, method)
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
        return np.nan

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
        return ap

    if method == 'voc2007' or method == 'voc2012':
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
    return ap, prec, rec


ave_precisions = score_detection_assignment
detection_confusions = assign_dets


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.metrics.detections all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
