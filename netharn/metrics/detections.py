import pandas as pd
import numpy as np
import ubelt as ub
from netharn import util
from netharn.util import profiler


@profiler.profile
def detection_confusions(true_boxes, true_cxs, true_weights, pred_boxes,
                         pred_scores, pred_cxs, bg_weight=1.0, ovthresh=0.5,
                         bg_cls=-1):
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

    Returns:
        pd.DataFrame: with relevant clf information

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
        >>> pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
           cx  pred  score  true  weight
        0   1     1 0.5000     1       1.0
        1   0     0 0.5000    -1       1.0
        2   0    -1 0.0000     0       1.0
        3   1    -1 0.0000     1       0.9
    """
    y_pred = []
    y_true = []
    y_score = []
    y_weight = []
    cxs = []

    if bg_weight is None:
        bg_weight = 1.0

    # Group true boxes by class
    # Keep track which true boxes are unused / not assigned
    cx_to_idxs = ub.group_items(range(len(true_cxs)), true_cxs)
    cx_to_unused = {cx: [True] * len(idxs)
                    for cx, idxs in cx_to_idxs.items()}

    # cx_to_boxes = ub.group_items(true_boxes, true_cxs)
    # cx_to_boxes = ub.map_vals(np.array, cx_to_boxes)

    # sort predictions by score
    sortx = pred_scores.argsort()[::-1]
    pred_boxes  = pred_boxes.take(sortx, axis=0)
    pred_cxs    = pred_cxs.take(sortx, axis=0)
    pred_scores = pred_scores.take(sortx, axis=0)
    for cx, box, score in zip(pred_cxs, pred_boxes, pred_scores):
        cls_true_idxs = cx_to_idxs.get(cx, [])

        ovmax = -np.inf
        ovidx = None
        weight = bg_weight

        if len(cls_true_idxs):
            cls_true_boxes = true_boxes.take(cls_true_idxs, axis=0)
            ovmax, ovidx = iou_overlap(cls_true_boxes, box)
            if true_weights is None:
                weight = 1.0
            else:
                true_idx = cls_true_idxs[ovidx]
                weight = true_weights[true_idx]
            unused = cx_to_unused[cx]

        if ovmax > ovthresh and unused[ovidx]:
            # Mark this prediction as a true positive
            if weight > 0:
                # Ignore matches to truth with weight 0 (difficult cases)
                y_pred.append(cx)
                y_true.append(cx)
                y_score.append(score)
                y_weight.append(weight)
                cxs.append(cx)
                unused[ovidx] = False
        else:
            # Mark this prediction as a false positive
            y_pred.append(cx)
            y_true.append(bg_cls)  # use -1 as background ignore class
            y_score.append(score)
            y_weight.append(weight)
            cxs.append(cx)

    # Mark true boxes we failed to predict as false negatives
    for cx, unused in cx_to_unused.items():
        for ovidx, flag in enumerate(unused):
            if flag:
                if true_weights is None:
                    weight = 1.0
                else:
                    cls_true_idxs = cx_to_idxs.get(cx, [])
                    true_idx = cls_true_idxs[ovidx]
                    weight = true_weights[true_idx]
                # if it has a nonzero weight
                if  weight > 0:
                    # Mark this prediction as a false negative
                    y_pred.append(-1)
                    y_true.append(cx)
                    y_score.append(0.0)
                    y_weight.append(weight)
                    cxs.append(cx)

    y = {
        'pred': y_pred,
        'true': y_true,
        'score': y_score,
        'weight': y_weight,
        'cx': cxs,
    }
    # y = pd.DataFrame()
    return y


def iou_overlap(true_boxes, pred_box):
    """
    Compute iou of `pred_box` with each `true_box in true_boxes`.
    Return the index and score of the true box with maximum overlap.
    Boxes should be in tlbr format.

    CommandLine:
        python -m netharn.metrics.detections iou_overlap

    Example:
        >>> true_boxes = np.array([[ 0,  0, 10, 10],
        >>>                        [10,  0, 20, 10],
        >>>                        [20,  0, 30, 10]])
        >>> pred_box = np.array([6, 2, 20, 10, .9])
        >>> ovmax, ovidx = iou_overlap(true_boxes, pred_box)
        >>> print('ovmax, ovidx = {:.2f}, {}'.format(ovmax, ovidx))
        ovmax, ovidx = 0.63, 1
    """
    # import yolo_utils
    true_boxes = np.array(true_boxes)
    pred_box = np.array(pred_box)
    overlaps = util.box_ious(
        true_boxes[:, 0:4].astype(np.float),
        pred_box[None, :][:, 0:4].astype(np.float), bias=1).ravel()
    ovidx = overlaps.argmax()
    ovmax = overlaps[ovidx]
    return ovmax, ovidx


def _ave_precision(rec, prec, use_07_metric=False):
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


def ave_precisions(y, labels=None, use_07_metric=False):
    """
    Example:
        >>> y_true = [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,-1,-1,-1]
        >>> y_pred = [1, 3,-1, 2,-1, 1, 2, 1,-1, 3, 3, 3, 3, 3, 4, 4, 3, 1, 1]
        >>> y = pd.DataFrame.from_dict({
        >>>     'true': y_true,
        >>>     'pred': y_pred,
        >>> })
        >>> y['score'] = 0.99
        >>> y['weight'] = 1.0
        >>> ave_precs = ave_precisions(y, use_07_metric=True)
        >>> print(ave_precs)  # xdoc: +IGNORE_WANT
           cx     ap
        0   1 0.2727
        1   2 0.1364
        2   3 1.0000
        3   4 1.0000
        >>> mAP = np.nanmean(ave_precs['ap'])
        >>> print('mAP = {:.4f}'.format(mAP))
        mAP = 0.6023
        >>> # -----------------
        >>> ave_precs = ave_precisions(y, use_07_metric=False)
        >>> print(ave_precs)  # xdoc: +IGNORE_WANT
           cx     ap
        0   1 0.2500
        1   2 0.1000
        2   3 1.0000
        3   4 1.0000
        >>> mAP = np.nanmean(ave_precs['ap'])
        >>> print('mAP = {:.4f}'.format(mAP))
        mAP = 0.5875
    """
    def group_metrics(group):
        # compute metrics on a per class basis
        if group is None:
            return np.nan
        group = group.sort_values('score', ascending=False)
        npos = sum(group.true >= 0)
        dets = group[group.pred > -1]
        if npos == 0:
            return np.nan
        if len(dets) == 0:
            if npos == 0:
                return np.nan
            return 0.0
        tp = (dets.pred == dets.true).values.astype(np.uint8)
        fp = 1 - tp
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        eps = np.finfo(np.float64).eps
        if npos == 0:
            rec = 1
        else:
            rec = tp / npos
        prec = tp / np.maximum(tp + fp, eps)

        ap = _ave_precision(rec, prec, use_07_metric=use_07_metric)
        return ap

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
        ap = group_metrics(group)
        class_aps.append((cx, ap))

    ave_precs = pd.DataFrame(class_aps, columns=['cx', 'ap'])
    return ave_precs


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.metrics.detections all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
