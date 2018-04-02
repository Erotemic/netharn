import pandas as pd
import numpy as np
import ubelt as ub


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

    y = pd.DataFrame({
        'pred': y_pred,
        'true': y_true,
        'score': y_score,
        'weight': y_weight,
        'cx': cxs,
    })
    return y


def iou_overlap(true_boxes, pred_box):
    """
    Compute iou of `pred_box` with each `true_box in true_boxes`.
    Return the index and score of the true box with maximum overlap.
    Boxes should be in tlbr format.

    Example:
        >>> true_boxes = np.array([[ 0,  0, 10, 10],
        >>>                        [10,  0, 20, 10],
        >>>                        [20,  0, 30, 10]])
        >>> pred_box = np.array([6, 2, 20, 10, .9])
        >>> ovmax, ovidx = iou_overlap(true_boxes, pred_box)
        >>> print('ovidx = {!r}'.format(ovidx))
        ovidx = 1
    """
    if 0:
        from netharn import util
        # import yolo_utils
        true_boxes = np.array(true_boxes)
        pred_box = np.array(pred_box)
        overlaps = util.bbox_ious(
            true_boxes[:, 0:4].astype(np.float),
            pred_box[None, :][:, 0:4].astype(np.float)).ravel()
    else:
        bb = pred_box
        # intersection
        ixmin = np.maximum(true_boxes[:, 0], bb[0])
        iymin = np.maximum(true_boxes[:, 1], bb[1])
        ixmax = np.minimum(true_boxes[:, 2], bb[2])
        iymax = np.minimum(true_boxes[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (true_boxes[:, 2] - true_boxes[:, 0] + 1.) *
               (true_boxes[:, 3] - true_boxes[:, 1] + 1.) - inters)

        overlaps = inters / uni
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
        >>> ave_precs = ave_precisions(y, use_07_metric=False)
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


# class EvaluateVOC(object):
#     """
#     Example:
#         >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes()
#         >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
#     """
#     def __init__(self, all_true_boxes, all_pred_boxes):
#         self.all_true_boxes = all_true_boxes
#         self.all_pred_boxes = all_pred_boxes

#     @classmethod
#     def perterb_boxes(EvaluateVOC, boxes, perterb_amount=.5, rng=None, cxs=None,
#                       num_classes=None):
#         n = boxes.shape[0]
#         if boxes.shape[0] == 0:
#             boxes = np.array([[10, 10, 50, 50, 1]])

#         # add twice as many boxes,
#         boxes = np.vstack([boxes, boxes])
#         n_extra = len(boxes) - n
#         # perterb the positions
#         xywh = np.hstack([boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2]])
#         scale = np.sqrt(xywh.max()) * perterb_amount
#         pred_xywh = xywh + rng.randn(*xywh.shape) * scale

#         # randomly keep some
#         keep1 = rng.rand(n) >= min(perterb_amount, .5)
#         keep2 = rng.rand(n_extra) < min(perterb_amount, .5)
#         keep = np.hstack([keep1, keep2])

#         if cxs is not None:
#             # randomly change class indexes
#             cxs2 = list(cxs) + list(rng.randint(0, num_classes, n_extra))
#             cxs2 = np.array(cxs2)
#             change = rng.rand(n) < min(perterb_amount, 1.0)
#             cxs2[:n][change] = list(rng.randint(0, num_classes, sum(change)))
#             cxs2 = cxs2[keep]

#         pred = pred_xywh[keep].astype(np.uint8)
#         pred_boxes = np.hstack([pred[:, 0:2], pred[:, 0:2] + pred[:, 2:4]])
#         # give dummy scores
#         pred_boxes = np.hstack([pred_boxes, rng.rand(len(pred_boxes), 1)])
#         if cxs is not None:
#             return pred_boxes, cxs2
#         else:
#             return pred_boxes

#     @classmethod
#     def random_boxes(EvaluateVOC, n=None, c=4, rng=None):
#         if rng is None:
#             rng = np.random
#         if n is None:
#             n = rng.randint(0, 10)
#         xywh = (rng.rand(n, 4) * 100).astype(np.int)
#         tlbr = np.hstack([xywh[:, 0:2], xywh[:, 0:2] + xywh[:, 2:4]])
#         cxs = (rng.rand(n) * c).astype(np.int)
#         return tlbr, cxs

#     @classmethod
#     def demodata_boxes(EvaluateVOC, perterb_amount=.5, rng=0):
#         """
#         Example:
#             >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(100, 0)
#             >>> print(ub.repr2(all_true_boxes, nl=3, precision=2))
#             >>> print(ub.repr2(all_pred_boxes, nl=3, precision=2))
#             >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(0, 0)
#             >>> print(ub.repr2(all_true_boxes, nl=3, precision=2))
#             >>> print(ub.repr2(all_pred_boxes, nl=3, precision=2))
#         """
#         all_true_boxes = [
#             # class 1
#             [
#                 # image 1
#                 [[100, 100, 200, 200, 1]],
#                 # image 2
#                 np.empty((0, 5)),
#                 # image 3
#                 [[0, 10, 10, 20, 1], [10, 10, 20, 20, 1], [20, 10, 30, 20, 1]],
#             ],
#             # class 2
#             [
#                 # image 1
#                 [[0, 0, 100, 100, 1], [0, 0, 50, 50, 1]],
#                 # image 2
#                 [[0, 0, 50, 50, 1], [50, 50, 100, 100, 1]],
#                 # image 3
#                 [[0, 0, 10, 10, 1], [10, 0, 20, 10, 1], [20, 0, 30, 10, 1]],
#             ],
#         ]
#         # convert to numpy
#         for cx, class_boxes in enumerate(all_true_boxes):
#             for gx, boxes in enumerate(class_boxes):
#                 all_true_boxes[cx][gx] = np.array(boxes)

#         # setup perterbubed demo predicted boxes
#         rng = np.random.RandomState(rng)

#         all_pred_boxes = []
#         for cx, class_boxes in enumerate(all_true_boxes):
#             all_pred_boxes.append([])
#             for gx, boxes in enumerate(class_boxes):
#                 pred_boxes = EvaluateVOC.perterb_boxes(boxes, perterb_amount,
#                                                        rng)
#                 all_pred_boxes[cx].append(pred_boxes)

#         return all_true_boxes, all_pred_boxes

#     @classmethod
#     def find_overlap(EvaluateVOC, true_boxes, pred_box):
#         """
#         Compute iou of `pred_box` with each `true_box in true_boxes`.
#         Return the index and score of the true box with maximum overlap.

#         Example:
#             >>> true_boxes = np.array([[ 0,  0, 10, 10, 1],
#             >>>                        [10,  0, 20, 10, 1],
#             >>>                        [20,  0, 30, 10, 1]])
#             >>> pred_box = np.array([6, 2, 20, 10, .9])
#             >>> ovmax, ovidx = EvaluateVOC.find_overlap(true_boxes, pred_box)
#             >>> print('ovidx = {!r}'.format(ovidx))
#             ovidx = 1
#         """
#         if 0:
#             from netharn.models.yolo2.utils import yolo_utils
#             true_boxes = np.array(true_boxes)
#             pred_box = np.array(pred_box)
#             overlaps = yolo_utils.bbox_ious(
#                 true_boxes[:, 0:4].astype(np.float),
#                 pred_box[None, :][:, 0:4].astype(np.float)).ravel()
#         else:
#             bb = pred_box
#             # intersection
#             ixmin = np.maximum(true_boxes[:, 0], bb[0])
#             iymin = np.maximum(true_boxes[:, 1], bb[1])
#             ixmax = np.minimum(true_boxes[:, 2], bb[2])
#             iymax = np.minimum(true_boxes[:, 3], bb[3])
#             iw = np.maximum(ixmax - ixmin + 1., 0.)
#             ih = np.maximum(iymax - iymin + 1., 0.)
#             inters = iw * ih

#             # union
#             uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
#                    (true_boxes[:, 2] - true_boxes[:, 0] + 1.) *
#                    (true_boxes[:, 3] - true_boxes[:, 1] + 1.) - inters)

#             overlaps = inters / uni
#         ovidx = overlaps.argmax()
#         ovmax = overlaps[ovidx]
#         return ovmax, ovidx

#     def compute(self, ovthresh=0.5):
#         """
#         Example:
#             >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(.5)
#             >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
#             >>> ovthresh = 0.5
#             >>> mean_ap = self.compute(ovthresh)[0]
#             >>> print('mean_ap = {:.2f}'.format(mean_ap))
#             mean_ap = 0.36
#             >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(0)
#             >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
#             >>> ovthresh = 0.5
#             >>> mean_ap = self.compute(ovthresh)[0]
#             >>> print('mean_ap = {:.2f}'.format(mean_ap))
#             mean_ap = 1.00
#         """
#         num_classes = len(self.all_true_boxes)
#         ap_list2 = []
#         for cx in range(num_classes):
#             rec, prec, ap = self.eval_class(cx, ovthresh)
#             ap_list2.append(ap)
#         mean_ap2 = np.nanmean(ap_list2)
#         return mean_ap2, ap_list2

#     def eval_class(self, cx, ovthresh=0.5):
#         all_true_boxes = self.all_true_boxes
#         all_pred_boxes = self.all_pred_boxes

#         cls_true_boxes = all_true_boxes[cx]
#         cls_pred_boxes = all_pred_boxes[cx]

#         # Flatten the predicted boxes
#         import pandas as pd
#         flat_pred_boxes = []
#         flat_pred_gxs = []
#         for gx, pred_boxes in enumerate(cls_pred_boxes):
#             flat_pred_boxes.extend(pred_boxes)
#             flat_pred_gxs.extend([gx] * len(pred_boxes))
#         flat_pred_boxes = np.array(flat_pred_boxes)

#         npos = sum([(b.T[4] > 0).sum() for b in cls_true_boxes if len(b)])
#         # npos = sum(map(len, cls_true_boxes))

#         if npos == 0:
#             return [], [], np.nan

#         if len(flat_pred_boxes) > 0:
#             flat_preds = pd.DataFrame({
#                 'box': flat_pred_boxes[:, 0:4].tolist(),
#                 'conf': flat_pred_boxes[:, 4],
#                 'gx': flat_pred_gxs
#             })
#             flat_preds = flat_preds.sort_values('conf', ascending=False)

#             # Keep track of which true boxes have been assigned in this class /
#             # image pair.
#             assign = {}

#             # Greedy assignment for scoring
#             nd = len(flat_preds)
#             tp = np.zeros(nd)
#             fp = np.zeros(nd)
#             # Iterate through predicted bounding boxes in order of descending
#             # confidence
#             for sx, (pred_id, pred) in enumerate(flat_preds.iterrows()):
#                 gx, pred_box = pred[['gx', 'box']]
#                 true_boxes = cls_true_boxes[gx]

#                 ovmax = -np.inf
#                 true_id = None
#                 if len(true_boxes):
#                     true_weights = true_boxes.T[4]
#                     ovmax, ovidx = self.find_overlap(true_boxes, pred_box)
#                     true_id = (gx, ovidx)

#                 if ovmax > ovthresh and true_id not in assign:
#                     if true_weights[ovidx] > 0:
#                         assign[true_id] = pred_id
#                         tp[sx] = 1
#                 else:
#                     fp[sx] = 1

#             # compute precision recall
#             fp = np.cumsum(fp)
#             tp = np.cumsum(tp)

#             if npos == 0:
#                 rec = 1
#             else:
#                 rec = tp / float(npos)
#             # avoid divide by zero in case the first detection matches a difficult
#             # ground truth
#             prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

#             ap = EvaluateVOC.voc_ap(rec, prec, use_07_metric=True)
#             return rec, prec, ap
#         else:
#             if npos == 0:
#                 return [], [], np.nan
#             else:
#                 return [], [], 0.0

#     @classmethod
#     def voc_ap(EvaluateVOC, rec, prec, use_07_metric=False):
#         """ ap = voc_ap(rec, prec, [use_07_metric])
#         Compute VOC AP given precision and recall.
#         If use_07_metric is true, uses the
#         VOC 07 11 point method (default:False).
#         """
#         if use_07_metric:
#             # 11 point metric
#             ap = 0.
#             for t in np.arange(0., 1.1, 0.1):
#                 if np.sum(rec >= t) == 0:
#                     p = 0
#                 else:
#                     p = np.max(prec[rec >= t])
#                 ap = ap + p / 11.
#         else:
#             # correct AP calculation
#             # first append sentinel values at the end
#             mrec = np.concatenate(([0.], rec, [1.]))
#             mpre = np.concatenate(([0.], prec, [0.]))

#             # compute the precision envelope
#             for i in range(mpre.size - 1, 0, -1):
#                 mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

#             # to calculate area under PR curve, look for points
#             # where X axis (recall) changes value
#             i = np.where(mrec[1:] != mrec[:-1])[0]

#             # and sum (\Delta recall) * prec
#             ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#         return ap

#     @classmethod
#     def sanity_check(EvaluateVOC, n=10):
#         """
#         CommandLine:
#             python ~/code/netharn/netharn/data/voc.py EvaluateVOC.sanity_check

#         Example:
#             >>> from netharn.data.voc import *
#             >>> EvaluateVOC.sanity_check(n=3)
#         """
#         import pandas as pd
#         n_images = n
#         ovthresh = 0.8
#         num_classes = 200
#         rng = np.random.RandomState(0)
#         for perterb_amount in [0, .00001, .0001, .0005, .001, .01, .1, .5]:
#             img_ys = []

#             all_true_boxes = [[] for cx in range(num_classes)]
#             all_pred_boxes = [[] for cx in range(num_classes)]

#             for index in range(n_images):
#                 n = rng.randint(0, 50)
#                 true_boxes, true_cxs = EvaluateVOC.random_boxes(n=n,
#                                                                 c=num_classes,
#                                                                 rng=rng)
#                 if len(true_boxes):
#                     # flip every other box to have weight 0
#                     true_boxes = np.hstack([true_boxes, np.ones((len(true_boxes), 1))])
#                     true_boxes[::2, 4] = 0

#                 pred_sboxes, pred_cxs = EvaluateVOC.perterb_boxes(
#                     true_boxes, perterb_amount=perterb_amount,
#                     cxs=true_cxs, rng=rng, num_classes=num_classes)
#                 pred_scores = pred_sboxes[:, 4]
#                 pred_boxes = pred_sboxes[:, 0:4]
#                 y = EvaluateVOC.image_confusions(true_boxes, true_cxs,
#                                                  pred_boxes, pred_scores,
#                                                  pred_cxs, ovthresh=ovthresh)
#                 y['gx'] = index
#                 img_ys.append(y)

#                 # Build format2
#                 for cx in range(num_classes):
#                     all_true_boxes[cx].append(true_boxes[true_cxs == cx])
#                     all_pred_boxes[cx].append(pred_sboxes[pred_cxs == cx])

#             y = pd.concat(img_ys)
#             mean_ap1, ap_list1 = EvaluateVOC.compute_map(y, num_classes)

#             self = EvaluateVOC(all_true_boxes, all_pred_boxes)
#             mean_ap2, ap_list2 = self.compute(ovthresh=ovthresh)
#             print('mean_ap1 = {!r}'.format(mean_ap1))
#             print('mean_ap2 = {!r}'.format(mean_ap2))
#             assert mean_ap2 == mean_ap1
#             print('-------')

#     @classmethod
#     def compute_map(EvaluateVOC, y, num_classes):
#         def group_metrics(group):
#             if group is None:
#                 return np.nan
#             group = group.sort_values('score', ascending=False)
#             npos = sum(group.true >= 0)
#             dets = group[group.pred > -1]
#             if npos == 0:
#                 return np.nan
#             if len(dets) == 0:
#                 if npos == 0:
#                     return np.nan
#                 return 0.0
#             tp = (dets.pred == dets.true).values.astype(np.uint8)
#             fp = 1 - tp
#             fp = np.cumsum(fp)
#             tp = np.cumsum(tp)

#             eps = np.finfo(np.float64).eps
#             if npos == 0:
#                 rec = 1
#             else:
#                 rec = tp / npos
#             prec = tp / np.maximum(tp + fp, eps)

#             ap = EvaluateVOC.voc_ap(rec, prec, use_07_metric=True)
#             return ap

#         # because we use -1 to indicate a wrong prediction we can use max to
#         # determine the class groupings.
#         cx_to_group = dict(iter(y.groupby('cx')))
#         ap_list1 = []
#         for cx in range(num_classes):
#             # for cx, group in cx_to_group.items():
#             group = cx_to_group.get(cx, None)
#             ap = group_metrics(group)
#             ap_list1.append(ap)
#         mean_ap1 = np.nanmean(ap_list1)
#         return mean_ap1, ap_list1
