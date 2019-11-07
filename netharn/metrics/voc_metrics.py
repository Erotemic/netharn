import sklearn
import warnings
import numpy as np
import ubelt as ub


class VOC_Metrics(object):
    def __init__(self):
        self.recs = {}
        self.cx_to_lines = ub.ddict(list)

    def add_truth(self, true_dets, gid):
        self.recs[gid] = []
        for bbox, cx, weight in zip(true_dets.boxes.to_tlbr().data,
                                    true_dets.class_idxs,
                                    true_dets.weights):
            self.recs[gid].append({
                'bbox': bbox,
                'difficult': weight < .5,
                'name': cx
            })

    def add_predictions(self, pred_dets, gid):
        for bbox, cx, score in zip(pred_dets.boxes.to_tlbr().data,
                                   pred_dets.class_idxs,
                                   pred_dets.scores):
            self.cx_to_lines[cx].append([gid, score] + list(bbox))

    def score(self, ovthresh=0.5, bias=1, method='voc2012'):
        perclass = ub.ddict(dict)
        for cx in self.cx_to_lines.keys():
            lines = self.cx_to_lines[cx]
            classname = cx
            rec, prec, ap = _voc_eval(lines, self.recs, classname,
                                      ovthresh=ovthresh, bias=bias,
                                      method=method)
            perclass[cx]['pr'] = (rec, prec)
            perclass[cx]['ap'] = ap

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
        >>> import netharn as nh
        >>> y2 = nh.util.DataFrameArray(y1)
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


def _voc_eval(lines, recs, classname, ovthresh=0.5, method='voc2012', bias=1):
    """
    Raw replication of matlab implementation of creating assignments and
    the resulting PR-curves and AP. Based on MATLAB code [1].

    References:
        [1] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
    """
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
        warnings.filterwarnings('ignore', message='invalid .* true_divide')

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

        ap = _voc_ave_precision(rec=rec, prec=prec, method=method)
    return rec, prec, ap


def _voc_ave_precision(rec, prec, method='voc2012'):
    """
    Compute AP from precision and recall
    Based on MATLAB code [1,2,3].

    Args:
        rec (ndarray): recall
        prec (ndarray): precision
        method (str): either voc2012 or voc2007

    Returns:
        float: ap: average precision

    References:
        [1] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
        [2] https://github.com/rbgirshick/voc-dpm/blob/master/test/pascal_eval.m
        [3] https://github.com/rbgirshick/voc-dpm/blob/c0b88564bd668bcc6216bbffe96cb061613be768/utils/bootstrap/VOCevaldet_bootstrap.m
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
