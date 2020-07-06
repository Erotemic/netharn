"""
DEPRECATED

USE kwcoco.metrics instead!
"""
import numpy as np
import warnings
from scipy.sparse import coo_matrix


def fast_confusion_matrix(y_true, y_pred, n_labels, sample_weight=None):
    """
    faster version of sklearn confusion matrix that avoids the
    expensive checks and label rectification

    Args:
        y_true (ndarray[int]): ground truth class label for each sample
        y_pred (ndarray[int]): predicted class label for each sample
        n_labels (int): number of labels
        sample_weight (ndarray[int|float]): weight of each sample

    Returns:
        ndarray[int64|float64, dim=2]:
            matrix where rows represent real and cols represent pred and the
            value at each cell is the total amount of weight

    Example:
        >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0,  0, 1])
        >>> y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1,  1, 1])
        >>> fast_confusion_matrix(y_true, y_pred, 2)
        array([[4, 2],
               [3, 1]])
        >>> fast_confusion_matrix(y_true, y_pred, 2).ravel()
        array([4, 2, 3, 1])
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true), dtype=np.uint8)
    # The accumulation dtype needs to have 64bits to avoid overflow
    dtype = np.float64 if sample_weight.dtype.kind == 'f' else np.int64
    matrix = coo_matrix((sample_weight, (y_true, y_pred)),
                        shape=(n_labels, n_labels),
                        dtype=dtype).toarray()
    return matrix


def _truncated_roc(y_df, bg_idx=-1, fp_cutoff=None):
    """
    Computes truncated ROC info
    """
    import sklearn
    try:
        from sklearn.metrics._ranking import _binary_clf_curve
    except ImportError:
        from sklearn.metrics.ranking import _binary_clf_curve
    y_true = (y_df['true'] == y_df['pred'])
    y_score = y_df['score']
    sample_weight = y_df['weight']

    # y_true[y_true == -1] = 0

    # < TRUCNATED PART >
    # GET ROC CURVES AT A PARTICULAR FALSE POSITIVE COUNT CUTOFF
    # This will let different runs be more comparable
    realpos_total = sample_weight[(y_df['txs'] >= 0)].sum()

    fp_count, tp_count, count_thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=1, sample_weight=sample_weight)

    if len(count_thresholds) > 0 and count_thresholds[-1] == 0:
        # Chop off the last entry where it will jump
        count_thresholds = count_thresholds[:-1]
        tp_count = tp_count[:-1]
        fp_count = fp_count[:-1]

    # Cutoff the curves at a comparable point
    if fp_cutoff is None:
        fp_cutoff = np.inf
    idxs = np.where(fp_count > fp_cutoff)[0]
    if len(idxs) == 0:
        idx = len(fp_count)
    else:
        idx = idxs[0]
    trunc_fp_count = fp_count[:idx]
    trunc_tp_count = tp_count[:idx]
    trunc_thresholds = count_thresholds[:idx]

    # if the cuttoff was not reached, horizontally extend the curve
    # This will hurt the scores (aka we may be bias against small
    # scenes), but this will ensure that big scenes are comparable
    if len(fp_count) == 0:
        trunc_fp_count = np.array([fp_cutoff])
        trunc_tp_count = np.array([0])
        trunc_thresholds = np.array([0])
        # THIS WILL CAUSE AUC TO RAISE AN ERROR IF IT GETS HIT
    elif fp_count[-1] < fp_cutoff and np.isfinite(fp_cutoff):
        trunc_fp_count = np.hstack([trunc_fp_count, [fp_cutoff]])
        trunc_tp_count = np.hstack([trunc_tp_count, [trunc_tp_count[-1]]])
        trunc_thresholds = np.hstack([trunc_thresholds, [0]])

    falsepos_total = trunc_fp_count[-1]  # is this right?

    trunc_tpr = trunc_tp_count / realpos_total
    trunc_fpr = trunc_fp_count / falsepos_total
    trunc_auc = sklearn.metrics.auc(trunc_fpr, trunc_tpr)
    # < /TRUCNATED PART >
    roc_info = {
        'fp_cutoff': fp_cutoff,
        'realpos_total': realpos_total,
        'tpr': trunc_tpr,
        'fpr': trunc_fpr,
        'fp_count': trunc_fp_count,
        'tp_count': trunc_tp_count,
        'thresholds': trunc_thresholds,
        'auc': trunc_auc,
    }
    return roc_info


def _pr_curves(y):
    """
    Compute a PR curve from a method

    Args:
        y (pd.DataFrame | DataFrameArray): output of detection_confusions

    Returns:
        Tuple[float, ndarray, ndarray]

    Example:
        >>> # xdoctest: +REQUIRES(module:sklearn)
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
        >>> import kwarray
        >>> y2 = kwarray.DataFrameArray(y1)
        >>> _pr_curves(y2)
        >>> _pr_curves(y1)
    """
    import sklearn
    # compute metrics on a per class basis
    if y is None:
        return np.nan, [], []

    # References [Manning2008] and [Everingham2010] present alternative
    # variants of AP that interpolate the precision-recall curve. Currently,
    # average_precision_score does not implement any interpolated variant
    # http://scikit-learn.org/stable/modules/model_evaluation.html

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


def _average_precision(tpr, ppv):
    """
    Compute average precision of a binary PR curve. This is simply the area
    under the curve.

    Args:
        tpr (ndarray): true positive rate - aka recall
        ppv (ndarray): positive predictive value - aka precision
    """
    # The average precision is simply the area under the PR curve.
    xdata = tpr
    ydata = ppv
    if xdata[0] > xdata[-1]:
        xdata = xdata[::-1]
        ydata = ydata[::-1]
    # Note: we could simply use sklearn.metrics.auc, which has more robust
    # checks.
    ap = np.trapz(y=ydata, x=xdata)
    return ap
