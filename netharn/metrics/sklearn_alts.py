"""
DEPRECATED

USE kwcoco.metrics instead!

Faster pure-python versions of sklearn functions that avoid expensive checks
and label rectifications. It is assumed that all labels are consecutive
non-negative integers.
"""
from scipy.sparse import coo_matrix
import numpy as np


def confusion_matrix(y_true, y_pred, n_labels=None, labels=None,
                     sample_weight=None):
    """
    faster version of sklearn confusion matrix that avoids the
    expensive checks and label rectification

    Runs in about 0.7ms

    Returns:
        ndarray: matrix where rows represent real and cols represent pred

    Example:
        >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0,  0, 1])
        >>> y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 1,  1, 1])
        >>> confusion_matrix(y_true, y_pred, 2)
        array([[4, 2],
               [3, 1]])
        >>> confusion_matrix(y_true, y_pred, 2).ravel()
        array([4, 2, 3, 1])

    Benchmarks:
        import ubelt as ub
        y_true = np.random.randint(0, 2, 10000)
        y_pred = np.random.randint(0, 2, 10000)

        n = 1000
        for timer in ub.Timerit(n, bestof=10, label='py-time'):
            sample_weight = [1] * len(y_true)
            confusion_matrix(y_true, y_pred, 2, sample_weight=sample_weight)

        for timer in ub.Timerit(n, bestof=10, label='np-time'):
            sample_weight = np.ones(len(y_true), dtype=np.int)
            confusion_matrix(y_true, y_pred, 2, sample_weight=sample_weight)
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y_true), dtype=np.int)
    if n_labels is None:
        n_labels = len(labels)
    CM = coo_matrix((sample_weight, (y_true, y_pred)),
                    shape=(n_labels, n_labels),
                    dtype=np.int64).toarray()
    return CM


def global_accuracy_from_confusion(cfsn):
    # real is rows, pred is columns
    n_ii = np.diag(cfsn)
    # sum over pred = columns = axis1
    t_i = cfsn.sum(axis=1)
    global_acc = n_ii.sum() / t_i.sum()
    return global_acc


def class_accuracy_from_confusion(cfsn):
    # real is rows, pred is columns
    n_ii = np.diag(cfsn)
    # sum over pred = columns = axis1
    t_i = cfsn.sum(axis=1)
    per_class_acc = (n_ii / t_i).mean()
    class_acc = np.nan_to_num(per_class_acc).mean()
    return class_acc
