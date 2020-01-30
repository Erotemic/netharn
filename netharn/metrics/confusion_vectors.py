import numpy as np
import ubelt as ub
import warnings
from .functional import fast_confusion_matrix


class ConfusionVectors(ub.NiceRepr):
    """
    Stores information used to construct a confusion matrix. This includes
    corresponding vectors of predicted labels, true labels, sample weights,
    etc...

    Attributes:
        data (DataFrameArray) : should at least have keys true, pred, weight
        classes (Sequence | CategoryTree): list of category names or category graph
        probs (ndarray, optional): probabilities for each class

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> from netharn.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
        >>> self = dmet.confusion_vectors()
        >>> print(self.data._pandas())  # xdoctest: IGNORE_WANT
            pred_raw  pred  true   score  weight     iou  txs  pxs  gid
        0          2     2     2 10.0000  1.0000  1.0000    0    4    0
        1          2     2     2  7.5025  1.0000  1.0000    1    3    0
        2          1     1     1  5.0050  1.0000  1.0000    2    2    0
        3          3     3    -1  2.5075  1.0000 -1.0000   -1    1    0
        4          2     2    -1  0.0100  1.0000 -1.0000   -1    0    0
        5         -1    -1     2  0.0000  1.0000 -1.0000    3   -1    0
        6         -1    -1     2  0.0000  1.0000 -1.0000    4   -1    0
        7          2     2     2 10.0000  1.0000  1.0000    0    5    1
        8          2     2     2  8.0020  1.0000  1.0000    1    4    1
        9          1     1     1  6.0040  1.0000  1.0000    2    3    1
        ..       ...   ...   ...     ...     ...     ...  ...  ...  ...
        62        -1    -1     2  0.0000  1.0000 -1.0000    7   -1    7
        63        -1    -1     3  0.0000  1.0000 -1.0000    8   -1    7
        64        -1    -1     1  0.0000  1.0000 -1.0000    9   -1    7
        65         1     1    -1 10.0000  1.0000 -1.0000   -1    0    8
        66         1     1     1  0.0100  1.0000  1.0000    0    1    8
        67         3     3    -1 10.0000  1.0000 -1.0000   -1    3    9
        68         2     2     2  6.6700  1.0000  1.0000    0    2    9
        69         2     2     2  3.3400  1.0000  1.0000    1    1    9
        70         3     3    -1  0.0100  1.0000 -1.0000   -1    0    9
        71        -1    -1     2  0.0000  1.0000 -1.0000    2   -1    9
        ...
    """

    def __init__(self, data, classes, probs=None):
        self.data = data
        self.classes = classes
        self.probs = probs

    def __nice__(self):
        return self.data.__nice__()

    @classmethod
    def demo(self):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> self = ConfusionVectors.demo()
            >>> print('self = {!r}'.format(self))
            >>> cx_to_binvecs = self.binarize_ovr()
            >>> print('cx_to_binvecs = {!r}'.format(cx_to_binvecs))
        """
        from netharn.metrics import DetectionMetrics
        dmet = DetectionMetrics.demo(
            nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
        # print('dmet = {!r}'.format(dmet))
        self = dmet.confusion_vectors()
        self.data._data = ub.dict_isect(self.data._data, [
            'true', 'pred', 'score', 'weight',
        ])
        return self

    @classmethod
    def from_arrays(ConfusionVectors, true, pred=None, score=None, weight=None,
                    probs=None, classes=None):
        """
        Construct confusion vector data structure from component arrays

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import kwarray
            >>> classes = ['person', 'vehicle', 'object']
            >>> rng = kwarray.ensure_rng(0)
            >>> true = (rng.rand(10) * len(classes)).astype(np.int)
            >>> probs = rng.rand(len(true), len(classes))
            >>> self = ConfusionVectors.from_arrays(true=true, probs=probs, classes=classes)
            >>> self.confusion_matrix()
            pred     person  vehicle  object
            real
            person        0        0       0
            vehicle       2        4       1
            object        2        1       0
        """
        import kwarray
        if pred is None:
            if probs is not None:
                import ndsampler
                if isinstance(classes, ndsampler.CategoryTree):
                    if not classes.is_mutex():
                        raise Exception('Graph categories require explicit pred')
                # We can assume all classes are mutually exclusive here
                pred = probs.argmax(axis=1)
            else:
                raise ValueError('Must specify pred (or probs)')

        data = {
            'true': true,
            'pred': pred,
            'score': score,
            'weight': weight,
        }

        data = {k: v for k, v in data.items() if v is not None}
        cfsn_data = kwarray.DataFrameArray(data)
        self = ConfusionVectors(cfsn_data, probs=probs, classes=classes)
        return self

    def confusion_matrix(self, raw=False, compress=False):
        """
        Builds a confusion matrix from the confusion vectors.

        Args:
            raw (bool): if True uses 'pred_raw' otherwise used 'pred'

        Returns:
            pd.DataFrame : cm : the labeled confusion matrix
                (Note:  we should write a efficient replacement for
                 this use case. #remove_pandas)

        CommandLine:
            xdoctest -m ~/code/netharn/netharn/metrics/confusion_vectors.py ConfusionVectors.confusion_matrix

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from netharn.metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), n_fn=(0, 1), nclasses=3, cls_noise=.2)
            >>> self = dmet.confusion_vectors()
            >>> cm = self.confusion_matrix()
            >>> print(cm.to_string(float_format=lambda x: '%.2f' % x))
            pred        background  cat_1  cat_2  cat_3
            real
            background           0      1      1      1
            cat_1                2     12      0      1
            cat_2                2      0     14      1
            cat_3                1      0      1     17
        """
        data = self.data

        y_true = data['true'].copy()
        if raw:
            y_pred = data['pred_raw'].copy()
        else:
            y_pred = data['pred'].copy()

        if 'background' in self.classes:
            bg_idx = self.classes.index('background')
            y_true[y_true < 0] = bg_idx
            y_pred[y_pred < 0] = bg_idx
        else:
            if np.any(y_true < 0):
                raise IndexError('y_true contains invalid indices')
            if np.any(y_pred < 0):
                raise IndexError('y_pred contains invalid indices')

        matrix = fast_confusion_matrix(
            y_true, y_pred, n_labels=len(self.classes),
            sample_weight=data.get('weight', None)
        )

        import pandas as pd
        cm = pd.DataFrame(matrix, index=list(self.classes),
                          columns=list(self.classes))
        if compress:
            iszero = matrix == 0
            unused = (np.all(iszero, axis=0) & np.all(iszero, axis=1))
            cm = cm[~unused].T[~unused].T
        cm.index.name = 'real'
        cm.columns.name = 'pred'
        return cm

    def coarsen(self, cxs):
        """
        Creates a coarsened set of vectors
        """
        import ndsampler
        import kwarray
        assert self.probs is not None, 'need probs'
        if not isinstance(self.classes, ndsampler.CategoryTree):
            raise TypeError('classes must be a ndsampler.CategoryTree')

        descendent_map = self.classes.idx_to_descendants_idxs(include_self=True)
        valid_descendant_mapping = ub.dict_isect(descendent_map, cxs)
        # mapping from current category indexes to the new coarse ones
        # Anything without an explicit key will be mapped to background

        bg_idx = self.classes.index('background')
        mapping = {v: k for k, vs in valid_descendant_mapping.items() for v in vs}
        new_true = np.array([mapping.get(x, bg_idx) for x in self.data['true']])
        new_pred = np.array([mapping.get(x, bg_idx) for x in self.data['pred']])

        new_score = np.array([p[x] for x, p in zip(new_pred, self.probs)])

        new_y_df = {
            'true': new_true,
            'pred': new_pred,
            'score': new_score,
            'weight': self.data['weight'],
            'txs': self.data['txs'],
            'pxs': self.data['pxs'],
            'gid': self.data['gid'],
        }
        new_y_df = kwarray.DataFrameArray(new_y_df)
        coarse_cfsn_vecs = ConfusionVectors(new_y_df, self.classes, self.probs)
        return coarse_cfsn_vecs

    def binarize_peritem(self):
        """
        Creates a binary representation of self where the an item is correct if
        the prediction matches the truth, and the score is the confidence in
        the prediction.

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from netharn.metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
            >>> self = dmet.confusion_vectors()
            >>> class_idxs = list(dmet.classes.node_to_idx.values())
            >>> binvecs = self.binarize_peritem()
        """
        import warnings
        import kwarray
        warnings.warn('binarize_peritem DOES NOT PRODUCE CORRECT RESULTS')

        bin_data = kwarray.DataFrameArray({
            'is_true': self.data['true'] == self.data['pred'],
            'pred_score': self.data['score'],
            'weight': self.data['weight'],
            'txs': self.data['txs'],
            'pxs': self.data['pxs'],
            'gid': self.data['gid'],
        })
        binvecs = BinaryConfusionVectors(bin_data)
        return binvecs

    def binarize_ovr(self, mode=1, keyby='name'):
        """
        Transforms self into one-vs-rest BinaryConfusionVectors for each category.

        Args:
            mode (int): 0 for heirarchy aware or 1 for voc like
            keyby : can be cx or name

        Returns:
            OneVsRestConfusionVectors: which behaves like
            Dict[int, BinaryConfusionVectors]: cx_to_binvecs

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> self = ConfusionVectors.demo()
            >>> print('self = {!r}'.format(self))
            >>> catname_to_binvecs = self.binarize_ovr(keyby='name')
            >>> print('catname_to_binvecs = {!r}'.format(cx_to_binvecs))

        Notes:
            Consider we want to measure how well we can classify beagles.

            Given a multiclass confusion vector, we need to carefully select a
            subset. We ignore any truth that is coarser than our current label.
            We also ignore any background predictions on irrelevant classes

            y_true     | y_pred     | score
            -------------------------------
            dog        | dog          <- ignore coarser truths
            dog        | cat          <- ignore coarser truths
            dog        | beagle       <- ignore coarser truths
            cat        | dog
            cat        | cat
            cat        | background   <- ignore failures to predict unrelated classes
            cat        | maine-coon
            beagle     | beagle
            beagle     | dog
            beagle     | background
            beagle     | cat
            Snoopy     | beagle
            Snoopy     | cat
            maine-coon | background    <- ignore failures to predict unrelated classes
            maine-coon | beagle
            maine-coon | cat

            Anything not marked as ignore is counted. We count anything marked
            as beagle or a finer grained class (e.g.  Snoopy) as a positive
            case. All other cases are negative. The scores come from the
            predicted probability of beagle, which must be remembered outside
            the dataframe.
        """
        import kwarray

        classes = self.classes
        data = self.data

        if mode == 0:
            if self.probs is None:
                raise ValueError('cannot binarize in mode=0 without probs')
            pdist = classes.idx_pairwise_distance()

        cx_to_binvecs = {}
        for cx in range(len(classes)):
            if classes[cx] == 'background':
                continue

            if mode == 0:
                import warnings
                warnings.warn(
                    'THIS CALCLUATION MIGHT BE WRONG. MANY OTHERS '
                    'IN THIS FILE WERE, AND I HAVENT CHECKED THIS ONE YET')

                # Lookup original probability predictions for the class of interest
                new_scores = self.probs[:, cx]

                # Determine which truth items have compatible classes
                # Note: we ignore any truth-label that is COARSER than the
                # class-of-interest.
                # E.g: how well do we classify Beagle? -> we should ignore any truth
                # label marked as Dog because it may or may not be a Beagle?
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    dist = pdist[cx]
                    coarser_cxs = np.where(dist < 0)[0]
                    finer_eq_cxs = np.where(dist >= 0)[0]

                is_finer_eq = kwarray.isect_flags(data['true'], finer_eq_cxs)
                is_coarser = kwarray.isect_flags(data['true'], coarser_cxs)

                # Construct a binary data frame to pass to sklearn functions.
                bin_data = {
                    'is_true': is_finer_eq.astype(np.uint8),
                    'pred_score': new_scores,
                    'weight': data['weight'] * (np.float32(1.0) - is_coarser),
                    'txs': self.data['txs'],
                    'pxs': self.data['pxs'],
                    'gid': self.data['gid'],
                }
                bin_data = kwarray.DataFrameArray(bin_data)

                # Ignore cases where we failed to predict an irrelevant class
                flags = (data['pred'] == -1) & (bin_data['is_true'] == 0)
                bin_data['weight'][flags] = 0
                # bin_data = bin_data.compress(~flags)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes)

            elif mode == 1:
                # More VOC-like, not hierarcy friendly

                flags1 = data['pred'] == cx
                flags2 = data['true'] == cx
                flags = flags1 | flags2

                y_group = data.compress(flags)

                bin_data = {
                    # NOTE: THIS USED TO CHECK IF PRED==TRUE BUT
                    # IT REALLY NEEDS TO KNOW IF TRUE IS THE LABEL OF INTEREST
                    # 'is_true': y_group['pred'] == y_group['true'],
                    'is_true': y_group['true'] == cx,

                    'pred_score': y_group['score'],
                }

                extra = ub.dict_isect(y_group._data, [
                    'txs', 'pxs', 'gid', 'weight'])
                bin_data.update(extra)

                bin_data = kwarray.DataFrameArray(bin_data)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes)
            cx_to_binvecs[cx] = bin_cfsn

        if keyby == 'cx':
            cx_to_binvecs = cx_to_binvecs
        elif keyby == 'name':
            cx_to_binvecs = ub.map_keys(self.classes, cx_to_binvecs)
        else:
            raise KeyError(keyby)

        ovr_cfns = OneVsRestConfusionVectors(cx_to_binvecs, self.classes)
        return ovr_cfns

    def classification_report(self):
        """
        Build a classification report with various metrics.
        """
        from netharn.metrics import clf_report
        report = clf_report.classification_report(
            y_true=self.data['true'],
            y_pred=self.data['pred'],
            sample_weight=self.data.get('weight', None),
            target_names=list(self.classes),
        )
        return report


class OneVsRestConfusionVectors(ub.NiceRepr):
    """
    Container for multiple one-vs-rest binary confusion vectors

    Attributes:
        cx_to_binvecs
        classes
    """
    def __init__(self, cx_to_binvecs, classes):
        self.cx_to_binvecs = cx_to_binvecs
        self.classes = classes

    def __nice__(self):
        # return ub.repr2(ub.map_vals(len, self.cx_to_binvecs))
        return ub.repr2(self.cx_to_binvecs)

    def keys(self):
        return self.cx_to_binvecs.keys()

    def __getitem__(self, cx):
        return self.cx_to_binvecs[cx]

    def precision_recall(self):
        perclass = {cx: binvecs.precision_recall()
                    for cx, binvecs in self.cx_to_binvecs.items()}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mAP = np.nanmean([item['ap'] for item in perclass.values()])
        return {
            'mAP': mAP,
            'perclass': perclass,
        }

    def roc(self):
        perclass = {cx: binvecs.roc()
                    for cx, binvecs in self.cx_to_binvecs.items()}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mAUC = np.nanmean([item['auc'] for item in perclass.values()])
        return {
            'mAUC': mAUC,
            'perclass': perclass,
        }


class BinaryConfusionVectors(ub.NiceRepr):
    """
    Stores information about a binary classification problem.
    This is always with respect to a specific class, which is given
    by `cx` and `classes`.

    The `data` DataFrameArray must contain
        `is_true` - if the row is an instance of class `classes[cx]`
        `pred_score` - the predicted probability of class `classes[cx]`, and
        `weight` - sample weight of the example

    Example:
        >>> self = BinaryConfusionVectors.demo(n=10)
        >>> print('self = {!r}'.format(self))
        >>> print('pr = {}'.format(ub.repr2(self.precision_recall())))
        >>> print('roc = {}'.format(ub.repr2(self.roc())))

        >>> self = BinaryConfusionVectors.demo(n=0)
        >>> print('pr = {}'.format(ub.repr2(self.precision_recall())))
        >>> print('roc = {}'.format(ub.repr2(self.roc())))

        >>> self = BinaryConfusionVectors.demo(n=1)
        >>> print('pr = {}'.format(ub.repr2(self.precision_recall())))
        >>> print('roc = {}'.format(ub.repr2(self.roc())))

        >>> self = BinaryConfusionVectors.demo(n=2)
        >>> print('self = {!r}'.format(self))
        >>> print('pr = {}'.format(ub.repr2(self.precision_recall())))
        >>> print('roc = {}'.format(ub.repr2(self.roc())))
    """

    def __init__(self, data, cx=None, classes=None):
        self.data = data
        self.cx = cx
        self.classes = classes

    @classmethod
    def demo(cls, n=10, p_true=0.5, p_error=0.3, rng=None):
        """
        Create random data for tests
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        score = rng.rand(n)

        data = kwarray.DataFrameArray({
            'is_true': score > p_true,
            'pred_score': score,
        })

        flags = rng.rand() > p_error
        data['is_true'][flags] = ~data['is_true'][flags]

        classes = ['c1', 'c2', 'c3']
        self = cls(data, cx=1, classes=classes)
        return self

    @property
    def catname(self):
        return self.classes[self.cx]

    def __nice__(self):
        return ub.repr2({
            'catname': self.catname,
            'data': self.data.__nice__(),
        }, nl=0)

    def __len__(self):
        return len(self.data)

    @ub.memoize_method
    def precision_recall(self, stabalize_thresh=8):
        """
        Example:
            >>> self = BinaryConfusionVectors.demo(n=11)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))
            >>> self = BinaryConfusionVectors.demo(n=7)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))
            >>> self = BinaryConfusionVectors.demo(n=5)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))
            >>> self = BinaryConfusionVectors.demo(n=3)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))
            >>> self = BinaryConfusionVectors.demo(n=2)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))
            >>> self = BinaryConfusionVectors.demo(n=1)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))

            >>> self = BinaryConfusionVectors.demo(n=0)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))

            >>> self = BinaryConfusionVectors.demo(n=1, p_true=0.5, p_error=0.5)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))


            >>> self = BinaryConfusionVectors.demo(n=3, p_true=0.5, p_error=0.5)
            >>> print('precision_recall = {}'.format(ub.repr2(self.precision_recall())))

        """
        import sklearn
        import sklearn.metrics  # NOQA
        data = self.data

        y_true = data['is_true'].astype(np.uint8)
        y_score = data['pred_score']
        sample_weight = data._data.get('weight', None)

        npad = 0
        if len(self) == 0:
            ap = np.nan
            prec = [np.nan]
            rec = [np.nan]
            thresholds = [np.nan]

        else:
            if len(self) <= stabalize_thresh:
                # add dummy data to stabalize the computation
                if sample_weight is None:
                    sample_weight = np.ones(len(self))

                npad = 7
                min_score = y_score.min()
                max_score = y_score.max()

                if max_score <= 1.0 and min_score >= 0.0:
                    max_score = 1.0
                    min_score = 0.0

                pad_true = np.ones(npad, dtype=np.uint8)
                pad_true[:npad // 2] = 0

                pad_score = np.linspace(min_score, max_score, num=npad,
                                        endpoint=True)
                pad_weight = np.exp(np.linspace(2.7, .01, npad))
                pad_weight /= pad_weight.sum()

                y_true = np.hstack([y_true, pad_true])
                y_score = np.hstack([y_score, pad_score])
                sample_weight = np.hstack([sample_weight, pad_weight])

            metric_kw = {
                'y_true': y_true,
                'sample_weight': sample_weight,
            }
            # print('metric_kw = {}'.format(ub.repr2(metric_kw, nl=1)))
            # print('y_score = {!r}'.format(y_score))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid .* true_divide')
                ap = sklearn.metrics.average_precision_score(
                    y_score=y_score, **metric_kw)
                prec, rec, thresholds = sklearn.metrics.precision_recall_curve(
                    probas_pred=y_score, **metric_kw)

        # FIXME
        # USING true == pred IS WRONG.
        # when pred=-1 and true=0 the score=0, but is_true=False.
        # THIS CAUSES the total number of TRUE sklearn vecs to be incorrect

        # Get the total weight (typically number of) positive and negative
        # examples of this class
        if sample_weight is None:
            weight = 1
            nsupport = len(y_true) - bool(npad)
        else:
            weight = sample_weight
            nsupport = sample_weight.sum() - bool(npad)

        realpos_total = (y_true * weight).sum()
        realneg_total = ((1 - y_true) * weight).sum()

        prs_info = {
            'ap': ap,
            'ppv': prec,   # (positive predictive value) == (precision)
            'tpr': rec,    # (true positive rate) == (recall)
            'thresholds': thresholds,
            'nsupport': nsupport,
            'realpos_total': realpos_total,
            'realneg_total': realneg_total,
        }
        if self.cx is not None:
            prs_info.update({
                'cx': self.cx,
                'node': self.classes[self.cx],
            })
        return prs_info

    @ub.memoize_method
    def roc(self, fp_cutoff=None):
        import sklearn
        import sklearn.metrics  # NOQA
        try:
            from sklearn.metrics._ranking import _binary_clf_curve
        except ImportError:
            from sklearn.metrics.ranking import _binary_clf_curve
        data = self.data
        y_true = data['is_true']
        y_score = data['pred_score']
        sample_weight = data._data.get('weight', None)

        # y_true[y_true == -1] = 0

        # < TRUCNATED PART >
        # GET ROC CURVES AT A PARTICULAR FALSE POSITIVE COUNT CUTOFF
        # This will let different runs be more comparable

        # Get the total weight (typically number of) positive and negative
        # examples of this class
        realpos_total = (data['is_true'] * data._data.get('weight', 1)).sum()
        realneg_total = ((1 - data['is_true']) * data._data.get('weight', 1)).sum()
        if sample_weight is None:
            nsupport = len(self)
        else:
            nsupport = sample_weight.sum()

        if len(self) == 0:
            fp_count = np.array([np.nan])
            tp_count = np.array([np.nan])
            count_thresholds = np.array([np.nan])
        else:
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

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid .* true_divide')
            trunc_tpr = trunc_tp_count / realpos_total
            trunc_fpr = trunc_fp_count / falsepos_total
            try:
                trunc_auc = sklearn.metrics.auc(trunc_fpr, trunc_tpr)
            except ValueError:
                # At least 2 points are needed to compute area under curve, but x.shape = 1
                trunc_auc = np.nan
        # < /TRUCNATED PART >
        roc_info = {
            'fp_cutoff': fp_cutoff,
            'realpos_total': realpos_total,
            'realneg_total': realneg_total,
            'nsupport': nsupport,
            'tpr': trunc_tpr,
            'fpr': trunc_fpr,
            'fp_count': trunc_fp_count,
            'tp_count': trunc_tp_count,
            'thresholds': trunc_thresholds,
            'auc': trunc_auc,
        }
        if self.cx is not None:
            roc_info.update({
                'cx': self.cx,
                'node': self.classes[self.cx],
            })
        return roc_info
