"""
DEPRECATED

USE kwcoco.metrics instead!
"""
import numpy as np
import ubelt as ub
import warnings
from scriptconfig.dict_like import DictLike
from netharn.metrics.functional import fast_confusion_matrix


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
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> print(cfsn_vecs.data._pandas())  # xdoctest: IGNORE_WANT
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

    def __init__(cfsn_vecs, data, classes, probs=None):
        cfsn_vecs.data = data
        cfsn_vecs.classes = classes
        cfsn_vecs.probs = probs

    def __nice__(cfsn_vecs):
        return cfsn_vecs.data.__nice__()

    @classmethod
    def demo(cfsn_vecs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> print('cfsn_vecs = {!r}'.format(cfsn_vecs))
            >>> cx_to_binvecs = cfsn_vecs.binarize_ovr()
            >>> print('cx_to_binvecs = {!r}'.format(cx_to_binvecs))
        """
        from netharn.metrics import DetectionMetrics
        dmet = DetectionMetrics.demo(
            nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
        # print('dmet = {!r}'.format(dmet))
        cfsn_vecs = dmet.confusion_vectors()
        cfsn_vecs.data._data = ub.dict_isect(cfsn_vecs.data._data, [
            'true', 'pred', 'score', 'weight',
        ])
        return cfsn_vecs

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
            >>> cfsn_vecs = ConfusionVectors.from_arrays(true=true, probs=probs, classes=classes)
            >>> cfsn_vecs.confusion_matrix()
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
        cfsn_vecs = ConfusionVectors(cfsn_data, probs=probs, classes=classes)
        return cfsn_vecs

    def confusion_matrix(cfsn_vecs, raw=False, compress=False):
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
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> cm = cfsn_vecs.confusion_matrix()
            ...
            >>> print(cm.to_string(float_format=lambda x: '%.2f' % x))
            pred        background  cat_1  cat_2  cat_3
            real
            background        0.00   1.00   1.00   1.00
            cat_1             2.00  12.00   0.00   1.00
            cat_2             2.00   0.00  14.00   1.00
            cat_3             1.00   0.00   1.00  17.00
        """
        data = cfsn_vecs.data

        y_true = data['true'].copy()
        if raw:
            y_pred = data['pred_raw'].copy()
        else:
            y_pred = data['pred'].copy()

        # FIXME: hard-coded background class
        if 'background' in cfsn_vecs.classes:
            bg_idx = cfsn_vecs.classes.index('background')
            y_true[y_true < 0] = bg_idx
            y_pred[y_pred < 0] = bg_idx
        else:
            if np.any(y_true < 0):
                raise IndexError('y_true contains invalid indices')
            if np.any(y_pred < 0):
                raise IndexError('y_pred contains invalid indices')

        matrix = fast_confusion_matrix(
            y_true, y_pred, n_labels=len(cfsn_vecs.classes),
            sample_weight=data.get('weight', None)
        )

        import pandas as pd
        cm = pd.DataFrame(matrix, index=list(cfsn_vecs.classes),
                          columns=list(cfsn_vecs.classes))
        if compress:
            iszero = matrix == 0
            unused = (np.all(iszero, axis=0) & np.all(iszero, axis=1))
            cm = cm[~unused].T[~unused].T
        cm.index.name = 'real'
        cm.columns.name = 'pred'
        return cm

    def coarsen(cfsn_vecs, cxs):
        """
        Creates a coarsened set of vectors
        """
        import ndsampler
        import kwarray
        assert cfsn_vecs.probs is not None, 'need probs'
        if not isinstance(cfsn_vecs.classes, ndsampler.CategoryTree):
            raise TypeError('classes must be a ndsampler.CategoryTree')

        descendent_map = cfsn_vecs.classes.idx_to_descendants_idxs(include_cfsn_vecs=True)
        valid_descendant_mapping = ub.dict_isect(descendent_map, cxs)
        # mapping from current category indexes to the new coarse ones
        # Anything without an explicit key will be mapped to background

        bg_idx = cfsn_vecs.classes.index('background')
        mapping = {v: k for k, vs in valid_descendant_mapping.items() for v in vs}
        new_true = np.array([mapping.get(x, bg_idx) for x in cfsn_vecs.data['true']])
        new_pred = np.array([mapping.get(x, bg_idx) for x in cfsn_vecs.data['pred']])

        new_score = np.array([p[x] for x, p in zip(new_pred, cfsn_vecs.probs)])

        new_y_df = {
            'true': new_true,
            'pred': new_pred,
            'score': new_score,
            'weight': cfsn_vecs.data['weight'],
            'txs': cfsn_vecs.data['txs'],
            'pxs': cfsn_vecs.data['pxs'],
            'gid': cfsn_vecs.data['gid'],
        }
        new_y_df = kwarray.DataFrameArray(new_y_df)
        coarse_cfsn_vecs = ConfusionVectors(new_y_df, cfsn_vecs.classes, cfsn_vecs.probs)
        return coarse_cfsn_vecs

    def binarize_peritem(cfsn_vecs, negative_classes=None):
        """
        Creates a binary representation useful for measuring the performance of
        detectors. It is assumed that scores of "positive" classes should be
        high and "negative" clases should be low.

        Args:
            negative_classes (List[str | int]): list of negative class names or
                idxs, by default chooses any class with a true class index of
                -1. These classes should ideally have low scores.

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> from netharn.metrics import DetectionMetrics
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> class_idxs = list(dmet.classes.node_to_idx.values())
            >>> binvecs = cfsn_vecs.binarize_peritem()
        """
        import kwarray
        # import warnings
        # warnings.warn('binarize_peritem DOES NOT PRODUCE CORRECT RESULTS')

        negative_cidxs = {-1}
        if negative_classes is not None:
            @ub.memoize
            def _lower_classes():
                if cfsn_vecs.classes is None:
                    raise Exception(
                        'classes must be known if negative_classes are strings')
                return [c.lower() for c in cfsn_vecs.classes]
            for c in negative_classes:
                import six
                if isinstance(c, six.string_types):
                    classes = _lower_classes()
                    try:
                        cidx = classes.index(c)
                    except Exception:
                        continue
                else:
                    cidx = int(c)
                negative_cidxs.add(cidx)

        is_false = kwarray.isect_flags(cfsn_vecs.data['true'], negative_cidxs)

        _data = {
            'is_true': ~is_false,
            'pred_score': cfsn_vecs.data['score'],
        }
        extra = ub.dict_isect(_data, [
            'txs', 'pxs', 'gid', 'weight'])
        _data.update(extra)
        bin_data = kwarray.DataFrameArray(_data)
        binvecs = BinaryConfusionVectors(bin_data)
        return binvecs

    def binarize_ovr(cfsn_vecs, mode=1, keyby='name', ignore_classes={'ignore'}):
        """
        Transforms cfsn_vecs into one-vs-rest BinaryConfusionVectors for each category.

        Args:
            mode (int, default=1): 0 for heirarchy aware or 1 for voc like.
                MODE 0 IS PROBABLY BROKEN
            keyby (int | str) : can be cx or name
            ignore_classes (Set[str]): category names to ignore

        Returns:
            OneVsRestConfusionVectors: which behaves like
                Dict[int, BinaryConfusionVectors]: cx_to_binvecs

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> print('cfsn_vecs = {!r}'.format(cfsn_vecs))
            >>> catname_to_binvecs = cfsn_vecs.binarize_ovr(keyby='name')
            >>> print('catname_to_binvecs = {!r}'.format(catname_to_binvecs))

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

        classes = cfsn_vecs.classes
        data = cfsn_vecs.data

        if mode == 0:
            if cfsn_vecs.probs is None:
                raise ValueError('cannot binarize in mode=0 without probs')
            pdist = classes.idx_pairwise_distance()

        cx_to_binvecs = {}
        for cx in range(len(classes)):
            if classes[cx] == 'background' or classes[cx] in ignore_classes:
                continue

            if mode == 0:
                import warnings
                warnings.warn(
                    'THIS CALCLUATION MIGHT BE WRONG. MANY OTHERS '
                    'IN THIS FILE WERE, AND I HAVENT CHECKED THIS ONE YET')

                # Lookup original probability predictions for the class of interest
                new_scores = cfsn_vecs.probs[:, cx]

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
                    'txs': cfsn_vecs.data['txs'],
                    'pxs': cfsn_vecs.data['pxs'],
                    'gid': cfsn_vecs.data['gid'],
                }
                bin_data = kwarray.DataFrameArray(bin_data)

                # Ignore cases where we failed to predict an irrelevant class
                flags = (data['pred'] == -1) & (bin_data['is_true'] == 0)
                bin_data['weight'][flags] = 0
                # bin_data = bin_data.compress(~flags)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes)

            elif mode == 1:
                # More VOC-like, not heirarchy friendly

                if cfsn_vecs.probs is not None:
                    # We know the actual score predicted for this category in
                    # this case.
                    is_true = cfsn_vecs.data['true'] == cx
                    pred_score = cfsn_vecs.probs[:, cx]
                else:
                    import warnings
                    warnings.warn(
                        'Binarize ovr is only approximate if not all probabilities are known')
                    # If we don't know the probabilities for non-predicted
                    # categories then we have to guess.
                    is_true = cfsn_vecs.data['true'] == cx

                    # do we know the actual predicted score for this category?
                    score_is_unknown = data['pred'] != cx
                    pred_score = data['score'].copy()

                    # These scores were for a different class, so assume
                    # other classes were predicted with a uniform prior
                    approx_score = (1 - pred_score[score_is_unknown]) / (len(classes) - 1)

                    # Except in the case where predicted class is -1. In this
                    # case no prediction was actually made (above a threshold)
                    # so the assumed score should be significantly lower, we
                    # conservatively choose zero.
                    unknown_preds = data['pred'][score_is_unknown]
                    approx_score[unknown_preds == -1] = 0

                    pred_score[score_is_unknown] = approx_score

                bin_data = {
                    # is_true denotes if the true class of the item is the
                    # category of interest.
                    'is_true': is_true,
                    'pred_score': pred_score,
                }

                extra = ub.dict_isect(data._data, [
                    'txs', 'pxs', 'gid', 'weight'])
                bin_data.update(extra)

                bin_data = kwarray.DataFrameArray(bin_data)
                bin_cfsn = BinaryConfusionVectors(bin_data, cx, classes)
            cx_to_binvecs[cx] = bin_cfsn

        if keyby == 'cx':
            cx_to_binvecs = cx_to_binvecs
        elif keyby == 'name':
            cx_to_binvecs = ub.map_keys(cfsn_vecs.classes, cx_to_binvecs)
        else:
            raise KeyError(keyby)

        ovr_cfns = OneVsRestConfusionVectors(cx_to_binvecs, cfsn_vecs.classes)
        return ovr_cfns

    def classification_report(cfsn_vecs, verbose=0):
        """
        Build a classification report with various metrics.

        Example:
            >>> from netharn.metrics.confusion_vectors import *  # NOQA
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> report = cfsn_vecs.classification_report(verbose=1)
        """
        from netharn.metrics import clf_report
        y_true = cfsn_vecs.data['true']
        y_pred = cfsn_vecs.data['pred']
        sample_weight = cfsn_vecs.data.get('weight', None)
        target_names = list(cfsn_vecs.classes)
        report = clf_report.classification_report(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            target_names=target_names,
            verbose=verbose,
        )
        return report


class OneVsRestConfusionVectors(ub.NiceRepr):
    """
    Container for multiple one-vs-rest binary confusion vectors

    Attributes:
        cx_to_binvecs
        classes

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> from netharn.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> self = cfsn_vecs.binarize_ovr(keyby='name')
        >>> print('self = {!r}'.format(self))
    """
    def __init__(self, cx_to_binvecs, classes):
        self.cx_to_binvecs = cx_to_binvecs
        self.classes = classes

    def __nice__(self):
        # return ub.repr2(ub.map_vals(len, self.cx_to_binvecs))
        return ub.repr2(self.cx_to_binvecs, strvals=True)

    @classmethod
    def demo(cls):
        cfsn_vecs = ConfusionVectors.demo()
        self = cfsn_vecs.binarize_ovr(keyby='name')
        return self

    def keys(self):
        return self.cx_to_binvecs.keys()

    def __getitem__(self, cx):
        return self.cx_to_binvecs[cx]

    def precision_recall(self, **kwargs):
        perclass = PerClass_PR_Result({
            cx: binvecs.precision_recall(**kwargs)
            for cx, binvecs in self.cx_to_binvecs.items()
        })
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mAP = np.nanmean([item['ap'] for item in perclass.values()])
        return {
            'mAP': mAP,
            'perclass': perclass,
        }

    def roc(self, **kwargs):
        perclass = PerClass_ROC_Result({
            cx: binvecs.roc(**kwargs)
            for cx, binvecs in self.cx_to_binvecs.items()
        })
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mAUC = np.nanmean([item['auc'] for item in perclass.values()])
        return {
            'mAUC': mAUC,
            'perclass': perclass,
        }

    def threshold_curves(self, **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> self = OneVsRestConfusionVectors.demo()
            >>> thresh_result = self.threshold_curves()['perclass']
        """
        perclass = PerClass_Threshold_Result({
            cx: binvecs.threshold_curves(**kwargs)
            for cx, binvecs in self.cx_to_binvecs.items()
        })
        return {
            'perclass': perclass,
        }

    def ovr_classification_report(self):
        raise NotImplementedError


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
    def demo(cls, n=10, p_true=0.5, p_error=0.2, rng=None):
        """
        Create random data for tests

        Example:
            >>> cfsn = BinaryConfusionVectors.demo(n=1000, p_error=0.1)
            >>> print(cfsn.data._pandas())
            >>> roc_info = cfsn.roc()
            >>> pr_info = cfsn.precision_recall()
            >>> print('roc_info = {!r}'.format(roc_info))
            >>> print('pr_info = {!r}'.format(pr_info))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, pnum=(1, 2, 1))
            >>> pr_info.draw()
            >>> kwplot.figure(fnum=1, pnum=(1, 2, 2))
            >>> roc_info.draw()
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        score = rng.rand(n)

        data = kwarray.DataFrameArray({
            'is_true': (score > p_true).astype(np.uint8),
            'pred_score': score,
        })

        flags = rng.rand(n) < p_error
        data['is_true'][flags] = 1 - data['is_true'][flags]

        classes = ['c1', 'c2', 'c3']
        self = cls(data, cx=1, classes=classes)
        return self

    @property
    def catname(self):
        if self.cx is None:
            return None
        return self.classes[self.cx]

    def __nice__(self):
        return ub.repr2({
            'catname': self.catname,
            'data': self.data.__nice__(),
        }, nl=0, strvals=True)

    def __len__(self):
        return len(self.data)

    def draw_distribution(self):
        data = self.data
        y_true = data['is_true'].astype(np.uint8)
        y_score = data['pred_score']

        y_true = y_true.astype(np.bool)

        nbins = 100
        all_freq, xdata = np.histogram(y_score, nbins)
        raw_scores = {
            'true': y_score[y_true],
            'false': y_score[~y_true],
        }
        color = {
            'true': 'dodgerblue',
            'false': 'red'
        }
        ydata = {k: np.histogram(v, bins=xdata)[0]
                 for k, v in raw_scores.items()}
        import kwplot
        return kwplot.multi_plot(xdata=xdata, ydata=ydata, color=color)

    # @ub.memoize_method
    def precision_recall(self, stabalize_thresh=7, stabalize_pad=7, method='sklearn'):
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
        try:
            from sklearn.metrics._ranking import _binary_clf_curve
        except ImportError:
            from sklearn.metrics.ranking import _binary_clf_curve

        data = self.data
        y_true = data['is_true'].astype(np.uint8)
        y_score = data['pred_score']
        sample_weight = data._data.get('weight', None)

        npad = 0
        if len(self) == 0:
            ap = np.nan
            prec = [np.nan]
            rec = [np.nan]
            fps = [np.nan]
            fns = [np.nan]
            tps = [np.nan]
            thresholds = [np.nan]

            realpos_total = 0
            realneg_total = 0
            nsupport = 0
        else:
            if len(self) <= stabalize_thresh:
                # add dummy data to stabalize the computation
                if sample_weight is None:
                    sample_weight = np.ones(len(self))
                npad = stabalize_pad
                y_true, y_score, sample_weight = _stabalilze_data(
                    y_true, y_score, sample_weight, npad=npad)

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

            """
            Notes:
                Apparently, consistent scoring is really hard to get right.

                For detection problems scoring via
                confusion_vectors+sklearn produces noticably different
                results than the VOC method. There are a few reasons for
                this.  The VOC method stops counting true positives after
                all assigned predicted boxes have been counted. It simply
                remembers the amount of original true positives to
                normalize the true positive reate. On the other hand,
                confusion vectors maintains a list of these unassigned true
                boxes and gives them a predicted index of -1 and a score of
                zero. This means that this function sees them as having a
                y_true of 1 and a y_score of 0, which allows the
                scikit-learn fps and tps counts to effectively get up to
                100% recall when the threshold is zero. The VOC method
                simply ignores these and handles them implicitly. The
                problem is that if you remove these from the scikit-learn
                inputs, it wont see the correct number of positives and it
                will incorrectly normalize the recall.  In summary:

                    VOC:
                        * remembers realpos_total
                        * doesn't count unassigned truths as TP when the
                        threshold is zero.

                    CV+SKL:
                        * counts unassigned truths as TP with score=0.
                        * Always ensure tpr=1, ppv=0 and ppv=1, tpr=0 cases
                        exist.
            """
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='invalid .* true_divide')

                if method.startswith('voc'):
                    y_score_ = y_score[y_score > 0]
                    y_true_ = y_true[y_score > 0]
                    fps, tps, _thresholds = _binary_clf_curve(
                        y_true_, y_score_, pos_label=1.0,
                        sample_weight=sample_weight)
                elif method == 'sklearn':
                    fps, tps, _thresholds = _binary_clf_curve(
                        y_true, y_score, pos_label=1.0,
                        sample_weight=sample_weight)
                else:
                    raise KeyError(method)

                # Slight tweak to sklearn.metrics.precision_recall_curve
                fns = realpos_total - tps

                precision = tps / (tps + fps)
                precision[np.isnan(precision)] = 0
                recall = tps / realpos_total

                # stop when full recall attained
                # and reverse the outputs so recall is decreasing
                last_ind = tps.searchsorted(tps[-1])
                sl = slice(last_ind, None, -1)
                prec, rec, thresholds = (
                    np.r_[precision[sl], 1],
                    np.r_[recall[sl], 0],
                    _thresholds[sl])

                if method.startswith('voc'):
                    from netharn.metrics.voc_metrics import _voc_ave_precision
                    ap = _voc_ave_precision(rec[::-1], prec[::-1], method=method)
                elif method == 'sklearn':
                    ap = sklearn.metrics.average_precision_score(
                        y_score=y_score, y_true=y_true,
                        sample_weight=sample_weight)

        prs_info = {
            'ap': ap,
            'ppv': prec,   # (positive predictive value) == (precision)
            'tpr': rec,    # (true positive rate) == (recall)
            'fp_count': fps,
            'tp_count': tps,
            'fn_count': fns,
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
        return PR_Result(prs_info)

    # @ub.memoize_method
    def roc(self, fp_cutoff=None, stabalize_thresh=7, stabalize_pad=7):
        """
        Example:
            >>> self = BinaryConfusionVectors.demo(n=0)
            >>> print('roc = {}'.format(ub.repr2(self.roc())))
            >>> self = BinaryConfusionVectors.demo(n=1, p_true=0.5, p_error=0.5)
            >>> print('roc = {}'.format(ub.repr2(self.roc())))
            >>> self = BinaryConfusionVectors.demo(n=3, p_true=0.5, p_error=0.5)
            >>> print('roc = {}'.format(ub.repr2(self.roc())))
        """
        import sklearn
        import sklearn.metrics  # NOQA
        try:
            from sklearn.metrics._ranking import _binary_clf_curve
        except ImportError:
            from sklearn.metrics.ranking import _binary_clf_curve

        data = self.data
        y_true = data['is_true'].astype(np.uint8)
        y_score = data['pred_score']
        sample_weight = data._data.get('weight', None)

        npad = 0
        if len(self) > 0:
            if len(self) <= stabalize_thresh:
                # add dummy data to stabalize the computation
                if sample_weight is None:
                    sample_weight = np.ones(len(self))
                npad = stabalize_pad
                y_true, y_score, sample_weight = _stabalilze_data(
                    y_true, y_score, sample_weight, npad=npad)

        if sample_weight is None:
            weight = 1
            nsupport = len(y_true) - bool(npad)
        else:
            weight = sample_weight
            nsupport = sample_weight.sum() - bool(npad)

        # y_true[y_true == -1] = 0

        # < TRUCNATED PART >
        # GET ROC CURVES AT A PARTICULAR FALSE POSITIVE COUNT CUTOFF
        # This will let different runs be more comparable

        # Get the total weight (typically number of) positive and negative
        # examples of this class
        realpos_total = (y_true * weight).sum()
        realneg_total = ((1 - y_true) * weight).sum()

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
        elif isinstance(fp_cutoff, str):
            if fp_cutoff == 'num_true':
                fp_cutoff = int(np.ceil(realpos_total))
            else:
                raise KeyError(fp_cutoff)

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

        falsepos_total = trunc_fp_count[-1]

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
        return ROC_Result(roc_info)

    def threshold_curves(self, stabalize_thresh=7, stabalize_pad=7):
        """
        Get statistics (F1, G1, MCC) versus thresholds

        Example:
            >>> self = BinaryConfusionVectors.demo(n=100)
            >>> self.threshold_curves()
        """
        # compute tp, fp, tn, fn at each point
        # compute mcc, f1, g1, etc
        # write plot functions
        info = self._binary_clf_curves(stabalize_thresh, stabalize_pad)

        tp = info['tp_count']
        fp = info['fp_count']
        tn = info['tn_count']
        fn = info['fn_count']

        ppv = tp / (tp + fp)
        tpr = tp / (tp + fn)

        # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        mcc_numer = (tp * tn) - (fp * fn)
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_denom[np.isnan(mcc_denom) | (mcc_denom == 0)] = 1
        info['mcc'] = mcc_numer / mcc_denom

        # https://erotemic.wordpress.com/2019/10/23/closed-form-of-the-mcc-when-tn-inf/
        info['g1'] = np.sqrt(ppv * tpr)

        f1_numer = (2 * ppv * tpr)
        f1_denom = (ppv + tpr)
        f1_denom[f1_denom == 0] = 1
        info['f1'] =  f1_numer / f1_denom

        tnr_denom = (tn + fp)
        tnr_denom[tnr_denom == 0] = 1
        tnr = tn / tnr_denom

        pnv_denom = (tn + fn)
        pnv_denom[pnv_denom == 0] = 1
        npv = tn / pnv_denom

        info['ppv'] = ppv

        info['tpr'] = tpr

        info['acc'] = (tp + tn) / (tp + tn + fp + fn)

        info['bm'] = tpr + tnr - 1  # informedness

        info['mk'] = ppv + npv - 1  # markedness

        keys = ['mcc', 'g1', 'f1', 'acc']
        for key in keys:
            measure = info[key]
            max_idx = measure.argmax()
            best_thresh = info['thresholds'][max_idx]
            best_measure = measure[max_idx]
            best_label = '{}={:0.2f}@{:0.2f}'.format(key, best_measure, best_thresh)
            info['max_{}'.format(key)] = best_label
            info['_max_{}'.format(key)] = (best_measure, best_thresh)

        return Threshold_Result(info)

    def _binary_clf_curves(self, stabalize_thresh=7, stabalize_pad=7):
        """
        Code common to ROC, PR, and threshold measures

        TODO: refactor ROC and PR curves to use this code, perhaps even
        memoizing it.
        """
        try:
            from sklearn.metrics._ranking import _binary_clf_curve
        except ImportError:
            from sklearn.metrics.ranking import _binary_clf_curve
        data = self.data
        y_true = data['is_true'].astype(np.uint8)
        y_score = data['pred_score']
        sample_weight = data._data.get('weight', None)

        npad = 0
        if len(self) == 0:
            fps = [np.nan]
            fns = [np.nan]
            tps = [np.nan]
            thresholds = [np.nan]

            realpos_total = 0
            realneg_total = 0
            nsupport = 0
        else:
            if len(self) <= stabalize_thresh:
                # add dummy data to stabalize the computation
                if sample_weight is None:
                    sample_weight = np.ones(len(self))
                npad = stabalize_pad
                y_true, y_score, sample_weight = _stabalilze_data(
                    y_true, y_score, sample_weight, npad=npad)

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

            fps, tps, thresholds = _binary_clf_curve(
                y_true, y_score, pos_label=1.0,
                sample_weight=sample_weight)

            # Adjust weighted totals to be robust to floating point errors
            if np.isclose(realneg_total, fps[-1]):
                realneg_total = max(realneg_total, fps[-1])
            if np.isclose(realpos_total, tps[-1]):
                realpos_total = max(realpos_total, tps[-1])

        tns = realneg_total - fps
        fns = realpos_total - tps

        info = {
            'fp_count': fps,
            'tp_count': tps,
            'tn_count': tns,
            'fn_count': fns,
            'thresholds': thresholds,
            'realpos_total': realpos_total,
            'realneg_total': realneg_total,
            'nsupport': nsupport,
        }
        if self.cx is not None:
            info.update({
                'cx': self.cx,
                'node': self.classes[self.cx],
            })
        return info


class DictProxy(DictLike):
    """
    Allows an object to proxy the behavior of a dict attribute
    """
    def __getitem__(self, key):
        return self.proxy[key]

    def __setitem__(self, key, value):
        self.proxy[key] = value

    def keys(self):
        return self.proxy.keys()

    def __json__(self):
        return ub.odict(self.proxy)


class ROC_Result(ub.NiceRepr, DictProxy):
    """
    Example:
        >>> self = BinaryConfusionVectors.demo(n=300, p_true=0.1, p_error=0.5).roc()
        >>> print('self = {!r}'.format(self))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=1)
        >>> self.draw()
        >>> kwplot.show_if_requested()
    """
    def __init__(self, roc_info):
        self.proxy = roc_info

    @property
    def catname(self):
        return self.get('node', None)

    def __nice__(self):
        return ub.repr2({
            'auc': self['auc'],
            'nsupport': self['nsupport'],
            'realpos_total': self['realpos_total'],
            'realneg_total': self['realneg_total'],
            'catname': self.get('node', None),
        }, nl=0, precision=4, strvals=True)

    def draw(self, prefix='', **kw):
        """
        Example:
            >>> from netharn.metrics.confusion_vectors import *  # NOQA
            >>> self = BinaryConfusionVectors.demo(n=100).roc()
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> self.draw()
            >>> kwplot.show_if_requested()
        """
        from netharn.metrics import drawing
        return drawing.draw_roc(self, prefix=prefix, **kw)


class PR_Result(ub.NiceRepr, DictProxy):
    """
    Example:
        >>> from netharn.metrics.confusion_vectors import *  # NOQA
        >>> self = BinaryConfusionVectors.demo(n=100, p_error=0.5).precision_recall()
        >>> print('self = {!r}'.format(self))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw()
        >>> kwplot.show_if_requested()
    """
    def __init__(self, roc_info):
        self.proxy = roc_info

    @property
    def catname(self):
        return self.get('node', None)

    def __nice__(self):
        return ub.repr2({
            'ap': self['ap'],
            'nsupport': self['nsupport'],
            'realpos_total': self['realpos_total'],
            'realneg_total': self['realneg_total'],
            'catname': self.get('node', None),
        }, nl=0, precision=4, strvals=True)

    def draw(self, prefix='', **kw):
        from netharn.metrics import drawing
        return drawing.draw_prcurve(self, prefix=prefix, **kw)


class Threshold_Result(ub.NiceRepr, DictProxy):
    """
    Example:
        >>> from netharn.metrics.confusion_vectors import *  # NOQA
        >>> binvecs = BinaryConfusionVectors.demo(n=100, p_error=0.5)
        >>> self = binvecs.threshold_curves()
        >>> print('self = {!r}'.format(self))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> self.draw()
        >>> kwplot.show_if_requested()
    """
    def __init__(self, roc_info):
        self.proxy = roc_info

    @property
    def catname(self):
        return self.get('node', None)

    def __nice__(self):
        return ub.repr2({
            'max_mcc': self['max_mcc'],
            'max_g1': self['max_g1'],
            # 'max_f1': self['max_f1'],
            'nsupport': self['nsupport'],
            'realpos_total': self['realpos_total'],
            'realneg_total': self['realneg_total'],
            'catname': self.get('node', None),
        }, nl=0, precision=4, strvals=True)

    def draw(self, prefix='', **kw):
        from netharn.metrics import drawing
        return drawing.draw_threshold_curves(self, prefix=prefix, **kw)


class PerClass_ROC_Result(ub.NiceRepr, DictProxy):
    """
    """
    def __init__(self, cx_to_roc):
        self.proxy = cx_to_roc

    def __nice__(self):
        return ub.repr2(self.proxy, nl=2, strvals=True)

    def draw(self, prefix='', **kw):
        from netharn.metrics import drawing
        return drawing.draw_perclass_roc(self, prefix=prefix, **kw)


class PerClass_PR_Result(ub.NiceRepr, DictProxy):
    """
    """
    def __init__(self, cx_to_pr):
        self.proxy = cx_to_pr

    def __nice__(self):
        return ub.repr2(self.proxy, nl=2, strvals=True)

    def draw(self, prefix='', **kw):
        from netharn.metrics import drawing
        return drawing.draw_perclass_prcurve(self, prefix=prefix, **kw)


class PerClass_Threshold_Result(ub.NiceRepr, DictProxy):
    """
    """
    def __init__(self, cx_to_info):
        self.proxy = cx_to_info

    def __nice__(self):
        return ub.repr2(self.proxy, nl=2, strvals=True)

    def draw(self, prefix='', **kw):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> # xdoctest: +REQUIRES(module:kwplot)
            >>> cfsn_vecs = ConfusionVectors.demo()
            >>> ovr_cfsn = cfsn_vecs.binarize_ovr(keyby='name')
            >>> self = ovr_cfsn.threshold_curves()['perclass']
            >>> import kwplot
            >>> kwplot.autompl()
            >>> self.draw()
        """
        from netharn.metrics import drawing
        return drawing.draw_perclass_thresholds(self, prefix=prefix, **kw)


def _stabalilze_data(y_true, y_score, sample_weight, npad=7):
    """
    Adds ideally calibrated dummy values to curves with few positive examples.
    This acts somewhat like a Baysian prior and smooths out the curve.
    """
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
    return y_true, y_score, sample_weight

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/metrics/confusion_vectors.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
