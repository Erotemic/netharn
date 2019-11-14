"""
TODO:
    - [ ] _fast_pdist_priority: Look at absolute difference in sibling entropy
        when deciding whether to go up or down in the tree.

    - [ ] medschool applications true-pred matching (applicant proposing) fast
        algorithm.

    - [ ] Maybe looping over truth rather than pred is faster? but it makes you
        have to combine pred score / ious, which is weird.

    - [x] preallocate ndarray and use hstack to build confusion vectors?
        - doesn't help

    - [ ] relevant classes / classes / classes-of-interest we care about needs
        to be a first class member of detection metrics.
"""
import warnings
import networkx as nx
import numpy as np
import ubelt as ub


def _assign_confusion_vectors(true_dets, pred_dets, bg_weight=1.0,
                              ovthresh=0.5, bg_cidx=-1, bias=0.0, classes=None,
                              compat='all', prioritize='iou'):
    """
    Create confusion vectors for detections by assigning to ground true boxes

    Given predictions and truth for an image return (y_pred, y_true,
    y_score), which is suitable for sklearn classification metrics

    Args:
        true_dets (Detections):
            groundtruth with boxes, classes, and weights

        pred_dets (Detections):
            predictions with boxes, classes, and scores

        ovthresh (float): overlap threshold

        bg_weight (ndarray): weight of background predictions
          (default=1)

        bias : for computing overlap either 1 or 0

        compat (str): can be ('mutex' | 'all' | 'ancestors').
            determines which pred boxes are allowed to match which true boxes.
            If 'mutex', then pred boxes can only match true boxes of the same
            class. If 'ancestors', then pred boxes can match true boxes that
            match or have a coarser label. If 'all', then any pred can match
            any true, regardless of if the predicted class label matches or
            not.

        prioritize (str): can be ('iou' | 'class' | 'correct'}
            determines which class to assign to if mutiple boxes overlap.
            if prioritize is iou, then the true box with maximum iou (above
            ovthresh) will be chosen. If prioritize is class, then it will
            prefer matching a compatible class above a higher iou. If
            prioritize is correct, then ancestors of the true class are
            preferred over descendents of the true class, over unreleated
            classes.

    TODO:
        - [ ] This is a bottleneck function. An implementation in C / C++ /
        Cython would likely improve the overall system.

    Returns:
        dict: with relevant confusion vectors. This keys of this dict can be
            interpreted as columns of a data frame. The `txs` / `pxs` columns
            represent the indexes of the true / predicted annotations that were
            assigned as matching. Additionally each row also contains the true
            and predicted class index, the predicted score, the true weight and
            the iou of the true and predicted boxes. A `txs` value of -1 means
            that the predicted box was not assigned to a true annotation and a
            `pxs` value of -1 means that the true annotation was not assigne to
            any predicted annotation.

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> import pandas as pd
        >>> import netharn as nh
        >>> import kwimage
        >>> # Given a raw numpy representation construct Detection wrappers
        >>> true_dets = kwimage.Detections(
        >>>     boxes=kwimage.Boxes(np.array([
        >>>         [ 0,  0, 10, 10], [10,  0, 20, 10],
        >>>         [10,  0, 20, 10], [20,  0, 30, 10]]), 'tlbr'),
        >>>     weights=np.array([1, 0, .9, 1]),
        >>>     class_idxs=np.array([0, 0, 1, 2]))
        >>> pred_dets = kwimage.Detections(
        >>>     boxes=kwimage.Boxes(np.array([
        >>>         [6, 2, 20, 10], [3,  2, 9, 7],
        >>>         [3,  9, 9, 7],  [3,  2, 9, 7],
        >>>         [2,  6, 7, 7],  [20,  0, 30, 10]]), 'tlbr'),
        >>>     scores=np.array([.5, .5, .5, .5, .5, .5]),
        >>>     class_idxs=np.array([0, 0, 1, 2, 0, 1]))
        >>> bg_weight = 1.0
        >>> compat = 'all'
        >>> ovthresh = 0.5
        >>> bias = 0.0
        >>> import ndsampler
        >>> classes = ndsampler.CategoryTree.from_mutex(list(range(3)))
        >>> bg_cidx = -1
        >>> y = _assign_confusion_vectors(true_dets, pred_dets, bias=bias,
        >>>                               bg_weight=bg_weight, ovthresh=ovthresh,
        >>>                               compat=compat)
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
           pred_raw  pred  true  score  weight     iou  txs  pxs
        0         1     1     2 0.5000  1.0000  1.0000    3    5
        1         0     0    -1 0.5000  1.0000 -1.0000   -1    4
        2         2     2    -1 0.5000  1.0000 -1.0000   -1    3
        3         1     1    -1 0.5000  1.0000 -1.0000   -1    2
        4         0     0    -1 0.5000  1.0000 -1.0000   -1    1
        5         0     0     0 0.5000  0.0000  0.6061    1    0
        6        -1    -1     0 0.0000  1.0000 -1.0000    0   -1
        7        -1    -1     1 0.0000  0.9000 -1.0000    2   -1

    Ignore:
        from xinspect.dynamic_kwargs import get_func_kwargs
        globals().update(get_func_kwargs(_assign_confusion_vectors))

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> import pandas as pd
        >>> from netharn.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(nimgs=1, nclasses=8,
        >>>                              nboxes=(0, 20), n_fp=20,
        >>>                              box_noise=.2, cls_noise=.3)
        >>> classes = dmet.classes
        >>> gid = 0
        >>> true_dets = dmet.true_detections(gid)
        >>> pred_dets = dmet.pred_detections(gid)
        >>> y = _assign_confusion_vectors(true_dets, pred_dets,
        >>>                               classes=dmet.classes,
        >>>                               compat='all', prioritize='class')
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
        >>> print(y[(y.pred_raw != y.pred)])
        >>> y = _assign_confusion_vectors(true_dets, pred_dets,
        >>>                               classes=dmet.classes,
        >>>                               compat='ancestors', ovthresh=.5)
        >>> y = pd.DataFrame(y)
        >>> print(y)  # xdoc: +IGNORE_WANT
        >>> print(y[(y.pred_raw != y.pred)])
    """
    import kwarray
    valid_compat_keys = {'ancestors', 'mutex', 'all'}
    if compat not in valid_compat_keys:
        raise KeyError(compat)
    if classes is None and compat == 'ancestors':
        compat = 'mutex'

    if compat == 'mutex':
        prioritize = 'iou'

    # Group true boxes by class
    # Keep track which true boxes are unused / not assigned
    unique_tcxs, tgroupxs = kwarray.group_indices(true_dets.class_idxs)
    cx_to_txs = dict(zip(unique_tcxs, tgroupxs))

    unique_pcxs = np.array(sorted(set(pred_dets.class_idxs)))

    if classes is None:
        import ndsampler
        # Build mutually exclusive category tree
        all_cxs = sorted(set(map(int, unique_pcxs)) | set(map(int, unique_tcxs)))
        all_cxs = list(range(max(all_cxs) + 1))
        classes = ndsampler.CategoryTree.from_mutex(all_cxs)

    cx_to_ancestors = classes.idx_to_ancestor_idxs()

    if prioritize == 'iou':
        pdist_priority = None  # TODO: cleanup
    else:
        pdist_priority = _fast_pdist_priority(classes, prioritize)

    if compat == 'mutex':
        # assume classes are mutually exclusive if hierarchy is not given
        cx_to_matchable_cxs = {cx: [cx] for cx in unique_pcxs}
    elif compat == 'ancestors':
        cx_to_matchable_cxs = {
            cx: sorted([cx] + sorted(ub.take(
                classes.node_to_idx,
                nx.ancestors(classes.graph, classes.idx_to_node[cx]))))
            for cx in unique_pcxs
        }
    elif compat == 'all':
        cx_to_matchable_cxs = {cx: unique_tcxs for cx in unique_pcxs}
    else:
        raise KeyError(compat)

    if compat == 'all':
        # In this case simply run the full pairwise iou
        common_true_idxs = np.arange(len(true_dets))
        cx_to_matchable_txs = {cx: common_true_idxs for cx in unique_pcxs}
        common_ious = pred_dets.boxes.ious(true_dets.boxes, impl='c', bias=bias)
        iou_lookup = dict(enumerate(common_ious))
    else:
        # For each pred-category find matchable true-indices
        cx_to_matchable_txs = {}
        for cx, compat_cx in cx_to_matchable_cxs.items():
            matchable_cxs = cx_to_matchable_cxs[cx]
            compat_txs = ub.dict_take(cx_to_txs, matchable_cxs, default=[])
            compat_txs = np.array(sorted(ub.flatten(compat_txs)), dtype=np.int)
            cx_to_matchable_txs[cx] = compat_txs

        # Batch up the IOU pre-computation between compatible truths / preds
        iou_lookup = {}
        unique_pred_cxs, pgroupxs = kwarray.group_indices(pred_dets.class_idxs)
        for cx, pred_idxs in zip(unique_pred_cxs, pgroupxs):
            true_idxs = cx_to_matchable_txs[cx]
            ious = pred_dets.boxes[pred_idxs].ious(true_dets.boxes[true_idxs], bias=bias)
            _px_to_iou = dict(zip(pred_idxs, ious))
            iou_lookup.update(_px_to_iou)
    isvalid_lookup = {px: ious > ovthresh for px, ious in iou_lookup.items()}

    # sort predictions by descending score
    if 'scores' in pred_dets.data:
        _scores = pred_dets.scores
    else:
        _scores = np.ones(len(pred_dets))

    _pred_sortx = _scores.argsort()[::-1]
    _pred_cxs = pred_dets.class_idxs.take(_pred_sortx, axis=0)
    _pred_scores = _scores.take(_pred_sortx, axis=0)

    return _critical_loop(true_dets, pred_dets, _pred_sortx, _pred_cxs,
                          _pred_scores, iou_lookup, isvalid_lookup,
                          cx_to_matchable_txs, bg_weight, prioritize, ovthresh,
                          pdist_priority, cx_to_ancestors, bg_cidx)


def _critical_loop(true_dets, pred_dets, _pred_sortx, _pred_cxs, _pred_scores,
                   iou_lookup, isvalid_lookup, cx_to_matchable_txs, bg_weight,
                   prioritize, ovthresh, pdist_priority, cx_to_ancestors,
                   bg_cidx):
    # Notes:
    # * Preallocating numpy arrays does not help
    # * It might be useful to code this critical loop up in C / Cython
    # * Could numba help? (I'm having an issue with cmath)
    import kwarray

    y_pred_raw = []
    y_pred = []
    y_true = []
    y_score = []
    y_weight = []
    y_iou = []
    y_pxs = []
    y_txs = []

    # Keep track of which true items have been used
    true_unused = np.ones(len(true_dets), dtype=np.bool)

    y_pred_raw.extend(_pred_cxs.tolist())
    y_pxs.extend(_pred_sortx.tolist())
    y_score.extend(_pred_scores.tolist())

    # Greedy assignment. For each predicted detection box.
    # Allow it to match the truth of compatible classes.
    for px, pred_cx, score in zip(_pred_sortx, _pred_cxs, _pred_scores):
        # Find compatible truth indices
        # raw_pred_cx = pred_cx
        true_idxs = cx_to_matchable_txs[pred_cx]
        # Filter out any truth that has already been used
        unused = true_unused[true_idxs]
        unused_true_idxs = true_idxs[unused]

        ovmax = -np.inf
        ovidx = None
        weight = bg_weight
        tx = -1  # we will set this to the index of the assignd gt

        if len(unused_true_idxs):
            # First grab all candidate unused true boxes and lookup precomputed
            # ious between this pred and true_idxs
            cand_true_idxs = unused_true_idxs

            if prioritize == 'iou':
                # simply match the true box with the highest iou regardless of
                # category
                cand_ious = iou_lookup[px].compress(unused)
                ovidx = cand_ious.argmax()
                ovmax = cand_ious[ovidx]
                if ovmax > ovthresh:
                    tx = cand_true_idxs[ovidx]
            elif prioritize == 'correct' or prioritize == 'class':
                # Choose which (if any) of the overlapping true boxes to match
                # If there are any correct matches above the overlap threshold
                # choose to match that.
                # Flag any unused true box that overlaps
                overlap_flags = isvalid_lookup[px][unused]

                if overlap_flags.any():
                    cand_ious = iou_lookup[px][unused]
                    cand_true_cxs = true_dets.class_idxs[cand_true_idxs]
                    cand_true_idxs = cand_true_idxs[overlap_flags]
                    cand_true_cxs = cand_true_cxs[overlap_flags]
                    cand_ious = cand_ious[overlap_flags]

                    # Choose candidate with highest priority
                    # (prefer finer-grained correct classes over higher overlap,
                    #  but choose highest overlap in a tie).
                    cand_class_priority = pdist_priority[pred_cx][cand_true_cxs]

                    # ovidx = ub.argmax(zip(cand_class_priority, cand_ious))
                    ovidx = kwarray.arglexmax([cand_ious, cand_class_priority])

                    ovmax = cand_ious[ovidx]
                    tx = cand_true_idxs[ovidx]
            else:
                raise KeyError(prioritize)

        if tx > -1:
            # If the prediction matched a true object, mark the assignment
            # as either a true or false positive
            tx = unused_true_idxs[ovidx]
            true_unused[tx] = False  # mark this true box as used

            if 'weights' in true_dets.data:
                weight = true_dets.weights[tx]
            else:
                weight = 1.0
            true_cx = true_dets.class_idxs[tx]
            # If the prediction is a finer-grained category than the truth
            # change the prediction to match the truth (because it is
            # compatible). This is the key to hierarchical scoring.
            if true_cx in cx_to_ancestors[pred_cx]:
                pred_cx = true_cx

            # y_pred_raw.append(raw_pred_cx)
            y_pred.append(pred_cx)
            y_true.append(true_cx)
            # y_score.append(score)
            y_weight.append(weight)
            y_iou.append(ovmax)
            # y_pxs.append(px)
            y_txs.append(tx)
        else:
            # Assign this prediction to a the background
            # Mark this prediction as a false positive

            # y_pred_raw.append(raw_pred_cx)
            y_pred.append(pred_cx)
            y_true.append(bg_cidx)
            # y_score.append(score)
            y_weight.append(bg_weight)
            y_iou.append(-1)
            # y_pxs.append(px)
            y_txs.append(tx)

    # All pred boxes have been assigned to a truth box or the background.
    # Mark unused true boxes we failed to predict as false negatives
    bg_px = -1
    unused_txs = np.where(true_unused)[0]
    n = len(unused_txs)

    unused_y_true = true_dets.class_idxs[unused_txs].tolist()
    if 'weights' in true_dets.data:
        unused_y_weight = true_dets.weights[unused_txs].tolist()
    else:
        unused_y_weight = [1.0] * n

    y_pred_raw.extend([-1] * n)
    y_pred.extend([-1] * n)
    y_true.extend(unused_y_true)
    y_score.extend([0.0] * n)
    y_iou.extend([-1] * n)
    y_weight.extend(unused_y_weight)
    y_pxs.extend([bg_px] * n)
    y_txs.extend(unused_txs.tolist())

    # TODO:
    # Can we augment these vectors to balance these false negatives with
    # implicit false positives?

    BALANCE_FN = False
    if BALANCE_FN:
        # balance each false negatives with a false positive
        # Note: doing this is more correct, but is also more expensive Upweight
        # these True negative examples to capture the fact that there are
        # effectively infinite true negatives.
        w_factor = 10000000
        y_pred_raw.extend(unused_y_true)
        y_pred.extend(unused_y_true)
        y_true.extend([-1] * n)
        y_score.extend([0.0] * n)
        y_iou.extend([-1] * n)
        y_weight.extend((np.array(unused_y_weight) * w_factor).tolist())
        y_pxs.extend([bg_px] * n)
        y_txs.extend([bg_px] * n)

    y = {
        'pred_raw': y_pred_raw,
        'pred': y_pred,
        'true': y_true,
        'score': y_score,
        'weight': y_weight,
        'iou': y_iou,
        'txs': y_txs,  # index into the original true box for this row
        'pxs': y_pxs,  # index into the original pred box for this row
    }
    # val_lens = ub.map_vals(len, y)
    # print('val_lens = {!r}'.format(val_lens))
    # assert ub.allsame(val_lens.values())
    return y


def _fast_pdist_priority(classes, prioritize, _cache={}):
    """
    Custom priority computation. Needs some vetting.

    This is the priority used when deciding which prediction to assign to which
    truth.

    TODO:
        - [ ] Look at absolute difference in sibling entropy when deciding
              whether to go up or down in the tree.
    """
    #  Note: distances to ancestors will be negative and distances
    #  to descendants will be positive. Prefer matching ancestors
    #  over descendants.
    key = ub.hash_data('\n'.join(list(map(str, classes))), hasher='sha1')
    # key = ub.repr2(classes.__json__())
    if key not in _cache:
        # classes = ndsampler.CategoryTree.from_json(classes)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid .* less')
            warnings.filterwarnings('ignore', message='invalid .* greater_equal')
            # Get basic distance between nodes
            pdist = classes.idx_pairwise_distance()
            pdist_priority = np.array(pdist, dtype=np.float32, copy=True)
            if prioritize == 'correct':
                # Prioritizes all ancestors first, and then descendants
                # afterwords, nodes off the direct lineage are ignored.
                valid_vals = pdist_priority[np.isfinite(pdist_priority)]
                maxval = (valid_vals.max() - valid_vals.min()) + 1
                is_ancestor = (pdist_priority >= 0)
                is_descend = (pdist_priority < 0)
                # Prioritize ALL ancestors first
                pdist_priority[is_ancestor] = (
                    2 * maxval - pdist_priority[is_ancestor])
                # Prioritize ALL descendants next
                pdist_priority[is_descend] = (
                    maxval + pdist_priority[is_descend])
                pdist_priority[np.isnan(pdist_priority)] = -np.inf
            elif prioritize == 'class':
                # Prioritizes the exact match first, and then it alternates
                # between ancestors and desendants based on distance to self
                pdist_priority[pdist_priority < -1] += .5
                pdist_priority = np.abs(pdist_priority)
                pdist_priority[np.isnan(pdist_priority)] = np.inf
                pdist_priority = 1 / (pdist_priority + 1)
            else:
                raise KeyError(prioritize)
        _cache[key] = pdist_priority
    pdist_priority = _cache[key]
    return pdist_priority
