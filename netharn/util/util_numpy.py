import numpy as np
import ubelt as ub  # NOQA


def iter_reduce_ufunc(ufunc, arr_iter, out=None):
    """
    constant memory iteration and reduction

    applys ufunc from left to right over the input arrays

    Example:
        >>> arr_list = [
        ...     np.array([0, 1, 2, 3, 8, 9]),
        ...     np.array([4, 1, 2, 3, 4, 5]),
        ...     np.array([0, 5, 2, 3, 4, 5]),
        ...     np.array([1, 1, 6, 3, 4, 5]),
        ...     np.array([0, 1, 2, 7, 4, 5])
        ... ]
        >>> memory = np.array([9, 9, 9, 9, 9, 9])
        >>> gen_memory = memory.copy()
        >>> def arr_gen(arr_list, gen_memory):
        ...     for arr in arr_list:
        ...         gen_memory[:] = arr
        ...         yield gen_memory
        >>> print('memory = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> ufunc = np.maximum
        >>> res1 = iter_reduce_ufunc(ufunc, iter(arr_list), out=None)
        >>> res2 = iter_reduce_ufunc(ufunc, iter(arr_list), out=memory)
        >>> res3 = iter_reduce_ufunc(ufunc, arr_gen(arr_list, gen_memory), out=memory)
        >>> print('res1       = %r' % (res1,))
        >>> print('res2       = %r' % (res2,))
        >>> print('res3       = %r' % (res3,))
        >>> print('memory     = %r' % (memory,))
        >>> print('gen_memory = %r' % (gen_memory,))
        >>> assert np.all(res1 == res2)
        >>> assert np.all(res2 == res3)
    """
    # Get first item in iterator
    try:
        initial = next(arr_iter)
    except StopIteration:
        return None
    # Populate the outvariable if specified otherwise make a copy of the first
    # item to be the output memory
    if out is not None:
        out[:] = initial
    else:
        out = initial.copy()
    # Iterate and reduce
    for arr in arr_iter:
        ufunc(out, arr, out=out)
    return out


def isect_flags(arr, other):
    """
    Example:
        >>> arr = np.array([
        >>>     [1, 2, 3, 4],
        >>>     [5, 6, 3, 4],
        >>>     [1, 1, 3, 4],
        >>> ])
        >>> other = np.array([1, 4, 6])
        >>> mask = isect_flags(arr, other)
        >>> print(mask)
        [[ True False False  True]
         [False  True False  True]
         [ True  True False  True]]
    """
    flags = iter_reduce_ufunc(np.logical_or, (arr == item for item in other)).ravel()
    flags.shape = arr.shape
    return flags


def atleast_nd(arr, n, front=False):
    r"""
    View inputs as arrays with at least n dimensions.
    TODO: Submit as a PR to numpy

    Args:
        arr (array_like): One array-like object.  Non-array inputs are
                converted to arrays.  Arrays that already have n or more
                dimensions are preserved.
        n (int): number of dimensions to ensure
        tofront (bool): if True new dimensions are added to the front of the
            array.  otherwise they are added to the back.

    Returns
    -------
        ndarray :
            An array with ``a.ndim >= n``.  Copies are avoided where possible,
            and views with three or more dimensions are returned.  For example,
            a 1-D array of shape ``(N,)`` becomes a view of shape
            ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a view
            of shape ``(M, N, 1)``.

    See Also
    ---------
        ensure_shape, np.atleast_1d, np.atleast_2d, np.atleast_3d

    Example
    -------
        >>> n = 2
        >>> arr = np.array([1, 1, 1])
        >>> arr_ = atleast_nd(arr, n)
        >>> result = ub.repr2(arr_.tolist(), nl=0)
        >>> print(result)
        [[1], [1], [1]]

    Example
    -------

        >>> n = 4
        >>> arr1 = [1, 1, 1]
        >>> arr2 = np.array(0)
        >>> arr3 = np.array([[[[[1]]]]])
        >>> arr1_ = atleast_nd(arr1, n)
        >>> arr2_ = atleast_nd(arr2, n)
        >>> arr3_ = atleast_nd(arr3, n)
        >>> result1 = ub.repr2(arr1_.tolist(), nl=0)
        >>> result2 = ub.repr2(arr2_.tolist(), nl=0)
        >>> result3 = ub.repr2(arr3_.tolist(), nl=0)
        >>> result = '\n'.join([result1, result2, result3])
        >>> print(result)
        [[[[1]]], [[[1]]], [[[1]]]]
        [[[[0]]]]
        [[[[[1]]]]]

    Ignore:
        # Hmm, mine is actually faster
        %timeit atleast_nd(arr, 3)
        %timeit np.atleast_3d(arr)

    Benchmark:

        import ubelt
        N = 100

        t1 = ubelt.Timerit(N, label='mine')
        for timer in t1:
            arr = np.empty((10, 10))
            with timer:
                atleast_nd(arr, 3)

        t2 = ubelt.Timerit(N, label='baseline')
        for timer in t2:
            arr = np.empty((10, 10))
            with timer:
                np.atleast_3d(arr)

    """
    arr_ = np.asanyarray(arr)
    ndims = len(arr_.shape)
    if n is not None and ndims <  n:
        # append the required number of dimensions to the front or back
        if front:
            expander = (None,) * (n - ndims) + (Ellipsis,)
        else:
            expander = (Ellipsis,) + (None,) * (n - ndims)
        arr_ = arr_[expander]
    return arr_


def apply_grouping(items, groupxs, axis=0):
    """
    applies grouping from group_indicies
    apply_grouping

    Args:
        items (ndarray):
        groupxs (list of ndarrays):

    Returns:
        list of ndarrays: grouped items

    SeeAlso:
        group_indices
        invert_apply_grouping

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> items        = np.array([1, 8, 5, 5, 8, 6, 7, 5, 3, 0, 9])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> grouped_items = apply_grouping(items, groupxs)
        >>> result = str(grouped_items)
        >>> print(result)
        [array([8, 5, 6]), array([1, 5, 8, 7]), array([5, 3, 0, 9])]
    """
    # SHOULD DO A CONTIGUOUS CHECK HERE
    #items_ = np.ascontiguousarray(items)
    return [items.take(xs, axis=axis) for xs in groupxs]


def group_indices(idx2_groupid, assume_sorted=False):
    r"""
    group_indices

    Args:
        idx2_groupid (ndarray): numpy array of group ids (must be numeric)

    Returns:
        tuple (ndarray, list of ndarrays): (keys, groupxs)

    Example0:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> idx2_groupid = np.array([2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> result = ub.repr2((keys, groupxs), nobr=True, with_dtype=True)
        >>> print(result)
        np.array([1, 2, 3], dtype=np.int64),
        [
            np.array([1, 3, 5], dtype=np.int64),
            np.array([0, 2, 4, 6], dtype=np.int64),
            np.array([ 7,  8,  9, 10], dtype=np.int64)...

    Example1:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> idx2_groupid = np.array([[  24], [ 129], [ 659], [ 659], [ 24],
        ...       [659], [ 659], [ 822], [ 659], [ 659], [24]])
        >>> # 2d arrays must be flattened before coming into this function so
        >>> # information is on the last axis
        >>> (keys, groupxs) = group_indices(idx2_groupid.T[0])
        >>> result = ub.repr2((keys, groupxs), nobr=True, with_dtype=True)
        >>> print(result)
        np.array([ 24, 129, 659, 822], dtype=np.int64),
        [
            np.array([ 0,  4, 10], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2, 3, 5, 6, 8, 9], dtype=np.int64),
            np.array([7], dtype=np.int64)...

    Example2:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> idx2_groupid = np.array([True, True, False, True, False, False, True])
        >>> (keys, groupxs) = group_indices(idx2_groupid)
        >>> result = ub.repr2((keys, groupxs), nobr=True, with_dtype=True)
        >>> print(result)
        np.array([False,  True], dtype=np.bool),
        [
            np.array([2, 4, 5], dtype=np.int64),
            np.array([0, 1, 3, 6], dtype=np.int64)...

    Timeit:
        import numba
        group_indices_numba = numba.jit(group_indices)
        group_indices_numba(idx2_groupid)

    SeeAlso:
        apply_grouping

    References:
        http://stackoverflow.com/questions/4651683/
        numpy-grouping-using-itertools-groupby-performance

    TODO:
        Look into np.split
        http://stackoverflow.com/questions/21888406/
        getting-the-indexes-to-the-duplicate-columns-of-a-numpy-array
    """
    # Sort items and idx2_groupid by groupid
    if assume_sorted:
        sortx = np.arange(len(idx2_groupid))
        groupids_sorted = idx2_groupid
    else:
        sortx = idx2_groupid.argsort()
        groupids_sorted = idx2_groupid.take(sortx)

    # Ensure bools are internally cast to integers
    if groupids_sorted.dtype.kind == 'b':
        cast_groupids = groupids_sorted.astype(np.int8)
    else:
        cast_groupids = groupids_sorted

    num_items = idx2_groupid.size
    # Find the boundaries between groups
    diff = np.ones(num_items + 1, cast_groupids.dtype)
    np.subtract(cast_groupids[1:], cast_groupids[:-1], out=diff[1:num_items])
    idxs = np.flatnonzero(diff)
    # Groups are between bounding indexes
    # <len(keys) bottlneck>
    groupxs = [sortx[lx:rx] for lx, rx in zip(idxs, idxs[1:])]  # 34.5%
    # Unique group keys
    keys = groupids_sorted[idxs[:-1]]
    return keys, groupxs


def group_items(item_list, groupid_list, assume_sorted=False, axis=None):
    keys, groupxs = group_indices(groupid_list, assume_sorted=assume_sorted)
    grouped_values = apply_grouping(item_list, groupxs, axis=axis)
    return dict(zip(keys, grouped_values))


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_numpy all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
