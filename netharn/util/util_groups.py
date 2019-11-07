# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import ubelt as ub  # NOQA


def group_consecutive(arr, offset=1):
    """
    Returns lists of consecutive values

    Args:
        arr (ndarray): array of ordered values
        offset (float): any two values separated by this offset are grouped.
            In the default case, when offset=1, this groups increasing values
            like: 0, 1, 2. When offset is 0 it groups consecutive values
            thta are the same, e.g.: 4, 4, 4.

    Returns:
        list of ndarray: a list of arrays that are the groups from the input

    Notes:
        This is equivalent (and faster) to using:
        apply_grouping(data, group_consecutive_indices(data))

    References:
        http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy

    Example:
        >>> arr = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 99, 100, 101])
        >>> groups = group_consecutive(arr)
        >>> print('groups = {}'.format(list(map(list, groups))))
        groups = [[1, 2, 3], [5, 6, 7, 8, 9, 10], [15], [99, 100, 101]]
        >>> arr = np.array([0, 0, 3, 0, 0, 7, 2, 3, 4, 4, 4, 1, 1])
        >>> groups = group_consecutive(arr, offset=1)
        >>> print('groups = {}'.format(list(map(list, groups))))
        groups = [[0], [0], [3], [0], [0], [7], [2, 3, 4], [4], [4], [1], [1]]
        >>> groups = group_consecutive(arr, offset=0)
        >>> print('groups = {}'.format(list(map(list, groups))))
        groups = [[0, 0], [3], [0, 0], [7], [2], [3], [4, 4, 4], [1, 1]]
    """
    split_indicies = np.nonzero(np.diff(arr) != offset)[0] + 1
    groups = np.array_split(arr, split_indicies)
    return groups


def group_consecutive_indices(arr, offset=1):
    """
    Returns lists of indices pointing to consecurive values

    Args:
        arr (ndarray): array of ordered values
        offset (float): any two values separated by this offset are grouped.

    Returns:
        list of ndarray: groupxs: a list of indices

    SeeAlso:
        group_consecutive
        apply_grouping

    Example:
        >>> arr = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 99, 100, 101])
        >>> groupxs = group_consecutive_indices(arr)
        >>> print('groupxs = {}'.format(list(map(list, groupxs))))
        groupxs = [[0, 1, 2], [3, 4, 5, 6, 7, 8], [9], [10, 11, 12]]
        >>> assert all(np.array_equal(a, b) for a, b in zip(group_consecutive(arr, 1), apply_grouping(arr, groupxs)))
        >>> arr = np.array([0, 0, 3, 0, 0, 7, 2, 3, 4, 4, 4, 1, 1])
        >>> groupxs = group_consecutive_indices(arr, offset=1)
        >>> print('groupxs = {}'.format(list(map(list, groupxs))))
        groupxs = [[0], [1], [2], [3], [4], [5], [6, 7, 8], [9], [10], [11], [12]]
        >>> assert all(np.array_equal(a, b) for a, b in zip(group_consecutive(arr, 1), apply_grouping(arr, groupxs)))
        >>> groupxs = group_consecutive_indices(arr, offset=0)
        >>> print('groupxs = {}'.format(list(map(list, groupxs))))
        groupxs = [[0, 1], [2], [3, 4], [5], [6], [7], [8, 9, 10], [11, 12]]
        >>> assert all(np.array_equal(a, b) for a, b in zip(group_consecutive(arr, 0), apply_grouping(arr, groupxs)))
    """
    split_indicies = np.nonzero(np.diff(arr) != offset)[0] + 1
    groupxs = np.array_split(np.arange(len(arr)), split_indicies)
    return groupxs


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
    """
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
    """ Works like ub.group_items, but with numpy optimizations """
    keys, groupxs = group_indices(groupid_list, assume_sorted=assume_sorted)
    grouped_values = apply_grouping(item_list, groupxs, axis=axis)
    return dict(zip(keys, grouped_values))


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_groups all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
