# -*- coding: utf-8 -*-
"""
Generic Non-Maximum Suppression API with efficient backend implementations
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
import ubelt as ub
import warnings
from ._nms_backend import py_nms
from ._nms_backend import torch_nms


def daq_spatial_nms(tlbr, scores, diameter, thresh, max_depth=6,
                    stop_size=2048, recsize=2048, impl='auto'):
    """
    Divide and conquor speedup non-max-supression algorithm for when bboxes
    have a known max size

    Args:
        tlbr (ndarray): boxes in (tlx, tly, brx, bry) format
        scores (ndarray): scores of each box
        diameter (int or Tuple[int, int]): Distance from split point to
            consider rectification. If specified as an integer, then number
            is used for both height and width. If specified as a tuple, then
            dims are assumed to be in [height, width] format.
        max_depth (int): maximum number of times we can divide and conquor
        stop_size (int): number of boxes that triggers full NMS computation
        recsize (int): number of boxes that triggers full NMS recombination

    LookInfo:
        # Didn't read yet but it seems similar
        http://www.cyberneum.de/fileadmin/user_upload/files/publications/CVPR2010-Lampert_[0].pdf

        https://www.researchgate.net/publication/220929789_Efficient_Non-Maximum_Suppression

        # This seems very similar
        https://projet.liris.cnrs.fr/m2disco/pub/Congres/2006-ICPR/DATA/C03_0406.PDF

    Example:
        >>> import netharn as nh
        >>> # Make a bunch of boxes with the same width and height
        >>> boxes = nh.util.Boxes.random(237, scale=1000, format='cxywh')
        >>> boxes.data.T[2] = 10
        >>> boxes.data.T[3] = 10
        >>> #
        >>> tlbr = boxes.to_tlbr().data.astype(np.float32)
        >>> scores = np.arange(0, len(tlbr)).astype(np.float32)
        >>> #
        >>> n_megabytes = (tlbr.size * tlbr.dtype.itemsize) / (2 ** 20)
        >>> print('n_megabytes = {!r}'.format(n_megabytes))
        >>> #
        >>> thresh = iou_thresh = 0.01
        >>> impl = 'auto'
        >>> max_depth = 20
        >>> diameter = 10
        >>> stop_size = 2000
        >>> recsize = 500
        >>> #
        >>> import ubelt as ub
        >>> #
        >>> with ub.Timer(label='daq'):
        >>>     keep1 = daq_spatial_nms(tlbr, scores,
        >>>         diameter=diameter, thresh=thresh, max_depth=max_depth,
        >>>         stop_size=stop_size, recsize=recsize, impl=impl)
        >>> #
        >>> with ub.Timer(label='full'):
        >>>     keep2 = non_max_supression(tlbr, scores,
        >>>         thresh=thresh, impl=impl)
        >>> #
        >>> # Due to the greedy nature of the algorithm, there will be slight
        >>> # differences in results, but they will be mostly similar.
        >>> similarity = len(set(keep1) & set(keep2)) / len(set(keep1) | set(keep2))
        >>> print('similarity = {!r}'.format(similarity))
    """
    def _rectify(tlbr, both_keep, needs_rectify):
        if len(needs_rectify) == 0:
            keep = sorted(both_keep)
        else:
            nr_arr = np.array(sorted(needs_rectify))
            nr = needs_rectify
            bk = set(both_keep)
            rectified_keep = non_max_supression(
                tlbr[nr_arr], scores[nr_arr], thresh=thresh,
                impl=impl)
            rk = set(nr_arr[rectified_keep])
            keep = sorted((bk - nr) | rk)
        return keep

    def _recurse(tlbr, scores, dim, depth, diameter_wh):
        """
        Args:
            dim (int): flips between 0 and 1
            depth (int): recursion depth
        """
        # print('recurse')
        n_boxes = len(tlbr)
        if depth >= max_depth or n_boxes < stop_size:
            # print('n_boxes = {!r}'.format(n_boxes))
            # print('depth = {!r}'.format(depth))
            # print('stop')
            keep = non_max_supression(tlbr, scores, thresh=thresh, impl=impl)
            both_keep = sorted(keep)
            needs_rectify = set()
        else:
            # Break up the NMS into two subproblems.
            middle = np.median(tlbr.T[dim])
            left_flags = tlbr.T[dim] < middle
            right_flags = ~left_flags

            left_idxs = np.where(left_flags)[0]
            right_idxs = np.where(right_flags)[0]

            left_scores = scores[left_idxs]
            left_tlbr = tlbr[left_idxs]

            right_scores = scores[right_idxs]
            right_tlbr = tlbr[right_idxs]

            next_depth = depth + 1
            next_dim = 1 - dim

            # Solve each subproblem
            left_keep_, lrec_ = _recurse(
                left_tlbr, left_scores, depth=next_depth, dim=next_dim,
                diameter_wh=diameter_wh)

            right_keep_, rrec_ = _recurse(
                right_tlbr, right_scores, depth=next_depth, dim=next_dim,
                diameter_wh=diameter_wh)

            # Recombine the results (note that because we have a
            # diameter_wh, we have to check less results)
            rrec = set(right_idxs[sorted(rrec_)])
            lrec = set(left_idxs[sorted(lrec_)])

            left_keep = left_idxs[left_keep_]
            right_keep = right_idxs[right_keep_]

            both_keep = np.hstack([left_keep, right_keep])
            both_keep.sort()

            rectify_flags = np.abs(tlbr[both_keep].T[dim] - middle) < diameter_wh[dim]
            needs_rectify = set(both_keep[rectify_flags]) | rrec | lrec

            nrec = len(needs_rectify)
            # print('nrec = {!r}'.format(nrec))
            if nrec > recsize:
                both_keep = _rectify(tlbr, both_keep, needs_rectify)
                needs_rectify = set()
        return both_keep, needs_rectify

    if not ub.iterable(diameter):
        diameter_wh = [diameter, diameter]
    else:
        diameter_wh = diameter[::-1]

    depth = 0
    dim = 0
    both_keep, needs_rectify = _recurse(tlbr, scores, dim=dim, depth=depth,
                                        diameter_wh=diameter_wh)
    keep = _rectify(tlbr, both_keep, needs_rectify)
    return keep


_impls = None


class _NMS_Impls():
    # TODO: could make this prettier
    def __init__(self):
        self._impls = None
        self._automode = None

    def _init(self, verbose=0):
        _impls = {}
        _impls['py'] = py_nms.py_nms
        _impls['torch'] = torch_nms.torch_nms
        _automode = 'py'
        try:
            from ._nms_backend import cpu_nms
            _impls['cpu'] = cpu_nms.cpu_nms
            _automode = 'cpu'
            if verbose:
                print('Was able to import cpu_nms sucessfully')
        except Exception as ex:
            warnings.warn('cpu_nms is not available: {}'.format(str(ex)))
        try:
            if torch.cuda.is_available():
                from ._nms_backend import gpu_nms
                _impls['gpu'] = gpu_nms.gpu_nms
                _automode = 'gpu'
                if verbose:
                    print('Was able to import gpu_nms sucessfully')
        except Exception as ex:
            warnings.warn('gpu_nms is not available: {}'.format(str(ex)))
        self._automode = _automode
        self._impls = _impls


_impls = _NMS_Impls()


def available_nms_impls():
    """
    List available values for the `impl` kwarg of `non_max_supression`

    Example:
        >>> impls = available_nms_impls()
        >>> assert 'py' in impls
        >>> print('impls = {!r}'.format(impls))
    """
    if not _impls._impls:
        _impls._init()
    return list(_impls._impls.keys())


def non_max_supression(tlbr, scores, thresh, bias=0.0, classes=None,
                       impl='auto', device_id=None):
    """
    Non-Maximum Suppression - remove redundant bounding boxes

    Args:
        tlbr (ndarray[float32]): Nx4 boxes in tlbr format
        scores (ndarray[float32]): score for each bbox
        thresh (float): iou threshold
        bias (float): bias for iou computation either 0 or 1
        classes (ndarray[int64] or None): integer classes.
            If specified NMS is done on a perclass basis.
        impl (str): implementation can be auto, python, cpu, or gpu
        device_id (int): used if impl is gpu, device id to work on. If not
            specified `torch.cuda.current_device()` is used.

    References:
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx <- TODO

    Example:
        >>> tlbr = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>> ], dtype=np.float32)
        >>> scores = np.array([.1, .5, .9, .1])
        >>> thresh = .5
        >>> keep = non_max_supression(tlbr, scores, thresh, impl='py')
        >>> print('keep = {!r}'.format(keep))
        keep = [2, 1, 3]
        >>> thresh = 0.0
        >>> if 'py' in available_nms_impls():
        >>>     keep = non_max_supression(tlbr, scores, thresh, impl='py')
        >>>     print('keep = {!r}'.format(keep))
        >>>     assert list(keep) == [2, 1]
        >>> if 'cpu' in available_nms_impls():
        >>>     keep = non_max_supression(tlbr, scores, thresh, impl='cpu')
        >>>     print('keep = {!r}'.format(keep))
        >>>     assert list(keep) == [2, 1]
        >>> if 'gpu' in available_nms_impls():
        >>>     keep = non_max_supression(tlbr, scores, thresh, impl='gpu')
        >>>     print('keep = {!r}'.format(keep))
        >>>     assert list(keep) == [2, 1]
        >>> if 'torch' in available_nms_impls():
        >>>     keep = non_max_supression(tlbr, scores, thresh, impl='torch')
        >>>     print('keep = {!r}'.format(keep))
        >>>     assert set(keep.tolist()) == {2, 1}

    Example:
        >>> import ubelt as ub
        >>> tlbr = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32)
        >>> scores = np.linspace(0, 1, len(tlbr))
        >>> thresh = .2
        >>> solutions = {}
        >>> if not _impls._impls:
        >>>     _impls._init()
        >>> for impl in _impls._impls:
        >>>     keep = non_max_supression(tlbr, scores, thresh, impl=impl)
        >>>     solutions[impl] = sorted(keep)
        >>> assert 'py' in solutions
        >>> print('solutions = {}'.format(ub.repr2(solutions, nl=1)))
        >>> assert ub.allsame(solutions.values())
    """
    if not _impls._impls:
        _impls._init()

    if tlbr.shape[0] == 0:
        return []

    if impl == 'auto':
        # if torch.is_tensor(tlbr):
        #     impl = 'torch'
        # else:
        impl = _impls._automode

    if classes is not None:
        keep = []
        for idxs in ub.group_items(range(len(classes)), classes).values():
            # cls_tlbr = tlbr.take(idxs, axis=0)
            # cls_scores = scores.take(idxs, axis=0)
            cls_tlbr = tlbr[idxs]
            cls_scores = scores[idxs]
            cls_keep = non_max_supression(cls_tlbr, cls_scores, thresh=thresh,
                                          bias=bias, impl=impl)
            keep.extend(list(ub.take(idxs, cls_keep)))
        return keep
    else:
        if impl == 'py':
            func = _impls._impls['py']
            keep = func(tlbr, scores, thresh, bias=float(bias))
        elif impl == 'torch':
            was_tensor = torch.is_tensor(tlbr)
            if not was_tensor:
                tlbr = torch.Tensor(tlbr)
                scores = torch.Tensor(scores)
            func = _impls._impls['torch']
            # Default output of torch impl is a mask
            flags = func(tlbr, scores, thresh=thresh, bias=float(bias))
            keep = torch.nonzero(flags).view(-1)
        else:
            # TODO: it would be nice to be able to pass torch tensors here
            nms = _impls._impls[impl]
            if torch.is_tensor(tlbr):
                tlbr = tlbr.data.cpu().numpy()
                scores = scores.data.cpu().numpy()
            tlbr = tlbr.astype(np.float32)
            scores = scores.astype(np.float32)
            # dets = np.hstack((tlbr, scores[:, None])).astype(np.float32)
            if impl == 'gpu':
                # TODO: if the data is already on a torch GPU can we just
                # use it?
                # HACK: we should parameterize which device is used
                if device_id is None:
                    device_id = torch.cuda.current_device()
                keep = nms(tlbr, scores, float(thresh), bias=float(bias),
                           device_id=device_id)
            else:
                keep = nms(tlbr, scores, float(thresh), bias=float(bias))
        return keep


# TODO: soft nms


if __name__ == '__main__':
    """
    CommandLine:
        # Test to see that extensions are loaded properly
        python -c "import netharn.util.util_nms; netharn.util.util_nms._impls._init(verbose=1)"

        xdoctest -m netharn.util.util_nms
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
