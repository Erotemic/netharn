import torch
import numpy as np
import ubelt as ub
from netharn.util.nms import py_nms
from netharn.util import profiler
from netharn.util.nms import torch_nms
import warnings

_impls = {}
_impls['py'] = py_nms.py_nms
_impls['torch'] = torch_nms.torch_nms
_automode = 'py'
try:
    from netharn.util.nms import cpu_nms
    _impls['cpu'] = cpu_nms.cpu_nms
    _automode = 'cpu'
except Exception:
    warnings.warn('cpu_nms is not available')
try:
    if torch.cuda.is_available():
        from netharn.util.nms import gpu_nms
        _impls['gpu'] = gpu_nms.gpu_nms
        _automode = 'gpu'
except Exception:
    warnings.warn('gpu_nms is not available')


def available_nms_impls():
    return list(_impls.keys())


@profiler.profile
def non_max_supression(tlbr, scores, thresh, bias=0.0, classes=None,
                       impl='auto'):
    """
    Non-Maximum Suppression

    Args:
        tlbr (ndarray): Nx4 boxes in tlbr format
        scores (ndarray): score for each bbox
        thresh (float): iou threshold
        bias (float): bias for iou computation either 0 or 1
           (hint: choosing 1 is wrong computer vision community)
        classes (ndarray or None): integer classes. If specified NMS is done
            on a perclass basis.
        impl (str): implementation can be auto, python, cpu, or gpu


    CommandLine:
        python ~/code/netharn/netharn/util/nms/nms_core.py nms
        python ~/code/netharn/netharn/util/nms/nms_core.py nms:0
        python ~/code/netharn/netharn/util/nms/nms_core.py nms:1

    References:
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx <- TODO

    Example:
        >>> dets = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>> ], dtype=np.float32)
        >>> scores = np.array([.1, .5, .9, .1])
        >>> thresh = .5
        >>> keep = non_max_supression(dets, scores, thresh, impl='py')
        >>> print('keep = {!r}'.format(keep))
        keep = [2, 1, 3]
        >>> thresh = 0.0
        >>> if 'py' in available_nms_impls():
        >>>     keep = non_max_supression(dets, scores, thresh, impl='py')
        >>>     assert list(keep) == [2, 1]
        >>> if 'cpu' in available_nms_impls():
        >>>     keep = non_max_supression(dets, scores, thresh, impl='cpu')
        >>>     assert list(keep) == [2, 1]
        >>> if 'gpu' in available_nms_impls():
        >>>     keep = non_max_supression(dets, scores, thresh, impl='gpu')
        >>>     assert list(keep) == [2, 1]
        >>> if 'torch' in available_nms_impls():
        >>>     keep = non_max_supression(dets, scores, thresh, impl='torch')
        >>>     assert set(keep.tolist()) == {2, 1}

    Example:
        >>> import ubelt as ub
        >>> dets = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32)
        >>> scores = np.linspace(0, 1, len(dets))
        >>> thresh = .2
        >>> solutions = {}
        >>> for impl in _impls:
        >>>     solutions[impl] = sorted(non_max_supression(dets, scores, thresh, impl=impl))
        >>> print('solutions = {}'.format(ub.repr2(solutions, nl=1)))
        >>> assert ub.allsame(solutions.values())
    """
    if tlbr.shape[0] == 0:
        return []

    if impl == 'auto':
        impl = _automode

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
            keep = py_nms.py_nms(tlbr, scores, thresh, bias=float(bias))
        elif impl == 'torch':
            was_tensor = torch.is_tensor(tlbr)
            if not was_tensor:
                tlbr = torch.Tensor(tlbr)
                scores = torch.Tensor(scores)
            flags = torch_nms.torch_nms(tlbr, scores, thresh=thresh,
                                        bias=float(bias))
            keep = np.where(flags.cpu().numpy())[0]
        else:
            # TODO: it would be nice to be able to pass torch tensors here
            nms = _impls[impl]
            tlbr = tlbr.astype(np.float32)
            scores = scores.astype(np.float32)
            # dets = np.hstack((tlbr, scores[:, None])).astype(np.float32)
            if impl == 'gpu':
                # HACK: we should parameterize which device is used
                device = torch.cuda.current_device()
                keep = nms(tlbr, scores, float(thresh), bias=float(bias), device_id=device)
            else:
                keep = nms(tlbr, scores, float(thresh), bias=float(bias))
        return keep


# TODO: soft nms


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.nms.nms_core all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
