import torch
import numpy as np
from netharn.util.nms import py_nms

_impls = {}
_impls['py'] = py_nms.py_nms
_automode = 'py'
try:
    from netharn.util.nms import cpu_nms
    _impls['cpu'] = cpu_nms.cpu_nms
    _automode = 'cpu'
except Exception:
    raise
    pass
try:
    if torch.cuda.is_available():
        from netharn.util.nms import gpu_nms
        _impls['gpu'] = gpu_nms.gpu_nms
        _automode = 'gpu'
except Exception:
    raise
    pass


def non_max_supression(tlbr, scores, thresh, bias=0.0, impl='auto'):
    """
    Non-Maximum Suppression

    Args:
        tlbr (ndarray): Nx4 boxes in tlbr format
        scores (ndarray): score for each bbox
        thresh (float): iou threshold
        impl (str): implementation can be auto, python, cpu, or gpu
        bias (float): bias for iou computation either 0 or 1
           (hint: choosing 1 is wrong computer vision community)

    CommandLine:
        python ~/code/netharn/netharn/util/nms/nms_core.py nms
        python ~/code/netharn/netharn/util/nms/nms_core.py nms:0
        python ~/code/netharn/netharn/util/nms/nms_core.py nms:1

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

    Example:
        >>> import ubelt as ub
        >>> dets = np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>> ], dtype=np.float32)
        >>> scores = np.array([.1, .5, .9, .1])
        >>> thresh = .5
        >>> solutions = {}
        >>> for impl in _impls:
        >>>     solutions[impl] = non_max_supression(dets, scores, thresh, impl=impl)
        >>> print('solutions = {}'.format(ub.repr2(solutions, nl=1)))
        >>> assert ub.allsame(solutions.values())
    """
    if tlbr.shape[0] == 0:
        return []
    if impl == 'auto':
        impl = _automode
    if impl == 'py':
        keep = py_nms.py_nms(tlbr, scores, thresh, bias=float(bias))
    else:
        # TODO: it would be nice to be able to pass torch tensors here
        nms = _impls[impl]
        dets = np.hstack((tlbr, scores[:, np.newaxis])).astype(np.float32)
        if impl == 'gpu':
            device = torch.cuda.current_device()
            keep = nms(dets, thresh, bias=float(bias), device_id=device)
        else:
            keep = nms(dets, thresh, bias=float(bias))
    return keep


# TODO: soft nms


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.nms.nms_core all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
