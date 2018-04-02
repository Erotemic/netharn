# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
NUMPY_INCLUDE=$(python -c "import numpy as np; print(np.get_include())")
CPATH=$CPATH:$NUMPY_INCLUDE cythonize -a -i ~/code/clab/clab/models/yolo2/utils/nms/gpu_nms.pyx

See Also:
    https://github.com/bharatsingh430/soft-nms/blob/dc97adf100fb2cad66e04f0d09e031fce81948c5/lib/nms/py_cpu_nms.py
"""

import numpy as np
cimport cython
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp" nogil:
    void _nms(int*, int*, float*, int, int, float, int)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,
            np.int32_t device_id=0):
    cdef int boxes_num = dets.shape[0]
    cdef int boxes_dim = dets.shape[1]
    cdef int num_out
    cdef np.ndarray[np.int32_t, ndim=1] keep = np.zeros(boxes_num, dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
    cdef np.ndarray[np.float32_t, ndim=2] sorted_dets = dets[order, :]

    cdef float[:, :] sorted_dets_view = sorted_dets
    cdef int[:] keep_view = keep

    cdef float* det_00_ptr = &sorted_dets_view[0, 0]
    cdef int* keep0_ptr = &keep_view[0]
    cdef int* num_out_ptr = &num_out

    cdef float thresh_ = thresh
    cdef int device_id_ = device_id

    with nogil:
        _nms(keep0_ptr, num_out_ptr, det_00_ptr, boxes_num, 
             boxes_dim, thresh_, device_id_)

    keep = keep[:num_out]
    return list(order[keep])
