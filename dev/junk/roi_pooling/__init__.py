from . import roi_pool_py
try:
    from . import roi_pool_c
except Exception:
    roi_pool = roi_pool_py
else:
    roi_pool = roi_pool_c
