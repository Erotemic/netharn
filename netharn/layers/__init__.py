"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.layers')"
"""
# flake8: noqa

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.layers import reorg
    from netharn.layers import roi_pooling
    
    from netharn.layers.roi_pooling import (roi_pool, roi_pool,)
