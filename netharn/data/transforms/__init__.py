"""
mkinit netharn.data.transforms')"
"""
# flake8: noqa

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.data.transforms import augmenter_base
    from netharn.data.transforms import augmenters
    from netharn.data.transforms.augmenter_base import (ParamatarizedAugmenter,)
    from netharn.data.transforms.augmenters import (HSVShift, LetterboxResize,
                                                    Resize, demodata_hsv_image,)
    # </AUTOGEN_INIT>
