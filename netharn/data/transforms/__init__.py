"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.data.transforms')"
"""
# flake8: noqa

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.data.transforms import augmenter_base
    from netharn.data.transforms import augmenters
    from netharn.data.transforms.augmenter_base import (ParamatarizedAugmenter,)
    from netharn.data.transforms.augmenters import (HSVShift, LetterboxResize,
                                                    demodata_hsv_image,)
    # </AUTOGEN_INIT>
