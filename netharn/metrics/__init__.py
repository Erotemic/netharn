"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.metrics')"
python -m netharn
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.metrics import detections
    from netharn.metrics.detections import (EvaluateVOC, detection_confusions,
                                            iou_overlap,)
