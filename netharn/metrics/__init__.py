"""
mkinit netharn.metrics')"
python -m netharn
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.metrics import detections
    from netharn.metrics import sklearn_alts
    from netharn.metrics.detections import (ave_precisions, detection_confusions,
                                            iou_overlap,)
    from netharn.metrics.sklearn_alts import (confusion_matrix,)
