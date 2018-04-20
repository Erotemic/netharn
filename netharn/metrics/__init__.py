"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.metrics')"
python -m netharn
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = False
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.metrics import detections
    from netharn.metrics import sklearn_alts
    from netharn.metrics.detections import (ave_precisions, detection_confusions,
                                            iou_overlap,)
    from netharn.metrics.sklearn_alts import (confusion_matrix,)
