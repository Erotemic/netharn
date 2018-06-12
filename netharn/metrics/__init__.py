"""
mkinit netharn.metrics
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
    from netharn.metrics.sklearn_alts import (class_accuracy_from_confusion,
                                              confusion_matrix,
                                              global_accuracy_from_confusion,)

    __all__ = ['ave_precisions', 'class_accuracy_from_confusion',
               'confusion_matrix', 'detection_confusions', 'detections',
               'global_accuracy_from_confusion', 'iou_overlap', 'sklearn_alts']
