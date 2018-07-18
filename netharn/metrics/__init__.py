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
    from netharn.metrics import clf_report
    from netharn.metrics import detections
    from netharn.metrics import sklearn_alts

    from netharn.metrics.clf_report import (classification_report,
                                            ovr_classification_report,)
    from netharn.metrics.detections import (ave_precisions, detection_confusions,)
    from netharn.metrics.sklearn_alts import (class_accuracy_from_confusion,
                                              confusion_matrix,
                                              global_accuracy_from_confusion,)

    __all__ = ['ave_precisions', 'class_accuracy_from_confusion',
               'classification_report', 'clf_report', 'confusion_matrix',
               'detection_confusions', 'detections',
               'global_accuracy_from_confusion', 'ovr_classification_report',
               'sklearn_alts']
