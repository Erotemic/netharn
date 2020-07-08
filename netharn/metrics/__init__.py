"""
DEPRECATED

USE kwcoco.metrics instead!


mkinit netharn.metrics -w
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

# <AUTOGEN_INIT>
from netharn.metrics import assignment
from netharn.metrics import clf_report
from netharn.metrics import confusion_vectors
from netharn.metrics import detect_metrics
from netharn.metrics import drawing
from netharn.metrics import functional
from netharn.metrics import sklearn_alts
from netharn.metrics import voc_metrics

from netharn.metrics.clf_report import (classification_report,
                                        ovr_classification_report,)
from netharn.metrics.confusion_vectors import (BinaryConfusionVectors,
                                               ConfusionVectors, DictProxy,
                                               OneVsRestConfusionVectors,
                                               PR_Result, PerClass_PR_Result,
                                               PerClass_ROC_Result, ROC_Result,
                                               Threshold_Result,)
from netharn.metrics.detect_metrics import (DetectionMetrics,
                                            eval_detections_cli,)
from netharn.metrics.drawing import (draw_perclass_prcurve, draw_perclass_roc,
                                     draw_prcurve, draw_roc,
                                     draw_threshold_curves,)
from netharn.metrics.functional import (fast_confusion_matrix,)
from netharn.metrics.sklearn_alts import (class_accuracy_from_confusion,
                                          confusion_matrix,
                                          global_accuracy_from_confusion,)
from netharn.metrics.voc_metrics import (VOC_Metrics,)

__all__ = ['BinaryConfusionVectors', 'ConfusionVectors', 'DetectionMetrics',
           'DictProxy', 'OneVsRestConfusionVectors', 'PR_Result',
           'PerClass_PR_Result', 'PerClass_ROC_Result', 'ROC_Result',
           'Threshold_Result', 'VOC_Metrics', 'assignment',
           'class_accuracy_from_confusion', 'classification_report',
           'clf_report', 'confusion_matrix', 'confusion_vectors',
           'detect_metrics', 'draw_perclass_prcurve', 'draw_perclass_roc',
           'draw_prcurve', 'draw_roc', 'draw_threshold_curves', 'drawing',
           'eval_detections_cli', 'fast_confusion_matrix', 'functional',
           'global_accuracy_from_confusion', 'ovr_classification_report',
           'sklearn_alts', 'voc_metrics']
