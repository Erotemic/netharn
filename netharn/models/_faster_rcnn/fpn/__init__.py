"""
mkinit ~/code/netharn/netharn/models/faster_rcnn/fpn --noattrs
"""
from netharn.models.faster_rcnn.fpn import fpn
from netharn.models.faster_rcnn.fpn import resnet

__all__ = ['fpn', 'resnet']
