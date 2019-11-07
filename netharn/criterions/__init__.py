# -*- coding: utf-8 -*-
"""
mkinit netharn.criterions
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, TripletMarginLoss

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    # from . import contrastive_loss
    from . import focal
    from .contrastive_loss import (ContrastiveLoss,)
    from .focal import (FocalLoss, one_hot_embedding,)

    __all__ = ['ContrastiveLoss', 'FocalLoss', 'contrastive_loss', 'focal',
               'one_hot_embedding']
