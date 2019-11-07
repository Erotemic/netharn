# -*- coding: utf-8 -*-
"""
mkinit netharn.criterions
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss

__extra_all__ = [
    'CrossEntropyLoss', 'MSELoss',
]

# <AUTOGEN_INIT>
from netharn.criterions import contrastive_loss
from netharn.criterions import focal
from netharn.criterions import triplet

from netharn.criterions.contrastive_loss import (ContrastiveLoss,)
from netharn.criterions.focal import (ELEMENTWISE_MEAN, FocalLoss,
                                      nll_focal_loss, one_hot_embedding,)
from netharn.criterions.triplet import (TripletLoss, all_pairwise_distances,)

__all__ = ['ContrastiveLoss', 'CrossEntropyLoss', 'ELEMENTWISE_MEAN',
           'FocalLoss', 'MSELoss', 'TripletLoss', 'all_pairwise_distances',
           'contrastive_loss', 'focal', 'nll_focal_loss', 'one_hot_embedding',
           'triplet']
# </AUTOGEN_INIT>
