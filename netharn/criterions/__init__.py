"""
mkinit netharn.criterions
"""
# flake8: noqa
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, TripletMarginLoss

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.criterions import contrastive_loss
    from netharn.criterions import focal
    from netharn.criterions.contrastive_loss import (ContrastiveLoss,)
    from netharn.criterions.focal import (FocalLoss, one_hot_embedding,)
