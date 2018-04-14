"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.criterions')"
"""
# flake8: noqa
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, TripletMarginLoss

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.criterions import contrastive_loss
    from netharn.criterions import focal
    from netharn.criterions.contrastive_loss import (ContrastiveLoss,)
    from netharn.criterions.focal import (FocalLoss, one_hot_embedding,)
