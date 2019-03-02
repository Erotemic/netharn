"""
mkinit netharn.optimizers
"""
# flake8: noqa

from torch.optim import SGD, Adam

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.optimizers import adamw

    from netharn.optimizers.adamw import (AdamW,)

    __all__ = ['AdamW', 'adamw']
    # </AUTOGEN_INIT>
    pass
