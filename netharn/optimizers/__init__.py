"""
mkinit netharn.optimizers')"
"""
# flake8: noqa

from torch.optim import SGD, Adam

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    
    # </AUTOGEN_INIT>
    pass
