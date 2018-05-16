"""
mkinit netharn.models')"
"""
# flake8: noqa

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.models import toynet
    from netharn.models.toynet import (ToyNet1d, ToyNet2d,)
