"""
mkinit netharn.schedulers
"""

__DYNAMIC__ = False
if __DYNAMIC__:
    import mkinit
    exec(mkinit.dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.schedulers import listed

    from netharn.schedulers.listed import (Exponential, ListedLR,)

    __all__ = ['Exponential', 'ListedLR', 'listed']
