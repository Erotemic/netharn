"""
mkinit netharn.schedulers
"""
# flake8: noqa

__DYNAMIC__ = False
if __DYNAMIC__:
    import mkinit
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.schedulers import listed
    from netharn.schedulers.listed import (ListedLR,)
