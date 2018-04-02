"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.schedulers')"
"""
# flake8: noqa

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.schedulers import listed
    from netharn.schedulers.listed import (ListedLR,)
