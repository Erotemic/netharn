"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.optimizers')"
"""
# flake8: noqa

from torch.optim import SGD, Adam

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    # </AUTOGEN_INIT>
    pass

