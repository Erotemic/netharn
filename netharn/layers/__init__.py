"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.layers')"
"""
# flake8: noqa

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    pass
