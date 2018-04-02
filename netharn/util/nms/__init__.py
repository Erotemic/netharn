"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.util')"
python -m netharn
"""
# flake8: noqa
__all__ = ['non_max_supression']

from netharn.util.nms import nms_core
from netharn.util.nms.nms_core import (non_max_supression,)
