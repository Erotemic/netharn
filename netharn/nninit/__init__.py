"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.nninit')"
python -m netharn
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = True
if __DYNAMIC__:
    from ubelt._internal import dynamic_import
    exec(dynamic_import(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.nninit import base
    from netharn.nninit import core
    from netharn.nninit import lsuv
    from netharn.nninit.base import (NoOp, apply_initializer, load_partial_state,
                                     trainable_layers,)
    from netharn.nninit.core import (KaimingNormal, KaimingUniform, Orthogonal,
                                     Pretrained,)
    from netharn.nninit.lsuv import (LSUV, Orthonormal, svd_orthonormal,)
