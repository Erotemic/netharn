"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.initializers')"
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
    from netharn.initializers import base
    from netharn.initializers import core
    from netharn.initializers import lsuv
    from netharn.initializers.base import (NoOp, apply_initializer, load_partial_state,
                                     trainable_layers,)
    from netharn.initializers.core import (KaimingNormal, KaimingUniform, Orthogonal,
                                     Pretrained,)
    from netharn.initializers.lsuv import (LSUV, Orthonormal, svd_orthonormal,)
