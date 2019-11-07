# -*- coding: utf-8 -*-
"""
mkinit netharn.initializers
python -m netharn
python -c "from netharn.initializers import *"
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# <AUTOGEN_INIT>
from netharn.initializers import lsuv
from netharn.initializers import nninit_base
from netharn.initializers import nninit_core

from netharn.initializers.lsuv import (LSUV, Orthonormal, svd_orthonormal,)
from netharn.initializers.nninit_base import (Initializer, NoOp,
                                              apply_initializer,
                                              load_partial_state,)
from netharn.initializers.nninit_core import (KaimingNormal, KaimingUniform,
                                              Orthogonal, Pretrained,)

__all__ = ['Initializer', 'KaimingNormal', 'KaimingUniform', 'LSUV', 'NoOp',
           'Orthogonal', 'Orthonormal', 'Pretrained', 'apply_initializer',
           'load_partial_state', 'lsuv', 'nninit_base', 'nninit_core',
           'svd_orthonormal']
# </AUTOGEN_INIT>
