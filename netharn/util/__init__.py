"""
python -c "import ubelt._internal as a; a.autogen_init('netharn.util')"
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
    from netharn.util import profiler
    from netharn.util import torch_utils
    from netharn.util import util_fname
    from netharn.util import util_idstr
    from netharn.util.profiler import (IS_PROFILING, IS_PROFILING, KernprofParser,
                                       dump_global_profile_report, dynamic_profile,
                                       find_parent_class, find_pattern_above_row,
                                       find_pyclass_above_row, profile, profile,
                                       profile_onthefly,)
    from netharn.util.torch_utils import (grad_context, number_of_parameters,)
    from netharn.util.util_fname import (align_paths, check_aligned, dumpsafe,
                                         shortest_unique_prefixes,
                                         shortest_unique_suffixes,)
    from netharn.util.util_idstr import (compact_idstr, make_idstr,
                                         make_short_idstr,)
