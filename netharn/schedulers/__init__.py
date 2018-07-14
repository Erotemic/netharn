"""
mkinit netharn.schedulers
"""

__DYNAMIC__ = False
if __DYNAMIC__:
    import mkinit
    exec(mkinit.dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.schedulers import core
    from netharn.schedulers import iteration_lr
    from netharn.schedulers import listed

    from netharn.schedulers.core import (CommonMixin, NetharnScheduler,
                                         TorchNetharnScheduler, YOLOScheduler,)
    from netharn.schedulers.listed import (BatchLR, Exponential, ListedLR,)

    __all__ = ['BatchLR', 'CommonMixin', 'Exponential', 'ListedLR',
               'NetharnScheduler', 'TorchNetharnScheduler', 'YOLOScheduler',
               'core', 'iteration_lr', 'listed']
