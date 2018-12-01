"""
mkinit netharn.layers
"""
# flake8: noqa

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.layers import conv_norm
    from netharn.layers import reorg
    from netharn.layers import reshape
    from netharn.layers import roi_pooling

    from netharn.layers.conv_norm import (ConvNorm1d, ConvNorm2d, ConvNorm3d,
                                          rectify_conv, rectify_dropout,
                                          rectify_maxpool, rectify_nonlinearity,
                                          rectify_normalizer,)
    from netharn.layers.reshape import (Reshape,)
    from netharn.layers.roi_pooling import (roi_pool,)

    __all__ = ['ConvNorm1d', 'ConvNorm2d', 'ConvNorm3d', 'Reshape', 'conv_norm',
               'rectify_conv', 'rectify_dropout', 'rectify_maxpool',
               'rectify_nonlinearity', 'rectify_normalizer', 'reorg', 'reshape',
               'roi_pool', 'roi_pooling']
