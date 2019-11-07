# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from netharn.output_shape_for import OutputShapeFor
from netharn.layers import rectify
from netharn.layers import common
import ubelt as ub  # NOQA


class ConvNormNd(common.Sequential):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> from netharn.layers.conv_norm import ConvNormNd
        >>> self = ConvNormNd(dim=2, in_channels=16, out_channels=64,
        >>>                    kernel_size=3)
        >>> print(self)
        ConvNormNd(
          (conv): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))
          (norm): BatchNorm2d(64, ...)
          (noli): ReLU(...)
        )

    Example:
        >>> from netharn.layers.conv_norm import ConvNormNd
        >>> self = ConvNormNd(dim=0, in_channels=16, out_channels=64)
        >>> print(self)
        ConvNormNd(
          (conv): Linear(in_features=16, out_features=64, bias=True)
          (norm): BatchNorm1d(64, ...)
          (noli): ReLU(...)
        )
        >>> input_shape = (None, 16)
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden, nl=1))
        {
            'conv': (None, 64),
            'norm': (None, 64),
            'noli': (None, 64),
        }
        >>> print(ub.repr2(self.receptive_field_for()))
        {
            'crop': np.array([0., 0.], dtype=np.float64),
            'shape': np.array([1., 1.], dtype=np.float64),
            'stride': np.array([1., 1.], dtype=np.float64),
        }
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch'):
        super(ConvNormNd, self).__init__()

        conv_cls = rectify.rectify_conv(dim)
        if dim == 0:
            assert kernel_size == 1
            assert padding == 0
            assert stride == 1
            assert groups == 1
            assert dilation == 1
            conv = conv_cls(in_channels, out_channels, bias=bias,)
        else:
            conv = conv_cls(in_channels, out_channels, kernel_size=kernel_size,
                            padding=padding, stride=stride, groups=groups,
                            bias=bias, dilation=dilation)

        norm = rectify.rectify_normalizer(out_channels, norm, dim=dim)
        noli = rectify.rectify_nonlinearity(noli, dim=dim)

        self.add_module('conv', conv)
        if norm:
            self.add_module('norm', norm)
        if noli:
            self.add_module('noli', noli)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._dim = dim

    def output_shape_for(self, input_shape):
        return OutputShapeFor.sequential(self, input_shape)


class ConvNorm1d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> from netharn.layers.conv_norm import *
        >>> input_shape = [2, 3, 5]
        >>> self = ConvNorm1d(input_shape[1], 7, kernel_size=3)
        >>> OutputShapeFor(self)._check_consistency(input_shape)
        >>> self.output_shape_for(input_shape)
        (2, 7, 3)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch'):
        super(ConvNorm1d, self).__init__(dim=1, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         dilation=dilation, groups=groups)


class ConvNorm2d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> input_shape = [2, 3, 5, 7]
        >>> self = ConvNorm2d(input_shape[1], 11, kernel_size=3)
        >>> OutputShapeFor(self)._check_consistency(input_shape)
        >>> self.output_shape_for(input_shape)
        (2, 11, 3, 5)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch'):
        super(ConvNorm2d, self).__init__(dim=2, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         dilation=dilation, groups=groups)


class ConvNorm3d(ConvNormNd):
    """
    Backbone convolution component. The convolution hapens first, normalization
    and nonlinearity happen after the convolution.

    CONV[->NORM][->NOLI]

    Args:
        norm (str, dict, nn.Module): Type of normalizer,
            if None, then normalization is disabled.
        noli (str, dict, nn.Module): Type of nonlinearity,
            if None, then normalization is disabled.

    Example:
        >>> input_shape = [2, 3, 5, 7, 11]
        >>> self = ConvNorm3d(input_shape[1], 13, kernel_size=3)
        >>> OutputShapeFor(self)._check_consistency(input_shape)
        >>> self.output_shape_for(input_shape)
        (2, 13, 3, 5, 9)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, padding=0, noli='relu', norm='batch',
                 groups=1):
        super(ConvNorm3d, self).__init__(dim=3, in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=bias,
                                         padding=padding, noli=noli, norm=norm,
                                         groups=groups)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.layers.conv_norm all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
