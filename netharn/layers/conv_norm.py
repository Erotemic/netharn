import ubelt as ub
import torch
from netharn.output_shape_for import OutputShapeFor
import six
from netharn import util
from torch import nn
if six.PY2:
    from fractions import gcd
else:
    from math import gcd


def rectify_nonlinearity(key=ub.NoParam, dim=2):
    """
    Allows dictionary based specification of a nonlinearity

    Example:
        >>> rectify_nonlinearity('relu')
        ReLU(inplace)
        >>> rectify_nonlinearity('leaky_relu')
        LeakyReLU(negative_slope=0.01, inplace)
        >>> rectify_nonlinearity(None)
        None
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'relu'

    if isinstance(key, six.string_types):
        if key == 'relu':
            key = {'type': 'relu'}
        elif key == 'leaky_relu':
            key = {'type': 'leaky_relu', 'negative_slope': 1e-2}
        else:
            raise KeyError(key)
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))

    noli_type = key.pop('type')
    if 'inplace' not in key:
        key['inplace'] = True

    if noli_type == 'leaky_relu':
        cls = torch.nn.LeakyReLU
    elif noli_type == 'relu':
        cls = torch.nn.ReLU
    else:
        raise KeyError('unknown type: {}'.format(key))
    return cls(**key)


def rectify_normalizer(in_channels, key=ub.NoParam, dim=2):
    """
    Allows dictionary based specification of a normalizing layer

    Example:
        >>> rectify_normalizer(8)
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, 'batch')
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, {'type': 'batch'})
        BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, 'group')
        GroupNorm(8, 8, eps=1e-05, affine=True)
        >>> rectify_normalizer(8, {'type': 'group', 'num_groups': 2})
        GroupNorm(2, 8, eps=1e-05, affine=True)
        >>> rectify_normalizer(8, dim=3)
        BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> rectify_normalizer(8, None)
        None
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'batch'

    if isinstance(key, six.string_types):
        if key == 'batch':
            key = {'type': 'batch'}
        elif key == 'group':
            key = {'type': 'group', 'num_groups': ('gcd', min(in_channels, 32))}
        elif key == 'batch+group':
            key = {'type': 'batch+group'}
        else:
            raise KeyError(key)
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))

    norm_type = key.pop('type')
    if norm_type == 'batch':
        in_channels_key = 'num_features'

        if dim == 1:
            cls = torch.nn.BatchNorm1d
        elif dim == 2:
            cls = torch.nn.BatchNorm2d
        elif dim == 3:
            cls = torch.nn.BatchNorm3d
        else:
            raise ValueError(dim)
    elif norm_type == 'group':
        in_channels_key = 'num_channels'
        if isinstance(key['num_groups'], tuple):
            if key['num_groups'][0] == 'gcd':
                key['num_groups'] = gcd(
                    key['num_groups'][1], in_channels)
        if in_channels % key['num_groups'] != 0:
            raise AssertionError(
                'Cannot divide n_inputs {} by num groups {}'.format(
                    in_channels, key['num_groups']))
        cls = torch.nn.GroupNorm
    elif norm_type == 'batch+group':
        return torch.nn.Sequential(
            rectify_normalizer(in_channels, 'batch', dim=dim),
            rectify_normalizer(in_channels, ub.dict_union({'type': 'group'}, key), dim=dim),
        )
    else:
        raise KeyError('unknown type: {}'.format(key))
    assert in_channels_key not in key
    key[in_channels_key] = in_channels
    return cls(**key)


def rectify_conv(dim=2):
    conv_cls = {
        1: torch.nn.Conv1d,
        2: torch.nn.Conv2d,
        3: torch.nn.Conv3d,
    }[dim]
    return conv_cls


def rectify_dropout(dim=2):
    conv_cls = {
        1: torch.nn.Dropout,
        2: torch.nn.Dropout2d,
        3: torch.nn.Dropout3d,
    }[dim]
    return conv_cls


def rectify_maxpool(dim=2):
    conv_cls = {
        1: torch.nn.MaxPool1d,
        2: torch.nn.MaxPool2d,
        3: torch.nn.MaxPool3d,
    }[dim]
    return conv_cls


class _ConvNormNd(torch.nn.Sequential, util.ModuleMixin):
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
        >>> from netharn.layers.conv_norm import _ConvNormNd
        >>> self = _ConvNormNd(dim=2, in_channels=16, out_channels=64,
        >>>                    kernel_size=3)
        >>> print(self)
        _ConvNormNd(
          (conv): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1))
          (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (noli): ReLU(inplace)
        )
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, noli='relu',
                 norm='batch'):
        super(_ConvNormNd, self).__init__()

        conv_cls = rectify_conv(dim)
        conv = conv_cls(in_channels, out_channels, kernel_size=kernel_size,
                        padding=padding, stride=stride, groups=groups,
                        bias=bias, dilation=dilation)

        norm = rectify_normalizer(out_channels, norm, dim=dim)
        noli = rectify_nonlinearity(noli, dim=dim)

        self.add_module('conv', conv)
        if norm:
            self.add_module('norm', norm)
        if noli:
            self.add_module('noli', noli),

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._dim = dim

    def output_shape_for(self, input_shape):
        return OutputShapeFor.sequential(self, input_shape)

    def hidden_shapes_for(self, input_shape):
        shape = OutputShapeFor.sequential(self, input_shape)
        return shape, shape


class ConvNorm1d(_ConvNormNd):
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


class ConvNorm2d(_ConvNormNd):
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


class ConvNorm3d(_ConvNormNd):
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


class Sequential(nn.Sequential, util.ModuleMixin):
    """
    Like torch.sequential but implements hidden_shapes_for

    CommandLine:
        xdoctest -m ~/code/netharn/netharn/layers/conv_norm.py Sequential

    Example:
        >>> self = Sequential(
        >>>     nn.Conv2d(2, 3, kernel_size=3),
        >>>     nn.Conv2d(3, 5, kernel_size=3),
        >>>     nn.Conv2d(5, 7, kernel_size=3),
        >>> )
        >>> shapes, shape = self.hidden_shapes_for([1, 1, 7, 11])
        >>> print('shape = {}'.format(shape))
        >>> print('shapes = {}'.format(ub.repr2(shapes, nl=1)))
        shape = (1, 7, 1, 5)
        shapes = {
            '0': (1, 3, 5, 9),
            '1': (1, 5, 3, 7),
            '2': (1, 7, 1, 5),
        }
    """
    def hidden_shapes_for(self, input_shape):
        from netharn.hidden_shapes_for import HiddenShapesFor
        return HiddenShapesFor.sequential(self, input_shape)

    def output_shape_for(self, input_shape):
        from netharn.output_shape_for import OutputShapeFor
        return OutputShapeFor.sequential(self, input_shape)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.layers.conv_norm all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
