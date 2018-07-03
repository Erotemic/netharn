import ubelt as ub
import torch
from netharn import util
from netharn.output_shape_for import OutputShapeFor


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

    if isinstance(key, str):
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

    if isinstance(key, str):
        if key == 'batch':
            key = {'type': 'batch'}
        elif key == 'group':
            key = {'type': 'group', 'num_groups': 32}
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
        key['num_groups'] = min(in_channels, key['num_groups'])
        if in_channels % key['num_groups'] != 0:
            raise AssertionError(
                'Cannot divide n_inputs {} by num groups {}'.format(
                    in_channels, key['num_groups']))
        cls = torch.nn.GroupNorm
    else:
        raise KeyError('unknown type: {}'.format(key))
    assert in_channels_key not in key
    key[in_channels_key] = in_channels
    return cls(**key)


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
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, padding=0, noli='relu', norm='batch',
                 groups=1):
        super().__init__()

        if dim == 1:
            conv = torch.nn.Conv1d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding,
                                   stride=stride, groups=groups, bias=bias)
        elif dim == 2:
            conv = torch.nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding,
                                   stride=stride, groups=groups, bias=bias)
        elif dim == 3:
            conv = torch.nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding,
                                   stride=stride, groups=groups, bias=bias)
        else:
            raise ValueError(dim)

        norm = util.rectify_normalizer(out_channels, norm, dim=dim)
        noli = util.rectify_nonlinearity(noli, dim=dim)

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
                 bias=True, padding=0, noli='relu', norm='batch',
                 groups=1):
        super().__init__(dim=1, in_channels=in_channels,
                         out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, bias=bias, padding=padding, noli=noli,
                         norm=norm, groups=groups)


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
                 bias=True, padding=0, noli='relu', norm='batch',
                 groups=1):
        super().__init__(dim=2, in_channels=in_channels,
                         out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, bias=bias, padding=padding, noli=noli,
                         norm=norm, groups=groups)


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
        super().__init__(dim=3, in_channels=in_channels,
                         out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, bias=bias, padding=padding, noli=noli,
                         norm=norm, groups=groups)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.layers.conv_norm all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
