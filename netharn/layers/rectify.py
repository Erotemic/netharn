import six
import torch
import ubelt as ub
if six.PY2:
    from fractions import gcd
else:
    from math import gcd


def rectify_nonlinearity(key=ub.NoParam, dim=2):
    """
    Allows dictionary based specification of a nonlinearity

    Example:
        >>> rectify_nonlinearity('relu')
        ReLU(...)
        >>> rectify_nonlinearity('leaky_relu')
        LeakyReLU(negative_slope=0.01...)
        >>> rectify_nonlinearity(None)
        None
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'relu'

    if isinstance(key, six.string_types):
        key = {'type': key}
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))
    kw = key
    noli_type = kw.pop('type')
    if 'inplace' not in kw:
        kw['inplace'] = True

    if noli_type == 'leaky_relu':
        cls = torch.nn.LeakyReLU
    elif noli_type == 'relu':
        cls = torch.nn.ReLU
    elif noli_type == 'elu':
        cls = torch.nn.ELU
    elif noli_type == 'celu':
        cls = torch.nn.CELU
    elif noli_type == 'selu':
        cls = torch.nn.SELU
    elif noli_type == 'relu6':
        cls = torch.nn.ReLU6
    else:
        raise KeyError('unknown type: {}'.format(kw))
    return cls(**kw)


def rectify_normalizer(in_channels, key=ub.NoParam, dim=2, **kwargs):
    """
    Allows dictionary based specification of a normalizing layer

    Args:
        in_channels (int): number of input channels
        dim (int): dimensionality
        **kwargs: extra args

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
        >>> rectify_normalizer(8, key={'type': 'syncbatch'})
    """
    if key is None:
        return None

    if key is ub.NoParam:
        key = 'batch'

    if isinstance(key, six.string_types):
        key = {'type': key}
    elif isinstance(key, dict):
        key = key.copy()
    else:
        raise TypeError(type(key))

    norm_type = key.pop('type')
    if norm_type == 'batch':
        in_channels_key = 'num_features'

        if dim == 0:
            cls = torch.nn.BatchNorm1d
        elif dim == 1:
            cls = torch.nn.BatchNorm1d
        elif dim == 2:
            cls = torch.nn.BatchNorm2d
        elif dim == 3:
            cls = torch.nn.BatchNorm3d
        else:
            raise ValueError(dim)
    elif norm_type == 'syncbatch':
        in_channels_key = 'num_features'
        cls = torch.nn.SyncBatchNorm
    elif norm_type == 'group':
        in_channels_key = 'num_channels'
        if key.get('num_groups') is None:
            key['num_groups'] = ('gcd', min(in_channels, 32))

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

    try:
        import copy
        kw = copy.copy(key)
        kw.update(kwargs)
        return cls(**kw)
    except Exception:
        # Ignore kwargs
        import warnings
        warnings.warn('kwargs ignored in rectify normalizer')
        return cls(**key)


def rectify_conv(dim=2):
    conv_cls = {
        0: torch.nn.Linear,
        1: torch.nn.Conv1d,
        2: torch.nn.Conv2d,
        3: torch.nn.Conv3d,
    }[dim]
    return conv_cls


def rectify_dropout(dim=2):
    conv_cls = {
        0: torch.nn.Dropout,
        1: torch.nn.Dropout,
        2: torch.nn.Dropout2d,
        3: torch.nn.Dropout3d,
    }[dim]
    return conv_cls


def rectify_maxpool(dim=2):
    conv_cls = {
        0: torch.nn.MaxPool1d,
        1: torch.nn.MaxPool1d,
        2: torch.nn.MaxPool2d,
        3: torch.nn.MaxPool3d,
    }[dim]
    return conv_cls
