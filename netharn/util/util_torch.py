import numpy as np
import ubelt as ub
import torch


class ModuleMixin(object):
    """
    Adds convenince functions to a torch module
    """
    def number_of_parameters(self, trainable=True):
        return number_of_parameters(self, trainable)


def number_of_parameters(model, trainable=True):
    """
    Returns number of trainable parameters in a torch module

    Example:
        >>> model = nh.models.ToyNet2d()
        >>> number_of_parameters(model)
        824
    """
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params


class grad_context(object):
    """
    Context manager for controlling if autograd is enabled.
    """
    def __init__(self, flag):
        if tuple(map(int, torch.__version__.split('.')[0:2])) < (0, 4):
            self.prev = None
            self.flag = flag
        else:
            self.prev = torch.is_grad_enabled()
            self.flag = flag

    def __enter__(self):
        if self.prev is not None:
            torch.set_grad_enabled(self.flag)

    def __exit__(self, *args):
        if self.prev is not None:
            torch.set_grad_enabled(self.prev)
            return False


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
        n_feats_key = 'num_features'

        if dim == 1:
            cls = torch.nn.BatchNorm1d
        elif dim == 2:
            cls = torch.nn.BatchNorm2d
        elif dim == 3:
            cls = torch.nn.BatchNorm3d
        else:
            # might not work
            cls = torch.nn._BatchNorm

    elif norm_type == 'group':
        n_feats_key = 'num_channels'
        key['num_groups'] = min(in_channels, key['num_groups'])
        if in_channels % key['num_groups'] != 0:
            raise AssertionError(
                'Cannot divide n_inputs {} by num groups {}'.format(
                    in_channels, key['num_groups']))
        cls = torch.nn.GroupNorm
    else:
        raise KeyError('unknown type: {}'.format(key))
    assert n_feats_key not in key
    key[n_feats_key] = in_channels
    return cls(**key)
