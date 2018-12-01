# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import netharn as nh  # NOQA
import torch  # NOQA
import math
import torch.nn as nn  # NOQA
import torchvision
import ubelt as ub
import numpy as np
from netharn.output_shape_for import OutputShapeFor
try:
    from netharn.device import MountedModel
except ImportError:
    MountedModel = None

REGISTERED_RECEPTIVE_FEILDS = []


SHAPE_CLS = tuple  # We exepct shapes to be specified as this class


def compute_type(type):
    def _wrap(func):
        if type is not None:
            REGISTERED_RECEPTIVE_FEILDS.append((type, func))
        return func
    return _wrap


class _MixinPrimatives(object):
    """
    Receptive field formulas for pytorch primatives
    """

    @staticmethod
    def input(prev=None):
        """
        Basic input receptive field is just a single pixel
        """
        if prev is not None:
            raise ValueError('nothing can precede the input')
        prev = {
            'stride': np.array([1, 1]),  # receptive field stride
            'size': np.array([1, 1]),  # receptive field size
            # Note: at the end, assert start_right - start_left == j
            'start_center': np.array([0.5, 0.5]),  # upper left corner center
            # Sides of the start pixel
            'start_left': np.array([0.0, 0.0]),  # upper left corner (left side)
            'start_right': np.array([1.0, 1.0]),  # upper left corner (right side)
        }
        return prev, prev

    @staticmethod
    def _conv(module, prev=None):
        """ Receptive field formula for convolutional layers """
        if prev is None:
            prev = ReceptiveFieldFor.input()[1]
        k = np.array(module.kernel_size)
        s = np.array(module.stride)
        p = np.array(module.padding)
        field = {
            'stride': prev['stride'] * s,
            'size': prev['size'] + (k - 1) * prev['stride'],
            'start_center': prev['start_center'] + ((k - 1) / 2 - p) * prev['stride'],
            'start_left': prev['start_left'] + (np.floor((k - 1) / 2) - p) * prev['stride'],
            'start_right': prev['start_right'] + (np.ceil((k - 1) / 2) - p) * prev['stride'],
        }
        return field, field

    @staticmethod
    def _pool(module, prev=None):
        """ Receptive field formula for pooling layers """
        if prev is None:
            prev = ReceptiveFieldFor.input()[1]
        ndim = 2
        k = np.array([module.kernel_size] * ndim)
        s = np.array([module.stride] * ndim)
        p = np.array([module.padding] * ndim)

        field = {
            'stride': prev['stride'] * s,
            'size': prev['size'] + (k - 1) * prev['stride'],
            'start_center': prev['start_center'] + ((k - 1) / 2 - p) * prev['stride'],
            'start_left': prev['start_left'] + (np.floor((k - 1) / 2) - p) * prev['stride'],
            'start_right': prev['start_right'] + (np.ceil((k - 1) / 2) - p) * prev['stride'],
        }
        return field, field

    @staticmethod
    def _unchanged(module, prev=None):
        """ Formula for layers that do not change the receptive field """
        if prev is None:
            prev = ReceptiveFieldFor.input()[1]
        return prev, prev

    @staticmethod
    @compute_type(nn.Linear)
    def linear(module, prev=None):
        # Linear layers (sort-of) dont change the RF
        return ReceptiveFieldFor._unchanged(module, prev)
        # Perhaps we could do this if we knew the input shape
        # raise NotImplementedError(
        #     'Cannot compute receptive field size on a Linear layer')

    @compute_type(nn.modules.conv._ConvNd)
    def convnd(module, prev=None):
        return ReceptiveFieldFor._conv(module, prev)

    @staticmethod
    @compute_type(nn.modules.pooling._MaxPoolNd)
    def maxpoolnd(module, prev=None):
        return ReceptiveFieldFor._pool(module, prev)

    @staticmethod
    @compute_type(nn.modules.pooling._AvgPoolNd)
    def avepoolnd(module, prev=None):
        return ReceptiveFieldFor._pool(module, prev)

    @staticmethod
    @compute_type(nn.ReLU)
    def relu(module, prev=None):
        return ReceptiveFieldFor._unchanged(module, prev)

    @staticmethod
    @compute_type(nn.LeakyReLU)
    def leaky_relu(module, prev=None):
        return ReceptiveFieldFor._unchanged(module, prev)

    @staticmethod
    @compute_type(nn.modules.batchnorm._BatchNorm)
    def batchnorm(module, prev=None):
        return ReceptiveFieldFor._unchanged(module, prev)

    @staticmethod
    @compute_type(nn.modules.dropout._DropoutNd)
    def dropout(module, prev=None):
        return ReceptiveFieldFor._unchanged(module, prev)

    @staticmethod
    @compute_type(nn.Sequential)
    def sequential(module, prev=None):
        """
        Example:
            >>> self = nn.Sequential(
            >>>     nn.Conv2d(2, 3, kernel_size=3),
            >>>     nn.Conv2d(3, 5, kernel_size=3),
            >>>     nn.Conv2d(5, 7, kernel_size=3),
            >>> )
            >>> rfields, rfield = ReceptiveFieldFor(self)()
            >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1, with_dtype=False)))
            rfield = {
                'size': np.array([7, 7]),
                'start_center': np.array([3.5, 3.5]),
                'start_left': np.array([3., 3.]),
                'start_right': np.array([4., 4.]),
                'stride': np.array([1, 1]),
            }
        """
        if prev is None:
            prev = ReceptiveFieldFor.input()[1]
        rfield = prev
        rfields = ub.odict()
        for key, child in module._modules.items():
            if hasattr(child, 'receptive_field_for'):
                rfields[key], rfield = child.receptive_field_for(rfield)
            else:
                rfields[key], rfield = ReceptiveFieldFor(child)(rfield)
        return rfields, rfield

    @staticmethod
    @compute_type(torch.nn.DataParallel)
    def data_parallel(module, *args, **kw):
        return ReceptiveFieldFor(module.module)(*args, **kw)


class _TorchvisionMixin(object):
    """
    Compute receptive fields for components of torchvision models
    """

    @staticmethod
    @compute_type(torchvision.models.resnet.BasicBlock)
    def resent_basic_block(module, prev=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> import torchvision  # NOQA
            >>> module = torchvision.models.resnet18().layer1[0]
            >>> fields, field = ReceptiveFieldFor(module)()
            >>> print(ub.repr2(fields, nl=2, with_dtype=False))
        """
        if prev is None:
            prev = ReceptiveFieldFor.input()[1]
        rfields = ub.odict()

        residual_field = prev
        rfield = prev

        rfields['conv1'], rfield = ReceptiveFieldFor(module.conv1)(rfield)
        rfields['bn1'], rfield = ReceptiveFieldFor(module.bn1)(rfield)
        rfields['relu1'], rfield = ReceptiveFieldFor(module.relu)(rfield)

        rfields['conv2'], rfield = ReceptiveFieldFor(module.conv2)(rfield)
        rfields['bn2'], rfield = ReceptiveFieldFor(module.bn2)(rfield)
        rfields['relu2'], rfield = ReceptiveFieldFor(module.relu)(rfield)

        if module.downsample is not None:
            rfields['downsample'], residual_field = ReceptiveFieldFor(module.downsample)(prev)

        rfield = ReceptiveFieldFor(module.relu)(rfield)
        return rfields, rfield

    @staticmethod
    @compute_type(torchvision.models.resnet.Bottleneck)
    def resent_bottleneck(module, prev=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> import torchvision  # NOQA
            >>> module = torchvision.models.resnet50().layer1[0]
            >>> fields, field = ReceptiveFieldFor(module)()
            >>> print(ub.repr2(fields[-1], nl=1, with_dtype=False))
        """
        if prev is None:
            prev = ReceptiveFieldFor.input()[1]
        residual_field = prev
        rfield = prev

        rfields = ub.odict()

        rfields['conv1'], rfield = ReceptiveFieldFor(module.conv1)(rfield)
        rfields['bn1'], rfield = ReceptiveFieldFor(module.bn1)(rfield)
        rfields['relu1'], rfield = ReceptiveFieldFor(module.relu)(rfield)

        rfields['conv2'], rfield = ReceptiveFieldFor(module.conv2)(rfield)
        rfields['bn2'], rfield = ReceptiveFieldFor(module.bn2)(rfield)
        rfields['relu2'], rfield = ReceptiveFieldFor(module.relu)(rfield)

        rfields['conv3'], rfield = ReceptiveFieldFor(module.conv3)(rfield)
        rfields['bn3'], rfield = ReceptiveFieldFor(module.bn3)(rfield)

        if module.downsample is not None:
            rfields['downsample'], residual_field = ReceptiveFieldFor(module.downsample)(prev)

        rfield = ReceptiveFieldFor(module.relu)(rfield)
        return rfield

    @staticmethod
    @compute_type(torchvision.models.resnet.ResNet)
    def resnet_model(module, prev=None, input_shape=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> from netharn.receptive_field_for import *
            >>> module = torchvision.models.resnet50()
            >>> input_shape = (1, 3, 224, 224)
            >>> fields, field = ReceptiveFieldFor(module)(input_shape=input_shape)
            >>> print(ub.repr2(field, nl=1, with_dtype=False))

        Ignore:
            >>> input_shape = (1, 3, 448, 448)

            OutputShapeFor(module)(input_shape)
        """
        if prev is None:
            prev = ReceptiveFieldFor.input()[1]
        rfield = prev
        rfields = ub.odict()
        rfields['conv1'], rfield = ReceptiveFieldFor(module.conv1)(rfield)
        rfields['bn1'], rfield = ReceptiveFieldFor(module.bn1)(rfield)
        rfields['relu1'], rfield = ReceptiveFieldFor(module.relu)(rfield)
        rfields['maxpool'], rfield = ReceptiveFieldFor(module.maxpool)(rfield)

        rfields['layer1'], rfield = ReceptiveFieldFor(module.layer1)(rfield)
        rfields['layer2'], rfield = ReceptiveFieldFor(module.layer2)(rfield)
        rfields['layer3'], rfield = ReceptiveFieldFor(module.layer3)(rfield)
        rfields['layer4'], rfield = ReceptiveFieldFor(module.layer4)(rfield)

        rfields['avgpool'], rfield = ReceptiveFieldFor(module.avgpool)(rfield)

        if input_shape is None:
            raise ValueError('input shape is required')

        shape = input_shape
        shape = OutputShapeFor(module.conv1)(shape)
        shape = OutputShapeFor(module.bn1)(shape)
        shape = OutputShapeFor(module.relu)(shape)
        shape = OutputShapeFor(module.maxpool)(shape)
        shape = OutputShapeFor(module.layer1)(shape)
        shape = OutputShapeFor(module.layer2)(shape)
        shape = OutputShapeFor(module.layer3)(shape)
        shape = OutputShapeFor(module.layer4)(shape)
        shape = OutputShapeFor(module.avgpool)(shape)

        spatial_shape = np.array(shape[2:])

        # Keep everything the same except increase the RF size
        # based on how many output pixels there are.
        rfield_flatten = rfield.copy()
        # not sure if this is 100% correct
        rfield_flatten['size'] = rfield['size'] + (spatial_shape - 1) * rfield['stride']
        rfields['flatten'] = rfield = rfield_flatten

        # The reshape operation will blend the receptive fields of the inputs
        # but it will depend on the output shape of the layer.
        # rfield = (rfield[0], prod(rfield[1:]))

        rfields['fc'], rfield = ReceptiveFieldFor(module.fc)(rfield)
        return rfields, rfield


class ReceptiveFieldFor(_MixinPrimatives, _TorchvisionMixin):
    """
    Knows how to compute the receptive fields for many pytorch primatives and
    some torchvision components.

    References:
        https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

    Returns:
        Tuple[object, Dict]:
            fields: object: The hidden layer recepvive fields (can be complex due to nesting)
            field: Dict: a dictionary containing receptive field information.

    Notes:
        A 1-D Pixel
            +-----+
            ^  ^  ^
          left |  L right
               |
             center

    Example:
        >>> # Case where we have a registered func
        >>> self = nn.Sequential(
        >>>     nn.Conv2d(2, 3, kernel_size=3),
        >>>     nn.Conv2d(3, 5, kernel_size=3),
        >>> )
        >>> rfields, rfield = ReceptiveFieldFor(self)()
        >>> print('rfields = {}'.format(ub.repr2(rfields, nl=3)))
        >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1)))
        rfields = {
            '0': {
                'size': np.array([3, 3], dtype=np.int64),
                'start_center': np.array([1.5, 1.5], dtype=np.float64),
                'start_left': np.array([1., 1.], dtype=np.float64),
                'start_right': np.array([2., 2.], dtype=np.float64),
                'stride': np.array([1, 1], dtype=np.int64),
            },
            '1': {
                'size': np.array([5, 5], dtype=np.int64),
                'start_center': np.array([2.5, 2.5], dtype=np.float64),
                'start_left': np.array([2., 2.], dtype=np.float64),
                'start_right': np.array([3., 3.], dtype=np.float64),
                'stride': np.array([1, 1], dtype=np.int64),
            },
        }
        rfield = {
            'size': np.array([5, 5], dtype=np.int64),
            'start_center': np.array([2.5, 2.5], dtype=np.float64),
            'start_left': np.array([2., 2.], dtype=np.float64),
            'start_right': np.array([3., 3.], dtype=np.float64),
            'stride': np.array([1, 1], dtype=np.int64),
        }

    Example:
        >>> # Case where we haven't registered a func
        >>> # In this case rfields is not populated (but rfield is)
        >>> self = nn.Conv2d(2, 3, kernel_size=3)
        >>> rfields, rfield = ReceptiveFieldFor(self)()
        >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1)))
        rfield = {
            'size': np.array([3, 3], dtype=np.int64),
            'start_center': np.array([1.5, 1.5], dtype=np.float64),
            'start_left': np.array([1., 1.], dtype=np.float64),
            'start_right': np.array([2., 2.], dtype=np.float64),
            'stride': np.array([1, 1], dtype=np.int64),
        }

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import torchvision  # NOQA
        >>> module = torchvision.models.alexnet().features
        >>> fields, field = ReceptiveFieldFor(module)()
        >>> print(ub.repr2(fields[-1], nl=1, with_dtype=False))
        {
            'size': np.array([195, 195]),
            'start_center': np.array([31.5, 31.5]),
            'start_left': np.array([31., 31.]),
            'start_right': np.array([32., 32.]),
            'stride': np.array([32, 32]),
        }
    """
    math = math  # for hacking in sympy

    def __init__(self, module):
        self.module = module
        self._func = getattr(module, 'receptive_field_for', None)
        if self._func is None:
            # Lookup rfield func if we can't find it
            for type, _func in REGISTERED_RECEPTIVE_FEILDS:
                try:
                    if module is type or isinstance(module, type):
                        self._func = _func
                except TypeError:
                    pass
            if not self._func:
                raise TypeError('Unknown (rf) module type {}'.format(module))

    def __call__(self, *args, **kwargs):
        if isinstance(self.module, nn.Module):
            # bound methods dont need module
            is_bound  = hasattr(self._func, '__func__') and getattr(self._func, '__func__', None) is not None
            is_bound |= hasattr(self._func, 'im_func') and getattr(self._func, 'im_func', None) is not None
            if is_bound:
                rfields, rfield = self._func(*args, **kwargs)
            else:
                # nn.Module with state
                rfields, rfield = self._func(self.module, *args, **kwargs)
        else:
            # a simple pytorch func
            rfields, rfield = self._func(*args, **kwargs)
        return rfields, rfield


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.receptive_field_for
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
