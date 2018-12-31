# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import six  # NOQA
import torch.nn as nn
import torchvision
import ubelt as ub
import numpy as np
from collections import OrderedDict
from netharn.output_shape_for import OutputShapeFor
try:
    from netharn.device import MountedModel
except ImportError:
    MountedModel = None

REGISTERED_TYPES = []


def ensure_array_nd(data, n):
    if ub.iterable(data):
        return np.array(data)
    else:
        return np.array([data] * n)


def compute_type(*types):
    def _wrap(func):
        for type in types:
            if type is not None:
                REGISTERED_TYPES.append((type, func))
        return func
    return _wrap


# class ReceptiveField(ub.NiceRepr):
#     """ container for holding a receptive feild """
#     def __init__(self, stride, shape, crop):
#         self.data = {
#             'stride': stride,  # The stride / scale factor of the network
#             'shape': shape,  # The receptive feild shape of a single output pixel
#             'crop': crop,  # The amount of cropping / starting pixel location
#         }

#     def __nice__(self):
#         return ub.repr2(self.data, nl=1)

#     def __getitem__(self, key):
#         return self.data[key]

# ReceptiveField = dict

class ReceptiveField(OrderedDict):
    """ container for holding a receptive feild """
    def __init__(self, data, hidden=None):
        OrderedDict.__init__(self, sorted(data.items()))
        self.data = data
        self.hidden = hidden

    def __getitem__(self, key):
        return self.data[key]


class HiddenFields(OrderedDict, ub.NiceRepr):
    """
    Augments normal hidden fields dicts with a convinience setitem
    """
    def __nice__(self):
        return ub.repr2(self, nl=0)

    def __str__(self):
        return ub.NiceRepr.__str__(self)

    def __repr__(self):
        return ub.NiceRepr.__repr__(self)

    def __setitem__(self, key, value):
        if getattr(value, 'hidden', None) is not None:
            # When setting a value to an OutputShape object, if that object has
            # a hidden shape, then use that instead.
            value = value.hidden
        return OrderedDict.__setitem__(self, key, value)

    def shallow(self, n=1):
        """
        Grabs only the shallowest n layers of hidden fields
        """
        if n == 0:
            last = self
            # while isinstance(last, HiddenFields):
            while hasattr(last, 'shallow'):
                last = list(last.values())[-1]
            return last
        else:
            output = OrderedDict()
            for key, value in self.items():
                # if isinstance(value, HiddenFields):
                if hasattr(value, 'shallow'):
                    value = value.shallow(n - 1)
                output[key] = value
            return output


class _TorchMixin(object):
    """
    Receptive field formulas for PyTorch primatives
    """

    @staticmethod
    def input(input_field=None, n=2):
        """
        Basic input receptive field is just a single pixel.
        """
        if input_field is not None:
            raise ValueError('nothing can precede the input')
        input_field = ReceptiveField({
            # The input receptive field stride / scale factor is 1.
            'stride': ensure_array_nd(1.0, n),
            # The input receptive field shape is 1 pixel.
            'shape': ensure_array_nd(1.0, n),
            # Use the coordinate system where the top left corner is 0, 0 ( This is unlike [1], which uses 0.5)
            'crop': ensure_array_nd(0.0, n),
        })
        return input_field

    @staticmethod
    def _kernelized(module, input_field=None, ndim=None):
        """
        Receptive field formula for general sliding kernel based layers
        This works for both convolutional and pooling layers.

        Notes:
            Baseline formulas are from [1]. Information about how to include
            dilation (atrous) convolutions can be found in [2, 3].  Better info
            seems to be available in [4].

            * tensorflow has similar functionality
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/receptive_field/python/util/receptive_field.py

            * To preserve spatial extent, padding should equal `(k - 1) * d / 2`.

        References:
            [1] https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
            [2] http://www.erogol.com/dilated-convolution/
            [3] https://stackoverflow.com/questions/35582521/how-to-calculate-receptive-field-shape
            [4] https://arxiv.org/pdf/1603.07285.pdf

        Example:
            >>> module = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=2, dilation=3)
            >>> field = ReceptiveFieldFor._kernelized(module)
            >>> print(ub.repr2(field, nl=0, with_dtype=False))
            {'crop': np.array([4., 4.]), 'shape': np.array([13., 13.]), 'stride': np.array([2., 2.])}

            >>> module = nn.MaxPool2d(kernel_size=3, stride=2, padding=2, dilation=2)
            >>> field = ReceptiveFieldFor._kernelized(module)
            >>> print(ub.repr2(field, nl=0, with_dtype=False))
            {'crop': np.array([0., 0.]), 'shape': np.array([5., 5.]), 'stride': np.array([2., 2.])}

            >>> module = nn.MaxPool2d(kernel_size=3, stride=2, padding=2, dilation=1)
            >>> field = ReceptiveFieldFor._kernelized(module)
            >>> print(ub.repr2(field, nl=0, with_dtype=False))
            {'crop': np.array([-1., -1.]), 'shape': np.array([3., 3.]), 'stride': np.array([2., 2.])}

            >>> module = nn.AvgPool2d(kernel_size=3, stride=2, padding=2)
            >>> field = ReceptiveFieldFor._kernelized(module)
            >>> print(ub.repr2(field, nl=0, with_dtype=False))
            {'crop': np.array([-1., -1.]), 'shape': np.array([3., 3.]), 'stride': np.array([2., 2.])}
        """
        # impl = ReceptiveFieldFor.impl
        if input_field is None:
            input_field = ReceptiveFieldFor.input()

        # Hack to get the number of space-time dimensions
        if ndim is None:
            try:
                if module.__class__.__name__.endswith('1d'):
                    ndim = 1
                elif module.__class__.__name__.endswith('2d'):
                    ndim = 2
                elif module.__class__.__name__.endswith('3d'):
                    ndim = 3
            except AttributeError:
                if module.__name__.endswith('1d'):
                    ndim = 1
                elif module.__name__.endswith('2d'):
                    ndim = 2
                elif module.__name__.endswith('3d'):
                    ndim = 3
        if ndim is None:
            raise ValueError('Cannot infer ndim from {}'.format(module))

        k = ensure_array_nd(module.kernel_size, ndim)
        s = ensure_array_nd(module.stride, ndim)
        p = ensure_array_nd(module.padding, ndim)
        d = ensure_array_nd(getattr(module, 'dilation', 1), ndim)

        # To calculate receptive feild we first need to find the SUPPORT of
        # this layer. The support is the number/extent of extra surrounding
        # pixels adding this layer will take into account. Given this, we can
        # compute the receptive feild wrt the original input by combining this
        # information with the previous receptive feild.
        #
        # In the normal case (with no dilation, d=1) the support is (k - 1).
        # This is because because the operation is able to see a window of shape
        # k in the input, and produces a single output pixel (hence the k). The
        # center input pixel corresponds with the output, so it does not expand
        # the receptive feild (hence the -1), but all other input pixels do
        # expand the field (thus the k-1).
        #
        # The stride of this layer will not affect the support.
        #
        # The dilation of the current layer DOES impact the support.
        # This expands the effective kernel shape, but it does cause the data
        # each operation sees to become more diffuse. However, even though what
        # it sees in that extent is more diffuse, the RF is just a bound, so we
        # can ignore the diffuseness effect and simply scale the input kernel
        # shape by the dilation amount. Hense we get
        support = (k - 1) * d

        """
        Note the above is correct because:

            import sympy as sym
            k, d = sym.symbols('k, d')

            # Compute the support from formula in 5.1 of [4]
            # To understand the relationship tying the dilation rate d and the
            # output shape o, it is useful to think of the impact of d on the
            # effective kernel shape. A kernel of shape k dilated by a factor d
            # has an effective shape.
            effective_kernel_size = k + (k - 1) * (d - 1)
            support_v1 = sym.expand(effective_kernel_size - 1)

            # Compute support from our method
            support_v2 = sym.expand((k - 1) * d)

            # They are equivalent. QED
            assert sym.Eq(support_v1, support_v2)
        """

        # Compute how many pixels this layer takes off the side Note that an
        # even shape kernel results in half pixel crops.  This is expected and
        # correct. To use the crop in practice take the floor / ceil of the
        # final result, but in this intermediate stage, subpixel crops are
        # perfectly valid.
        crop = ((support / 2) - p)

        field = ReceptiveField({
            # The new stride only depends on the layer stride and the previous
            # stride.
            'stride': input_field['stride'] * s,

            # The stride of the current layer does not impact the receptive
            # feild, however the stride of the previous layer does. This is
            # because each pixel in the incoming layer really corresponds
            # `input_field['stride']` pixels in the original input.
            'shape':   input_field['shape'] + support * input_field['stride'],

            # Padding does not influence the RF shape, but it does influence
            # where the start pixel is (i.e. without the right amount of
            # padding the the edge of the previous layer is cropped).
            'crop': input_field['crop'] + crop * input_field['stride'],
        })
        return field

    @staticmethod
    def _unchanged(module, input_field=None):
        """ Formula for layers that do not change the receptive field """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()
        return input_field

    @staticmethod
    @compute_type(nn.Linear)
    def linear(module, input_field=None):
        # Linear layers (sort-of) dont change the RF
        return ReceptiveFieldFor._unchanged(module, input_field)
        # Perhaps we could do this if we knew the input shape
        # raise NotImplementedError(
        #     'Cannot compute receptive field shape on a Linear layer')

    def _kernelized_tranpose(module, input_field=None):
        """
        Receptive field formula for pooling layers

        Example:
            >>> from netharn.receptive_field_for import *
            >>> from netharn.output_shape_for import *
            >>> module = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=2)
            >>> ReceptiveFieldFor(module)()

            >>> # This network should effectively invert itself
            >>> module = nn.Sequential(ub.odict([
            >>>     #('a', nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)),
            >>>     ('c1', nn.Conv2d(1, 1, kernel_size=3, stride=2)),
            >>>     ('c2', nn.Conv2d(1, 1, kernel_size=3, stride=2)),
            >>>     ('c3', nn.Conv2d(1, 1, kernel_size=3, stride=2)),
            >>>     ('c3T', nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)),
            >>>     ('c2T', nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)),
            >>>     ('c1T', nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2)),
            >>> ]))
            >>> print(ub.repr2(ReceptiveFieldFor(module)()))
            >>> ReceptiveFieldFor(module)()
            >>> OutputShapeFor(module)._check_consistency([1, 1, 32, 32])

            >>> module = nn.Sequential(ub.odict([
            >>>     #('a', nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)),
            >>>     ('c1', nn.Conv2d(1, 1, kernel_size=3, stride=2, dilation=2)),
            >>>     ('c2', nn.Conv2d(1, 1, kernel_size=3, stride=2, dilation=2)),
            >>>     ('c3', nn.Conv2d(1, 1, kernel_size=3, stride=2, dilation=2)),
            >>>     ('c3T', nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, dilation=2)),
            >>>     ('c2T', nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, dilation=2)),
            >>>     ('c1T', nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, dilation=2)),
            >>> ]))
            >>> print(ub.repr2(ReceptiveFieldFor(module)()))

            >>> # This network is pathological
            >>> module = nn.Sequential(ub.odict([
            >>>     #('a', nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)),
            >>>     ('c1', nn.Conv2d(1, 1, kernel_size=3, stride=7, dilation=2)),
            >>>     ('c2', nn.Conv2d(1, 1, kernel_size=5, stride=6, padding=1)),
            >>>     ('c3', nn.Conv2d(1, 1, kernel_size=7, stride=5)),
            >>>     ('c3T', nn.ConvTranspose2d(1, 1, kernel_size=7, stride=6)),
            >>>     ('c2T', nn.ConvTranspose2d(1, 1, kernel_size=5, stride=7, padding=1)),
            >>>     ('c1T', nn.ConvTranspose2d(1, 1, kernel_size=3, stride=8, dilation=2)),
            >>> ]))
            >>> print(ub.repr2(ReceptiveFieldFor(module)()))
            >>> ReceptiveFieldFor(module)()
            >>> OutputShapeFor(module)([1, 1, 900, 900])
            >>> OutputShapeFor(module)([1, 1, 900, 900]).hidden
            >>> OutputShapeFor(module)._check_consistency([1, 1, 900, 900])

            >>> module = nn.Sequential(
            >>>     nn.Conv2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.Conv2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.Conv2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2),
            >>> )
            >>> ReceptiveFieldFor(module)()

            >>> module = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
            >>> ReceptiveFieldFor(module)()

            >>> OutputShapeFor(nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=0, output_padding=(1, 1)))._check_consistency([1, 1, 1, 1])

            >>> # Figure 4.4
            >>> OutputShapeFor(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=2))([1, 1, 5, 5])
            >>> OutputShapeFor(nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=2))._check_consistency([1, 1, 5, 5])
            >>> OutputShapeFor(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0))([1, 1, 7, 7])

            >>> # Figure 4.5
            >>> OutputShapeFor(nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=0))._check_consistency([1, 1, 5, 5])
            >>> OutputShapeFor(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0))([1, 1, 7, 7])

            >>> ReceptiveFieldFor(module)()
        """
        # impl = ReceptiveFieldFor.impl
        if input_field is None:
            input_field = ReceptiveFieldFor.input()

        # Hack to get the number of space-time dimensions
        ndim = None
        try:
            if module.__name__.endswith('1d'):
                ndim = 1
            elif module.__name__.endswith('2d'):
                ndim = 2
            elif module.__name__.endswith('3d'):
                ndim = 3
        except AttributeError:
            pass

        if ndim is None:
            if hasattr(module, '_dim'):
                ndim = module._dim

        # A non-trivial transpose convolution should:
        # * decrease the stride (because the stride is fractional)
        # the padding has to be equal to the shape of the kernel minus one
        """
        From [4]:

        A convolution described by k, s and p has an associated transposed convolution described by:
        * k' = k,
        * s' = 1,
        * p' = k - p - 1,
        * i' = the shape of the stretched input obtained by adding s − 1 zeros
            between each input unit,
        * a = (i + 2p − k) % s, represents the number of zeros added to the
         bottom and right edges of the input,

         And has output shape:
             o' = s(i' - 1) + a + k - 2p

        For convT it is always the case that s'=1, howver, note that s' is not
        what we use to compute the new stride of the output, because that is
        actually a fractional stride.
        """

        # Definitions:
        # In the following comments we discuss 3 distinct layers
        # (1) The original convolution (conv)
        # (2) The transpose convolution that inverts the original (convT)
        # (3) The regular convolution that is equivalent to the transpose
        # convolution given a specially transformed input tensor (convE)

        # The parameters of a convT are actually the parameters of conv, the
        # convolution we are trying to "undo", but we will refer to them as
        # parameters of convT (because they are that as well).
        k_ = ensure_array_nd(module.kernel_size, ndim)
        s_ = ensure_array_nd(module.stride, ndim)
        p_ = ensure_array_nd(module.padding, ndim)
        d_ = ensure_array_nd(getattr(module, 'dilation', 1), ndim)

        # TODO: incorporate output padding
        out_pad = ensure_array_nd(module.output_padding, ndim)
        assert np.all(out_pad == 0), 'cannot handle nonzero yet'

        # Howver, there is an equivalent way of forumulating a convT as convE:
        # a regular conv applied on a specially padded input tensor.
        # The parameters that define convE are:
        k = k_
        d = d_
        s = 1  # stride is always 1 because of the special input transform
        # p = k_ - p_ - 1  # NOTE: original formula likely assumed dilation=1
        p = (k_ - 1) * d_ - p_

        # In order for convE to be equivalent to convT, we need to apply convE
        # to a specially transformed (padded) input tensor.
        # The padding applied to the input tensor puts extra zeros between each
        # row/col. The number of extra zeros is the stride of the convT - 1.
        # The left and right sides of the input tensor are also padded but that
        # wont factor into the RF calculation.
        extra_zeros = s_ - 1
        # This means that the effective support added to the RF shape by convE
        # will be less than it normally would because we don't count the extra
        # zeros in our transformed input as real pixels.
        effective_support = (k - 1 - extra_zeros) * d
        # NOTE; if the stride is larger than the kernel, some output pixels
        # will actually just be zeros and have no receptive feild.
        effective_support = np.maximum(0, effective_support)

        # This special input transform also has the effect of decreasing the RF
        # stride.  Transpose conv are sometimes called fractional-stride
        # convolutions This is because they have an effective stride of 1 / s_
        effective_stride = 1 / s_

        # We calculate the support of convE as if were applied to a normal
        # input tensor in order to calculate how the start (top-left) pixel
        # position is modified.
        support = (k - 1) * d

        # After transformation the effective stride of the input is
        effective_input_stride = input_field['stride'] * effective_stride

        # how many pixels does this layer crop off the sides of the input
        crop = ((support / 2) - p)

        # print('effective_support = {!r}'.format(effective_support))

        field = ReceptiveField({
            # The new stride only depends on the layer stride and the previous
            # stride.
            'stride': effective_input_stride * s,

            # The stride of the current layer does not impact the receptive
            # feild, however the stride of the previous layer does. This is
            # because each pixel in the incoming layer really corresponds
            # `input_field['stride']` pixels in the original input.
            'shape':   input_field['shape'] + effective_support * input_field['stride'],

            # Padding does not influence the RF shape, but it does influence
            # where the start pixel is (i.e. without the right amount of
            # padding the the edge of the previous layer is cropped).
            'crop': input_field['crop'] + crop * effective_input_stride,
        })

        return field
        # raise NotImplementedError('todo')

    @compute_type(nn.modules.conv._ConvTransposeMixin)
    def convT(module, input_field=None):
        return ReceptiveFieldFor._kernelized_tranpose(module, input_field)

    @compute_type(nn.modules.conv.Conv1d, nn.modules.conv.Conv2d, nn.modules.conv.Conv3d)
    def convnd(module, input_field=None):
        return ReceptiveFieldFor._kernelized(module, input_field)

    @staticmethod
    @compute_type(nn.modules.pooling._MaxPoolNd)
    def maxpoolnd(module, input_field=None):
        return ReceptiveFieldFor._kernelized(module, input_field)

    @staticmethod
    @compute_type(nn.modules.pooling._AvgPoolNd)
    def avepoolnd(module, input_field=None):
        return ReceptiveFieldFor._kernelized(module, input_field)

    @staticmethod
    @compute_type(nn.ReLU)
    def relu(module, input_field=None):
        return ReceptiveFieldFor._unchanged(module, input_field)

    @staticmethod
    @compute_type(nn.LeakyReLU)
    def leaky_relu(module, input_field=None):
        return ReceptiveFieldFor._unchanged(module, input_field)

    @staticmethod
    @compute_type(torch.nn.modules.normalization._BatchNorm,
                  torch.nn.modules.normalization.GroupNorm,
                  torch.nn.modules.normalization.LocalResponseNorm,
                  torch.nn.modules.normalization.LayerNorm)
    def normalization(module, input_field=None):
        return ReceptiveFieldFor._unchanged(module, input_field)

    @staticmethod
    @compute_type(nn.modules.dropout._DropoutNd)
    def dropout(module, input_field=None):
        return ReceptiveFieldFor._unchanged(module, input_field)

    @staticmethod
    @compute_type(nn.Sequential)
    def sequential(module, input_field=None):
        """
        Example:
            >>> self = nn.Sequential(
            >>>     nn.Conv2d(2, 3, kernel_size=3),
            >>>     nn.Conv2d(3, 5, kernel_size=3),
            >>>     nn.Conv2d(5, 7, kernel_size=3),
            >>> )
            >>> rfield = ReceptiveFieldFor(self)()
            >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1, with_dtype=False)))
            rfield = {
                'crop': np.array([3., 3.]),
                'shape': np.array([7., 7.]),
                'stride': np.array([1., 1.]),
            }
        """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()
        rfield = input_field
        hidden = HiddenFields()
        for key, child in module._modules.items():
            if hasattr(child, 'receptive_field_for'):
                rfield = hidden[key] = child.receptive_field_for(rfield)
            else:
                rfield = hidden[key] = ReceptiveFieldFor(child)(rfield)
        rfield.hidden = hidden
        return rfield

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
    def resent_basic_block(module, input_field=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> import torchvision  # NOQA
            >>> module = torchvision.models.resnet18().layer1[0]
            >>> field = ReceptiveFieldFor(module)()
            >>> print(ub.repr2(field.hidden, nl=1, with_dtype=False))
            {
                'conv1': {'crop': np.array([0., 0.]), 'shape': np.array([3., 3.]), 'stride': np.array([1., 1.])},
                'bn1': {'crop': np.array([0., 0.]), 'shape': np.array([3., 3.]), 'stride': np.array([1., 1.])},
                'relu1': {'crop': np.array([0., 0.]), 'shape': np.array([3., 3.]), 'stride': np.array([1., 1.])},
                'conv2': {'crop': np.array([0., 0.]), 'shape': np.array([5., 5.]), 'stride': np.array([1., 1.])},
                'bn2': {'crop': np.array([0., 0.]), 'shape': np.array([5., 5.]), 'stride': np.array([1., 1.])},
                'relu2': {'crop': np.array([0., 0.]), 'shape': np.array([5., 5.]), 'stride': np.array([1., 1.])},
            }
        """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()
        hidden = HiddenFields()

        rfield = input_field

        rfield = hidden['conv1'] = ReceptiveFieldFor(module.conv1)(rfield)
        rfield = hidden['bn1'] = ReceptiveFieldFor(module.bn1)(rfield)
        rfield = hidden['relu1'] = ReceptiveFieldFor(module.relu)(rfield)

        rfield = hidden['conv2'] = ReceptiveFieldFor(module.conv2)(rfield)
        rfield = hidden['bn2'] = ReceptiveFieldFor(module.bn2)(rfield)
        rfield = hidden['relu2'] = ReceptiveFieldFor(module.relu)(rfield)

        if module.downsample is not None:
            hidden['downsample'] = ReceptiveFieldFor(module.downsample)(input_field)

        rfield = ReceptiveFieldFor(module.relu)(rfield)
        rfield.hidden = hidden
        return rfield

    @staticmethod
    @compute_type(torchvision.models.resnet.Bottleneck)
    def resent_bottleneck(module, input_field=None):
        """
        CommandLine:
            xdoctest -m netharn.receptive_field_for _TorchvisionMixin.resent_bottleneck --network

        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> import torchvision  # NOQA
            >>> module = torchvision.models.resnet50().layer1[0]
            >>> field = ReceptiveFieldFor(module)()
            >>> print(ub.repr2(field.hidden.shallow(1), nl=1, with_dtype=False))
            {
                'conv1': {'crop': ...([0., 0.]), 'shape': ...([1., 1.]), 'stride': ...([1., 1.])},
                'bn1': {'crop': ...([0., 0.]), 'shape': ...([1., 1.]), 'stride': ...([1., 1.])},
                'relu1': {'crop': ...([0., 0.]), 'shape': ...([1., 1.]), 'stride': ...([1., 1.])},
                'conv2': {'crop': ...([0., 0.]), 'shape': ...([3., 3.]), 'stride': ...([1., 1.])},
                'bn2': {'crop': ...([0., 0.]), 'shape': ...([3., 3.]), 'stride': ...([1., 1.])},
                'relu2': {'crop': ...([0., 0.]), 'shape': ...([3., 3.]), 'stride': ...([1., 1.])},
                'conv3': {'crop': ...([0., 0.]), 'shape': ...([3., 3.]), 'stride': ...([1., 1.])},
                'bn3': {'crop': ...([0., 0.]), 'shape': ...([3., 3.]), 'stride': ...([1., 1.])},
                'downsample': {'crop': ...([0., 0.]), 'shape': ...([1., 1.]), 'stride': ...([1., 1.])},
            }
        """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()
        rfield = input_field
        hidden = HiddenFields()

        rfield = hidden['conv1'] = ReceptiveFieldFor(module.conv1)(rfield)
        rfield = hidden['bn1'] = ReceptiveFieldFor(module.bn1)(rfield)
        rfield = hidden['relu1'] = ReceptiveFieldFor(module.relu)(rfield)

        rfield = hidden['conv2'] = ReceptiveFieldFor(module.conv2)(rfield)
        rfield = hidden['bn2'] = ReceptiveFieldFor(module.bn2)(rfield)
        rfield = hidden['relu2'] = ReceptiveFieldFor(module.relu)(rfield)

        rfield = hidden['conv3'] = ReceptiveFieldFor(module.conv3)(rfield)
        rfield = hidden['bn3'] = ReceptiveFieldFor(module.bn3)(rfield)

        if module.downsample is not None:
            hidden['downsample'] = ReceptiveFieldFor(module.downsample)(input_field)

        rfield = ReceptiveFieldFor(module.relu)(rfield)
        rfield.hidden = hidden
        return rfield

    @staticmethod
    @compute_type(torchvision.models.resnet.ResNet)
    def resnet_model(module, input_field=None, input_shape=None):
        """
        CommandLine:
            xdoctest -m netharn.receptive_field_for _TorchvisionMixin.resnet_model --network

        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> from netharn.receptive_field_for import *
            >>> module = torchvision.models.resnet50()
            >>> input_shape = (1, 3, 224, 224)
            >>> field = ReceptiveFieldFor(module)(input_shape=input_shape)
            >>> print(ub.repr2(field.hidden.shallow(1), nl=1, with_dtype=False))
            {
                'conv1': {'crop': ...([0., 0.]), 'shape': ...([7., 7.]), 'stride': ...([2., 2.])},
                'bn1': {'crop': ...([0., 0.]), 'shape': ...([7., 7.]), 'stride': ...([2., 2.])},
                'relu1': {'crop': ...([0., 0.]), 'shape': ...([7., 7.]), 'stride': ...([2., 2.])},
                'maxpool': {'crop': ...([0., 0.]), 'shape': ...([11., 11.]), 'stride': ...([4., 4.])},
                'layer1': {'crop': ...([0., 0.]), 'shape': ...([35., 35.]), 'stride': ...([4., 4.])},
                'layer2': {'crop': ...([0., 0.]), 'shape': ...([91., 91.]), 'stride': ...([8., 8.])},
                'layer3': {'crop': ...([0., 0.]), 'shape': ...([267., 267.]), 'stride': ...([16., 16.])},
                'layer4': {'crop': ...([0., 0.]), 'shape': ...([427., 427.]), 'stride': ...([32., 32.])},
                'avgpool': {'crop': ...([96., 96.]), 'shape': ...([619., 619.]), 'stride': ...([32., 32.])},
                'flatten': {'crop': ...([96., 96.]), 'shape': ...([811., 811.]), 'stride': ...([32., 32.])},
                'fc': {'crop': ...([96., 96.]), 'shape': ...([811., 811.]), 'stride': ...([32., 32.])},
            }

        """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()
        rfield = input_field
        hidden = HiddenFields()
        rfield = hidden['conv1'] = ReceptiveFieldFor(module.conv1)(rfield)
        rfield = hidden['bn1'] = ReceptiveFieldFor(module.bn1)(rfield)
        rfield = hidden['relu1'] = ReceptiveFieldFor(module.relu)(rfield)
        rfield = hidden['maxpool'] = ReceptiveFieldFor(module.maxpool)(rfield)

        rfield = hidden['layer1'] = ReceptiveFieldFor(module.layer1)(rfield)
        rfield = hidden['layer2'] = ReceptiveFieldFor(module.layer2)(rfield)
        rfield = hidden['layer3'] = ReceptiveFieldFor(module.layer3)(rfield)
        rfield = hidden['layer4'] = ReceptiveFieldFor(module.layer4)(rfield)

        rfield = hidden['avgpool'] = ReceptiveFieldFor(module.avgpool)(rfield)

        if input_shape is None:
            raise ValueError('input shape is required')

        output_shape = OutputShapeFor(module)(input_shape)
        avgpool_shape = output_shape.hidden.shallow(1)['layer4']
        spatial_shape = np.array(avgpool_shape[2:])

        # Keep everything the same except increase the RF shape
        # based on how many output pixels there are.
        rfield_flatten = ReceptiveField(dict(**rfield))
        # not sure if this is 100% correct
        rfield_flatten['shape'] = rfield['shape'] + (spatial_shape - 1) * rfield['stride']
        hidden['flatten'] = rfield = rfield_flatten

        # The reshape operation will blend the receptive fields of the inputs
        # but it will depend on the output shape of the layer.
        # rfield = (rfield[0], prod(rfield[1:]))

        rfield = hidden['fc'] = ReceptiveFieldFor(module.fc)(rfield)
        rfield.hidden = hidden
        return rfield


class ReceptiveFieldFor(_TorchMixin, _TorchvisionMixin):
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
        >>> from netharn.receptive_field_for import *
        >>> self = nn.Sequential(
        >>>     nn.Conv2d(2, 3, kernel_size=3),
        >>>     nn.Conv2d(3, 5, kernel_size=3),
        >>> )
        >>> rfield = ReceptiveFieldFor(self)()
        >>> print('rfield.hidden = {}'.format(ub.repr2(rfield.hidden, nl=3, with_dtype=False)))
        >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1, with_dtype=False)))
        rfield.hidden = {
            '0': {
                'crop': np.array([1., 1.]),
                'shape': np.array([3., 3.]),
                'stride': np.array([1., 1.]),
            },
            '1': {
                'crop': np.array([2., 2.]),
                'shape': np.array([5., 5.]),
                'stride': np.array([1., 1.]),
            },
        }
        rfield = {
            'crop': np.array([2., 2.]),
            'shape': np.array([5., 5.]),
            'stride': np.array([1., 1.]),
        }

    Example:
        >>> # Case where we haven't registered a func
        >>> self = nn.Conv2d(2, 3, kernel_size=3)
        >>> rfield = ReceptiveFieldFor(self)()
        >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1, with_dtype=False)))
        rfield = {
            'crop': np.array([1., 1.]),
            'shape': np.array([3., 3.]),
            'stride': np.array([1., 1.]),
        }

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import torchvision  # NOQA
        >>> module = torchvision.models.alexnet().features
        >>> field = ReceptiveFieldFor(module)()
        >>> print(ub.repr2(field, nl=1, with_dtype=False))
        {
            'crop': np.array([31., 31.]),
            'shape': np.array([195., 195.]),
            'stride': np.array([32., 32.]),
        }
    """
    # impl = math  # for hacking in sympy

    def __init__(self, module):
        self.module = module
        self._func = getattr(module, 'receptive_field_for', None)
        if self._func is None:
            # Lookup rfield func if we can't find it
            found = []
            for type, _func in REGISTERED_TYPES:
                try:
                    if module is type or isinstance(module, type):
                        found.append(_func)
                except TypeError:
                    pass
            if len(found) == 1:
                self._func = found[0]
            elif len(found) == 0:
                raise TypeError('Unknown (rf) module type {}'.format(module))
            else:
                raise AssertionError('Ambiguous (rf) module {}. Found {}'.format(module, found))

    def __call__(self, *args, **kwargs):
        if isinstance(self.module, nn.Module):
            # bound methods dont need module
            is_bound  = hasattr(self._func, '__func__') and getattr(self._func, '__func__', None) is not None
            is_bound |= hasattr(self._func, 'im_func') and getattr(self._func, 'im_func', None) is not None
            if is_bound:
                rfield = self._func(*args, **kwargs)
            else:
                # nn.Module with state
                rfield = self._func(self.module, *args, **kwargs)
        else:
            # a simple pytorch func
            rfield = self._func(*args, **kwargs)
        return rfield


def effective_receptive_feild(module, inputs, output_key=None, sigma=0,
                              thresh=1.00, ignore_norms=True,
                              ignore_extra=None):
    """
    Empirically measures the effective receptive feild of a network

    Method from [0], implementation loosely based on [1].

    Args:
        module (torch.nn.Module) : the network

        inputs (torch.nn.Tensor) : the input to the network. Must share the
            same device as `module`.

        output_key (None | str | Callable): If the network outputs a non-tensor
            then this should be a function that does postprocessing and returns
            a relevant Tensor that can be used to compute gradients. If the
            output is a dictionary then this can also be a string-based key
            used to lookup the appropriate output.

        sigma (float, default=0): smoothness factor (via gaussian blur)

        thresh (float, default=1.00): only consider this fraction of the
            data as meaningful (i.e. find the effective RF shape that explains
            95% of the data). A threshold of 1.0 or greater does nothing.

        ignore_norms (bool, default=True): if True ignores normalization layers
            like batch and group norm which adds negligable, but non-zero
            impact everywhere and causes the ERF shape estimation to be
            dramatically greater than it should be (although the impact still
            makes sense).

        ignore_extra (List[type], optioanl): if specified, any layer that is a
            subclass of one of these types is also ignored.

    Returns:
        dict: containing keys
            'shape' containing the effective RF shape and
            'impact' which contains the thresholded distribution

    References:
        [0] https://arxiv.org/pdf/1701.04128.pdf
        [1] https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/compute_RF.py

    Example:
        >>> from netharn.receptive_field_for import *
        >>> import torchvision  # NOQA
        >>> module = nn.Sequential(*[nn.Conv2d(1, 1, 3) for i in range(10)])
        >>> inputs = torch.rand(1, 1, 200, 200)
        >>> emperical_field = effective_receptive_feild(module, inputs)
        >>> theoretic_field = ReceptiveFieldFor(module)()
        >>> # The emperical results should never be bigger than the theoretical
        >>> assert np.all(emperical_field['shape'] <= theoretic_field['shape'])

        >>> # xdoctest: +REQUIRES(--slow)
        >>> module = torchvision.models.alexnet().features
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> emperical_field = effective_receptive_feild(module, inputs)
        >>> theoretic_field = ReceptiveFieldFor(module)()
        >>> # The emperical results should never be bigger than the theoretical
        >>> assert np.all(emperical_field['shape'] <= theoretic_field['shape'])

        >>> # xdoctest: +REQUIRES(--slow)
        >>> import netharn as nh
        >>> xpu = nh.XPU.cast('auto')
        >>> module = xpu.move(torchvision.models.vgg11_bn().features)
        >>> inputs = xpu.move(torch.rand(1, 3, 224, 224))
        >>> emperical_field = effective_receptive_feild(module, inputs)
        >>> theoretic_field = ReceptiveFieldFor(module)()
        >>> # The emperical results should never be bigger than the theoretical
        >>> assert np.all(emperical_field['shape'] <= theoretic_field['shape'])

        >>> # xdoctest: +REQUIRES(--show)
        >>> nh.util.autompl()
        >>> nh.util.imshow(emperical_field['impact'], doclf=True)

    Ignore:
        >>> xpu = nh.XPU.cast('auto')
        >>> module = xpu.move(torchvision.models.resnet50())
        >>> inputs = xpu.move(torch.rand(8, 3, 224, 224))
        >>> emperical_field = effective_receptive_feild(module, inputs)
        >>> nh.util.autompl()
        >>> nh.util.imshow(emperical_field['impact'], doclf=True)
    """
    import netharn as nh

    # zero gradients
    for p in module.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    if inputs.grad is not None:
        inputs.grad.detach_()
        inputs.grad.zero_()

    inputs.requires_grad = True
    # if inputs.grad is not None:
    #     raise ValueError('inputs alread has accumulated gradients')

    # Completely ignore BatchNorm layers as they will give the entire input
    # some negligable but non-zero effect on the receptive feild.
    ignored = []
    if ignore_norms:
        ignored += [
            torch.nn.modules.normalization._BatchNorm,
            torch.nn.modules.normalization.GroupNorm,
            torch.nn.modules.normalization.LocalResponseNorm,
            torch.nn.modules.normalization.LayerNorm,
            nh.layers.L2Norm,
        ]
    if ignore_extra:
        ignored += ignore_extra
    with nh.util.IgnoreLayerContext(module, tuple(ignored)):
        outputs = module(inputs)

    # Note: grab a single (likely FCN) output channel
    if callable(output_key):
        output_y = output_key(outputs)
    elif output_key is None:
        output_y = outputs
    else:
        output_y = outputs[output_key]
    # elif isinstance(output_key, (six.string_types, int)):
    # else:
    #     raise TypeError('output_key={} is not understood'.format(output_key))

    if not isinstance(output_y, torch.Tensor):
        raise TypeError(
            'The output is a {}, not a tensor. Please specify '
            'output_key and ensure it returns a Tensor.'.format(type(outputs)))

    # Note: this still does the right thing if there is no spatial component.
    # because all outputs are center outputs.
    center_dims = (np.array(output_y.shape[2:]) // 2).tolist()
    center_slice = [slice(None), slice(None)] + center_dims

    # We dont need to compute a loss because we can explicitly set gradients.
    # Yay torch!
    # Explicilty set ∂l/∂y[:] = 0
    # Explicilty set ∂l/∂y[center] = 1
    grad_loss_wrt_y = torch.zeros_like(output_y)
    grad_loss_wrt_y[...] = 0
    grad_loss_wrt_y[center_slice] = 1

    # Backpropogate as if the grad of the loss wrt to y[center] was 1.
    output_y.backward(gradient=grad_loss_wrt_y)

    # The input gradient is now a measure of how much it can impact the output.
    impact = inputs.grad.abs()

    # Average the impact over all batches and all channels
    average_impact = impact.mean(dim=0).mean(dim=0)

    if isinstance(average_impact, torch.Tensor):
        average_impact = average_impact.data.cpu().numpy()

    idx_nonzeros = np.where(average_impact != 0)
    rf_bounds = [(0, 0) if len(idx) == 0 else (idx.min(), idx.max()) for idx in idx_nonzeros]
    rf_shape = [(mx - mn + 1) for mn, mx in rf_bounds]
    rf_slice = tuple([slice(mn, mx + 1) for mn, mx in rf_bounds])

    # Crop out the average impact zone for visualization
    # Normalize to have a maximum value of 1.0
    rf_impact = average_impact[rf_slice]
    rf_impact /= rf_impact.max()

    rf_impact = torch.FloatTensor(rf_impact)
    if sigma > 0:
        # Smooth things out
        _blur = nh.layers.GaussianBlurNd(dim=1, num_features=1, sigma=sigma)
        _blur.to(rf_impact.device)
        rf_impact = _blur(rf_impact[None, None])[0, 0]

    if thresh < 1:
        density = rf_impact.contiguous().view(-1).cpu().numpy().copy()
        density.sort()
        density = density[::-1]
        # Find the value threshold that explains thresh (e.g. 95%) of the data
        idx = np.where(density.cumsum() > thresh * density.sum())[0]
        lowval = float(density[idx[0]])

        effective_impact = rf_impact * (rf_impact > lowval).float()
        effective_idx_nonzeros = np.where(effective_impact != 0)
        effective_rf_bounds = [(idx.min(), idx.max()) for idx in effective_idx_nonzeros]
        effective_shape = [(mx - mn + 1) for mn, mx in effective_rf_bounds]
    else:
        effective_impact = rf_impact
        effective_rf_bounds = rf_shape
        effective_shape = rf_shape

    emperical_field = {
        'shape': effective_shape,
        'impact': effective_impact,
        'thresh': thresh,
    }
    return emperical_field


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.receptive_field_for all --network
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
