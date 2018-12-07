# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torchvision
import ubelt as ub
import numpy as np
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
#     def __init__(self, stride, size, crop):
#         self.data = {
#             'stride': stride,  # The stride / scale factor of the network
#             'size': size,  # The receptive feild size of a single output pixel
#             'crop': crop,  # The amount of cropping / starting pixel location
#         }

#     def __nice__(self):
#         return ub.repr2(self.data, nl=1)

#     def __getitem__(self, key):
#         return self.data[key]


ReceptiveField = dict


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
        input_field = ReceptiveField(**{
            # The input receptive field stride / scale factor is 1.
            'stride': ensure_array_nd(1, n),
            # The input receptive field size is 1 pixel.
            'size': ensure_array_nd(1, n),
            # Use the coordinate system where the top left corner is 0, 0 ( This is unlike [1], which uses 0.5)
            'crop': ensure_array_nd(0.0, n),
        })
        return input_field, input_field

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
            [3] https://stackoverflow.com/questions/35582521/how-to-calculate-receptive-field-size
            [4] https://arxiv.org/pdf/1603.07285.pdf

        Example:
            >>> module = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=2, dilation=3)
            >>> ReceptiveFieldFor._kernelized(module)[1]

            >>> module = nn.MaxPool2d(kernel_size=3, stride=2, padding=2, dilation=2)
            >>> module = nn.MaxPool2d(kernel_size=3, stride=2, padding=2, dilation=1)
            >>> ReceptiveFieldFor(module)()[1]
        """
        # impl = ReceptiveFieldFor.impl
        if input_field is None:
            input_field = ReceptiveFieldFor.input()[1]

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
        d = ensure_array_nd(module.dilation, ndim)

        # To calculate receptive feild we first need to find the SUPPORT of
        # this layer. The support is the number/extent of extra surrounding
        # pixels adding this layer will take into account. Given this, we can
        # compute the receptive feild wrt the original input by combining this
        # information with the previous receptive feild.
        #
        # In the normal case (with no dilation, d=1) the support is (k - 1).
        # This is because because the operation is able to see a window of size
        # k in the input, and produces a single output pixel (hence the k). The
        # center input pixel corresponds with the output, so it does not expand
        # the receptive feild (hence the -1), but all other input pixels do
        # expand the field (thus the k-1).
        #
        # The stride of this layer will not affect the support.
        #
        # The dilation of the current layer DOES impact the support.
        # This expands the effective kernel size, but it does cause the data
        # each operation sees to become more diffuse. However, even though what
        # it sees in that extent is more diffuse, the RF is just a bound, so we
        # can ignore the diffuseness effect and simply scale the input kernel
        # size by the dilation amount. Hense we get
        support = (k - 1) * d

        """
        Note the above is correct because:

            import sympy as sym
            k, d = sym.symbols('k, d')

            # Compute the support from formula in 5.1 of [4]
            # To understand the relationship tying the dilation rate d and the
            # output size o, it is useful to think of the impact of d on the
            # effective kernel size. A kernel of size k dilated by a factor d
            # has an effective size.
            effective_kernel_size = k + (k - 1) * (d - 1)
            support_v1 = sym.expand(effective_kernel_size - 1)

            # Compute support from our method
            support_v2 = sym.expand((k - 1) * d)

            # They are equivalent. QED ☐
            assert sym.Eq(support_v1, support_v2)
        """

        # Compute how many pixels this layer takes off the side
        crop = ((support / 2) - p)

        field = ReceptiveField(**{
            # The new stride only depends on the layer stride and the previous
            # stride.
            'stride': input_field['stride'] * s,

            # The stride of the current layer does not impact the receptive
            # feild, however the stride of the previous layer does. This is
            # because each pixel in the incoming layer really corresponds
            # `input_field['stride']` pixels in the original input.
            'size':   input_field['size'] + support * input_field['stride'],

            # Padding does not influence the RF size, but it does influence
            # where the start pixel is (i.e. without the right amount of
            # padding the the edge of the previous layer is cropped).
            'crop': input_field['crop'] + crop * input_field['stride'],
        })
        return field, field

    @staticmethod
    def _unchanged(module, input_field=None):
        """ Formula for layers that do not change the receptive field """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()[1]
        return input_field, input_field

    @staticmethod
    @compute_type(nn.Linear)
    def linear(module, input_field=None):
        # Linear layers (sort-of) dont change the RF
        return ReceptiveFieldFor._unchanged(module, input_field)
        # Perhaps we could do this if we knew the input shape
        # raise NotImplementedError(
        #     'Cannot compute receptive field size on a Linear layer')

    @compute_type(nn.modules.conv._ConvTransposeMixin)
    def convT(module, input_field=None):
        """
        Receptive field formula for pooling layers

        Example:
            >>> from netharn.receptive_field_for import *
            >>> from netharn.output_shape_for import *
            >>> from netharn.hidden_shapes_for import *
            >>> module = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=2)
            >>> ReceptiveFieldFor(module)()[1]

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
            >>> print(ub.repr2(ReceptiveFieldFor(module)()[0]))
            >>> ReceptiveFieldFor(module)()[1]
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
            >>> print(ub.repr2(ReceptiveFieldFor(module)()[0]))

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
            >>> print(ub.repr2(ReceptiveFieldFor(module)()[0]))
            >>> ReceptiveFieldFor(module)()[1]
            >>> OutputShapeFor(module)([1, 1, 900, 900])
            >>> HiddenShapesFor(module)([1, 1, 900, 900])
            >>> OutputShapeFor(module)._check_consistency([1, 1, 900, 900])

            >>> module = nn.Sequential(
            >>>     nn.Conv2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.Conv2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.Conv2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2),
            >>>     nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2),
            >>> )
            >>> ReceptiveFieldFor(module)()[1]

            >>> module = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
            >>> ReceptiveFieldFor(module)()[1]

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
            input_field = ReceptiveFieldFor.input()[1]

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

        # A non-trivial transpose convolution should:
        # * decrease the stride (because the stride is fractional)
        # the padding has to be equal to the size of the kernel minus one
        """
        From [4]:

        A convolution described by k, s and p has an associated transposed convolution described by:
        * k' = k,
        * s' = 1,
        * p' = k - p - 1,
        * i' = the size of the stretched input obtained by adding s − 1 zeros
            between each input unit,
        * a = (i + 2p − k) % s, represents the number of zeros added to the
         bottom and right edges of the input,

         And has output size:
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
        d_ = ensure_array_nd(module.dilation, ndim)

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
        # This means that the effective support added to the RF size by convE
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

        field = ReceptiveField(**{
            # The new stride only depends on the layer stride and the previous
            # stride.
            'stride': effective_input_stride * s,

            # The stride of the current layer does not impact the receptive
            # feild, however the stride of the previous layer does. This is
            # because each pixel in the incoming layer really corresponds
            # `input_field['stride']` pixels in the original input.
            'size':   input_field['size'] + effective_support * input_field['stride'],

            # Padding does not influence the RF size, but it does influence
            # where the start pixel is (i.e. without the right amount of
            # padding the the edge of the previous layer is cropped).
            'crop': input_field['crop'] + crop * effective_input_stride,
        })

        return field, field
        # raise NotImplementedError('todo')

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
    @compute_type(nn.modules.batchnorm._BatchNorm)
    def batchnorm(module, input_field=None):
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
            >>> rfields, rfield = ReceptiveFieldFor(self)()
            >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1, with_dtype=False)))
            rfield = {
                'crop': np.array([3., 3.]),
                'size': np.array([7, 7]),
                'stride': np.array([1, 1]),
            }
        """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()[1]
        rfield = input_field
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
    def resent_basic_block(module, input_field=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> import torchvision  # NOQA
            >>> module = torchvision.models.resnet18().layer1[0]
            >>> fields, field = ReceptiveFieldFor(module)()
            >>> print(ub.repr2(fields, nl=2, with_dtype=False))
        """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()[1]
        rfields = ub.odict()

        residual_field = input_field
        rfield = input_field

        rfields['conv1'], rfield = ReceptiveFieldFor(module.conv1)(rfield)
        rfields['bn1'], rfield = ReceptiveFieldFor(module.bn1)(rfield)
        rfields['relu1'], rfield = ReceptiveFieldFor(module.relu)(rfield)

        rfields['conv2'], rfield = ReceptiveFieldFor(module.conv2)(rfield)
        rfields['bn2'], rfield = ReceptiveFieldFor(module.bn2)(rfield)
        rfields['relu2'], rfield = ReceptiveFieldFor(module.relu)(rfield)

        if module.downsample is not None:
            rfields['downsample'], residual_field = ReceptiveFieldFor(module.downsample)(input_field)

        rfield = ReceptiveFieldFor(module.relu)(rfield)
        return rfields, rfield

    @staticmethod
    @compute_type(torchvision.models.resnet.Bottleneck)
    def resent_bottleneck(module, input_field=None):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> import torchvision  # NOQA
            >>> module = torchvision.models.resnet50().layer1[0]
            >>> fields, field = ReceptiveFieldFor(module)()
            >>> print(ub.repr2(fields[-1], nl=1, with_dtype=False))
        """
        if input_field is None:
            input_field = ReceptiveFieldFor.input()[1]
        residual_field = input_field
        rfield = input_field

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
            rfields['downsample'], residual_field = ReceptiveFieldFor(module.downsample)(input_field)

        rfield = ReceptiveFieldFor(module.relu)(rfield)
        return rfield

    @staticmethod
    @compute_type(torchvision.models.resnet.ResNet)
    def resnet_model(module, input_field=None, input_shape=None):
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
        if input_field is None:
            input_field = ReceptiveFieldFor.input()[1]
        rfield = input_field
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
        >>> self = nn.Sequential(
        >>>     nn.Conv2d(2, 3, kernel_size=3),
        >>>     nn.Conv2d(3, 5, kernel_size=3),
        >>> )
        >>> rfields, rfield = ReceptiveFieldFor(self)()
        >>> print('rfields = {}'.format(ub.repr2(rfields, nl=3)))
        >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1)))
        rfields = {
            '0': {
                'crop': np.array([1., 1.], dtype=np.float64),
                'size': np.array([3, 3], dtype=np.int64),
                'stride': np.array([1, 1], dtype=np.int64),
            },
            '1': {
                'crop': np.array([2., 2.], dtype=np.float64),
                'size': np.array([5, 5], dtype=np.int64),
                'stride': np.array([1, 1], dtype=np.int64),
            },
        }
        rfield = {
            'crop': np.array([2., 2.], dtype=np.float64),
            'size': np.array([5, 5], dtype=np.int64),
            'stride': np.array([1, 1], dtype=np.int64),
        }

    Example:
        >>> # Case where we haven't registered a func
        >>> # In this case rfields is not populated (but rfield is)
        >>> self = nn.Conv2d(2, 3, kernel_size=3)
        >>> rfields, rfield = ReceptiveFieldFor(self)()
        >>> print('rfield = {}'.format(ub.repr2(rfield, nl=1)))
        rfield = {
            'crop': np.array([1., 1.], dtype=np.float64),
            'size': np.array([3, 3], dtype=np.int64),
            'stride': np.array([1, 1], dtype=np.int64),
        }

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import torchvision  # NOQA
        >>> module = torchvision.models.alexnet().features
        >>> fields, field = ReceptiveFieldFor(module)()
        >>> print(ub.repr2(fields[-1], nl=1, with_dtype=False))
        {
            'crop': np.array([31., 31.]),
            'size': np.array([195, 195]),
            'stride': np.array([32, 32]),
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
