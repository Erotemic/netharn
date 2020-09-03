from netharn.layers import common
from netharn.layers import rectify
from netharn.layers import conv_norm
import numpy as np


class MultiLayerPerceptronNd(common.Module):
    """
    A multi-layer perceptron network for n dimensional data

    Choose the number and size of the hidden layers, number of output channels,
    wheather to user residual connections or not, nonlinearity, normalization,
    dropout, and more.

    Args:
        dim (int): specify if the data is 0, 1, 2, 3, or 4 dimensional.
        in_channels (int):
        hidden_channels (List[int]): or an int specifying the number of hidden
            layers (we choose the channel size to linearly interpolate between
            input and output channels)
        out_channels (int):
        dropout (float, default=0): amount of dropout to use
        norm (str, default='batch'): type of normalization layer
            (e.g. batch or group), set to None for no normalization.
        noli (str, default='relu'): type of nonlinearity
        residual (bool, default=False):
            if true includes a resitual skip connection between inputs and
            outputs.

    CommandLine:
        xdoctest -m ~/code/netharn/netharn/layers/perceptron.py MultiLayerPerceptronNd:0

    Example:
        >>> from netharn.layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(dim=1, in_channels=128, hidden_channels=3, out_channels=2)
        >>> print(self)
        MultiLayerPerceptronNd...
        >>> shape = self.output_shape_for([1, 128, 7])
        >>> print('shape = {!r}'.format(shape))
        >>> print('shape.hidden = {}'.format(ub.repr2(shape.hidden, nl=1)))
        shape = (1, 2, 7)
        shape.hidden = {
            'hidden0': {'conv': (1, 96, 7), 'norm': (1, 96, 7), 'noli': (1, 96, 7)},
            'dropout0': (1, 96, 7),
            'hidden1': {'conv': (1, 65, 7), 'norm': (1, 65, 7), 'noli': (1, 65, 7)},
            'dropout1': (1, 65, 7),
            'hidden2': {'conv': (1, 34, 7), 'norm': (1, 34, 7), 'noli': (1, 34, 7)},
            'dropout2': (1, 34, 7),
            'output': (1, 2, 7),
        }
        >>> import netharn as nh
        >>> nh.OutputShapeFor(self)._check_consistency([1, 128, 7])
        (1, 2, 7)
        >>> print('self._hidden_channels = {!r}'.format(self._hidden_channels))

    Example:
        >>> from netharn.layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(0, 128, [256, 64], residual=True,
        >>>                               norm='group', out_channels=2)
        >>> print(self)
        >>> input_shape = (None, 128)
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden, nl=-1))

    Example:
        >>> from netharn.layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(0, 128, [], residual=False,
        >>>                               norm='group', out_channels=2)
        >>> print(self)
        >>> input_shape = (None, 128)
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden, nl=-1))
    """
    def __init__(self, dim, in_channels, hidden_channels, out_channels,
                 bias=True, dropout=0, noli='relu', norm='batch',
                 residual=False):

        super(MultiLayerPerceptronNd, self).__init__()
        dropout_cls = rectify.rectify_dropout(dim)
        conv_cls = rectify.rectify_conv(dim=dim)
        curr_in = in_channels

        if isinstance(hidden_channels, int):
            n = hidden_channels
            hidden_channels = np.linspace(in_channels, out_channels, n + 1,
                                          endpoint=False)[1:]
            hidden_channels = hidden_channels.round().astype(np.int).tolist()
        self._hidden_channels = hidden_channels

        hidden = self.hidden = common.Sequential()
        for i, curr_out in enumerate(hidden_channels):
            layer = conv_norm.ConvNormNd(dim, curr_in, curr_out, kernel_size=1,
                                         bias=False, noli=noli, norm=norm)
            hidden.add_module('hidden{}'.format(i), layer)
            hidden.add_module('dropout{}'.format(i), dropout_cls(p=dropout))
            curr_in = curr_out

        outkw = {'bias': bias}
        if dim > 0:
            outkw['kernel_size'] = 1
        self.hidden.add_module('output', conv_cls(curr_in, out_channels, **outkw))

        if residual:
            self.skip = conv_cls(in_channels, out_channels, **outkw)
        else:
            self.skip = None

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inputs):
        outputs = self.hidden(inputs)
        if self.skip:
            outputs = self.skip(inputs) + outputs
        return outputs

    def output_shape_for(self, input_shape):
        outputs = self.hidden.output_shape_for(input_shape)
        if self.skip:
            import netharn as nh
            skip = nh.OutputShapeFor(self.skip)(input_shape)
            outputs.hidden['skip'] = skip
        return outputs

    def receptive_field_for(self, input_field=None):
        import netharn as nh
        field = nh.ReceptiveFieldFor(self.hidden)(input_field)
        if self.skip:
            skip = nh.ReceptiveFieldFor(self.skip)(field)
            field.hidden['skip'] = skip
        return field
