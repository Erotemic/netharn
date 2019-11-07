from netharn.layers import common
from netharn.layers import rectify
from netharn.layers import conv_norm


class MultiLayerPerceptronNd(common.Module):
    """
    Args:
        in_channels (int):
        hidden_channels (List[int]):
        out_channels (int):

    CommandLine:
        xdoctest -m netharn.layers.perceptron MultiLayerPerceptronNd

    Example:
        >>> from netharn.layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(1, 128, [256, 64], out_channels=2)
        >>> print(self)
        MultiLayerPerceptronNd...
        >>> shape = self.output_shape_for([1, 128, 7])
        >>> print('shape = {!r}'.format(shape))
        >>> print('shape.hidden = {}'.format(ub.repr2(shape.hidden, nl=1)))
        shape = (1, 2, 7)
        shape.hidden = {
            'hidden0': {'conv': (1, 256, 7), 'noli': (1, 256, 7)},
            'dropout0': (1, 256, 7),
            'hidden1': {'conv': (1, 64, 7), 'noli': (1, 64, 7)},
            'dropout1': (1, 64, 7),
            'output': (1, 2, 7),
        }
        >>> import netharn as nh
        >>> nh.OutputShapeFor(self)._check_consistency([1, 128, 7])
        (1, 2, 7)

    Example:
        >>> from netharn.layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(0, 128, [256, 64], residual=True,
        >>>                               norm='group', out_channels=2)
        >>> print(self)
        >>> input_shape = (None, 128)
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden, nl=-1))
        {
            'hidden0': {
                'conv': (None, 256),
                'norm': (None, 256),
                'noli': (None, 256)
            },
            'dropout0': (None, 256),
            'hidden1': {
                'conv': (None, 64),
                'norm': (None, 64),
                'noli': (None, 64)
            },
            'dropout1': (None, 64),
            'output': (None, 2),
            'skip': (None, 2)
        }

    Example:
        >>> from netharn.layers.perceptron import *
        >>> import ubelt as ub
        >>> self = MultiLayerPerceptronNd(0, 128, [], residual=False,
        >>>                               norm='group', out_channels=2)
        >>> print(self)
        >>> input_shape = (None, 128)
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden, nl=-1))

    Ignore:
        >>> from netharn.layers.perceptron import *  # NOQA
        >>> closer = nh.export.closer.Closer()
        >>> closer.add_dynamic(MultiLayerPerceptronNd)
        >>> print(closer.current_sourcecode())
    """
    def __init__(self, dim, in_channels, hidden_channels, out_channels,
                 bias=True, dropout=0, noli='relu', norm=None, residual=False):
        super(MultiLayerPerceptronNd, self).__init__()
        dropout_cls = rectify.rectify_dropout(dim)
        conv_cls = rectify.rectify_conv(dim=dim)
        curr_in = in_channels
        if hidden_channels is None:
            hidden_channels = []

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
