from netharn.layers import common
from netharn.layers import rectify
from netharn.layers import conv_norm


class MultiLayerPerceptronNd(common.Sequential):
    """
    Args:
        in_channels (int):
        hidden_channels (List[int]):
        out_channels (int):

    CommandLine:
        xdoctest -m netharn.layers.perceptron MultiLayerPerceptronNd

    Example:
        >>> self = MultiLayerPerceptronNd(1, 128, [256, 64], out_channels=2)
        >>> print(self)
        MultiLayerPerceptronNd...
        >>> import ubelt as ub
        >>> shapes, shape = self.hidden_shapes_for([1, 128])
        >>> print(ub.repr2(shapes, nl=1))
        {
            'hidden0': (1, 256),
            'dropout0': (1, 256),
            'hidden1': (1, 64),
            'dropout1': (1, 64),
            'output': (1, 2),
        }

    """
    def __init__(self, dim, in_channels, hidden_channels, out_channels,
                 bias=True, dropout=0, noli='relu', norm='batch'):
        super().__init__()
        dropout_cls = rectify.rectify_dropout(dim)
        conv_cls = rectify.rectify_conv(dim=dim)
        curr_in = in_channels
        for i, curr_out in enumerate(hidden_channels):
            layer = conv_norm.ConvNormNd(dim, curr_in, curr_out, kernel_size=1,
                                         bias=False, noli=noli, norm=norm)
            self.add_module('hidden{}'.format(i), layer)
            self.add_module('dropout{}'.format(i), dropout_cls(p=dropout))
            curr_in = curr_out
        layer = conv_cls(curr_in, out_channels, kernel_size=1, bias=bias)
        self.add_module('output', layer)
