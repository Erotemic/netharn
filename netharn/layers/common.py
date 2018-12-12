from netharn import util
from torch import nn


# TODO: move module mixin from util to here
ModuleMixin = util.ModuleMixin


class Module(nn.Module, util.ModuleMixin):
    """
    Like torch.nn.Module but contains ModuleMixin (for number_of_parameters)
    """


class Sequential(nn.Sequential, util.ModuleMixin):
    """
    Like torch.nn.Sequential but implements output_shape_for / hidden_shapes_for

    Example:
        >>> import ubelt as ub
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

    def receptive_field_for(self, input_field=None):
        from netharn.receptive_field_for import ReceptiveFieldFor
        return ReceptiveFieldFor.sequential(self, input_field)


class Identity(Sequential):
    """
    A identity-function layer.

    Example:
        >>> import torch
        >>> self = Identity()
        >>> a = torch.rand(3, 3)
        >>> b = self(a)
        >>> assert torch.all(a == b)
    """
    def __init__(self):
        super(Identity, self).__init__()
