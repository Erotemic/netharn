from torch import nn
from netharn.output_shape_for import OutputShapeFor  # NOQA
from netharn.output_shape_for import SHAPE_CLS


class Reshape(nn.Module):
    """
    Wrapper class around `torch.view` that implements `output_shape_for`

    Args:
        *shape: same ars that would be passed to view

    Example:
        >>> OutputShapeFor(Reshape(-1, 3))._check_consistency((20, 6, 20))
        (800, 3)
        >>> OutputShapeFor(Reshape(100, -1, 5))._check_consistency((10, 10, 15))
        (100, 3, 5)
        >>> Reshape(7, -1, 3).output_shape_for((None, 1))  # broken?
        (7, None, 3)
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
        if len(shape) == 0:
            raise ValueError('Reshape dims cannot be empty')
        if sum(d < 0 for d in shape) > 1:
            raise ValueError('Can only specify one negative dimension')

    def forward(self, input):
        return input.view(self.shape)

    def extra_repr(self):
        """
        Example:
            >>> print(Reshape(-1, 10))
            Reshape(-1, 10)
            >>> print(Reshape(5, 5, 5))
            Reshape(5, 5, 5)
        """
        return '{}'.format(', '.join(str(s) for s in self.shape))

    def output_shape_for(self, input_shape):
        # Not sure if this works in all cases
        # TODO: find a cleaner (and correct) implementation

        if len(input_shape) == 0:
            raise ValueError('input shape cannot be empty')

        # Check how many total numbers are in the input / if the input
        # has an unspecified batch dimension.
        input_has_none = input_shape[0] is None
        input_total = 1 if input_has_none else input_shape[0]

        for i, d in enumerate(input_shape[1:], start=1):
            if d is None:
                raise ValueError(
                    'Invalid input shape: input_shape[{}] = None, '
                    'but only the first item can be None'.format(i))
            input_total *= d

        # Check the total numbers that the output shape wants
        neg_dims = []
        unused = input_total
        output_shape = list(self.shape)
        for j, s in enumerate(self.shape):
            if s == -1:
                neg_dims.append(j)
            else:
                if not input_has_none:
                    if s > input_total or input_total % s != 0:
                        raise ValueError('does not fit')
                    unused = unused // s

        if neg_dims:
            if len(neg_dims) > 1:
                raise ValueError('Can only specify -1 in reshape dim once')
            j = neg_dims[0]
            if input_has_none:
                output_shape[j] = None
            else:
                output_shape[j] = unused
        else:
            if not input_has_none:
                assert unused == 1

        return SHAPE_CLS(output_shape)
