from torch import nn
from netharn.output_shape_for import OutputShapeFor  # NOQA
from netharn.output_shape_for import SHAPE_CLS


class Reshape(nn.Module):
    """
    Wrapper class around `torch.view` that implements `output_shape_for`

    Args:
        *shape: same ars that would be passed to view.
            if an item in shape is None it means that the output
            shape should keep the input shap value in that dimension


    Example:
        >>> OutputShapeFor(Reshape(-1, 3))._check_consistency((20, 6, 20))
        (800, 3)
        >>> OutputShapeFor(Reshape(100, -1, 5))._check_consistency((10, 10, 15))
        (100, 3, 5)
        >>> Reshape(7, -1, 3).output_shape_for((None, 1))  # broken?
        (7, None, 3)
        >>> OutputShapeFor(Reshape(None, -1, 4))._check_consistency((10, 32, 32, 16))

    Ignore:
        >>> self = Reshape(None, -1, 4)
        >>> input_shape = (10, 32, 32, 16)
    """
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
        if len(shape) == 0:
            raise ValueError('Reshape dims cannot be empty')

        self._none_dims = []
        self._neg_dims = []
        for i, d in enumerate(shape):
            if d is None:
                self._none_dims.append(i)
            elif d < 0:
                self._neg_dims.append(i)
        if len(self._neg_dims) > 1:
            raise ValueError('Can only specify one negative dimension')

    def forward(self, input):
        if not self._none_dims:
            output_shape = self.shape
        else:
            output_shape = list(self.shape)
            input_shape = input.shape
            for i in self._none_dims:
                if i >= len(input_shape):
                    raise ValueError('input shape does not correspond')
                output_shape[i] = input_shape[i]
        return input.view(*output_shape)

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

        # If any dim in output_shape is set to None, it should use whatever the
        # corresonding value in input shape is. This feature is not part of
        # standard torch.view
        output_shape = list(self.shape)
        for i in self._none_dims:
            if i >= len(input_shape):
                raise ValueError('input shape does not correspond')
            output_shape[i] = input_shape[i]

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
        for j, s in enumerate(output_shape):
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
