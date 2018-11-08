import torch.nn.functional as F
from torch import nn
import torch


class L2Norm(nn.Module):
    """
    L2Norm layer across all channels

    Notes:
        "The L2 normalization technique introduced in [12] to scale the feature
        norm at each location in the feature map to `scale` and learn the scale
        during back propagation."

    References:
        [12] Liu, Rabinovich, Berg - ParseNet: Looking wider to see better (ILCR) (2016)

    Example:
        >>> import numpy as np
        >>> import ubelt as ub
        >>> in_features = 7
        >>> self = L2Norm(in_features, scale=20)
        >>> x = torch.rand(1, in_features, 2, 2)
        >>> y = self(x)
        >>> norm = np.linalg.norm(y.data.cpu().numpy(), axis=1)
        >>> print(ub.repr2(norm, precision=2))
        np.array([[[20., 20.],
                   [20., 20.]]], dtype=np.float32)

    Example:
        >>> from netharn.output_shape_for import OutputShapeFor
        >>> self = L2Norm(in_features=7, scale=20)
        >>> OutputShapeFor(self)._check_consistency((1, 7, 2, 2))
        (1, 7, 2, 2)
    """

    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self._initial_scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self._initial_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x

    def output_shape_for(self, input_shape):
        return input_shape


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def output_shape_for(self, input_shape):
        return tuple([input_shape[i] for i in self.dims])
