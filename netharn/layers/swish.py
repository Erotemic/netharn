"""
References:
    https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py

    https://discuss.pytorch.org/t/implementation-of-swish-a-self-gated-activation-function/8813

    https://arxiv.org/pdf/1710.05941.pdf
"""
import torch
from torch import nn


class _SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """
    When beta=1 this is Sigmoid-weighted Linear Unit (SiL)

    References:
        https://arxiv.org/pdf/1710.05941.pdf

    Example:
        >>> x = torch.linspace(-20, 20, 100, requires_grad=True)
        >>> self = Swish()
        >>> y = self(x)
        >>> y.sum().backward()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, y.data)}, fnum=1, pnum=(1, 2, 1))
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, x.grad)}, fnum=1, pnum=(1, 2, 2))
        >>> kwplot.show_if_requestd()

    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        """
        Equivalent to ``x * torch.sigmoid(x)``
        """
        if self.beta == 1:
            return _SwishFunction.apply(x)
        else:
            return x * torch.sigmoid(x * self.beta)

    def receptive_field_for(self, field):
        return field

    def output_shape_for(self, shape):
        return shape
