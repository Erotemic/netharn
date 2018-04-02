import torch
from torch.autograd import Function
from ._ext import reorg_layer


class ReorgFunction(Function):
    """
    Example:
        >>> x = torch.randn(5, 512, 12, 12)
        >>> self = ReorgFunction()
        >>> out = self.forward(x)
        >>> print(tuple(out.shape))
        (5, 2048, 6, 6)
    """
    def __init__(self, stride=2):
        self.stride = stride

    def forward(self, x):
        stride = self.stride

        bsize, c, h, w = x.size()
        out_w, out_h, out_c = int(w / stride), int(h / stride), c * (stride * stride)  # noqa
        out = torch.FloatTensor(bsize, out_c, out_h, out_w)

        if x.is_cuda:
            out = out.cuda()
            reorg_layer.reorg_cuda(x, out_w, out_h, out_c, bsize,
                                   stride, 0, out)
        else:
            reorg_layer.reorg_cpu(x, out_w, out_h, out_c, bsize,
                                  stride, 0, out)

        return out

    def backward(self, grad_top):
        stride = self.stride
        bsize, c, h, w = grad_top.size()

        out_w, out_h, out_c = w * stride, h * stride, c / (stride * stride)
        grad_bottom = torch.FloatTensor(bsize, int(out_c), out_h, out_w)

        # rev_stride = 1. / stride    # reverse
        if grad_top.is_cuda:
            grad_bottom = grad_bottom.cuda()
            reorg_layer.reorg_cuda(grad_top, w, h, c, bsize,
                                   stride, 1, grad_bottom)
        else:
            reorg_layer.reorg_cpu(grad_top, w, h, c, bsize,
                                  stride, 1, grad_bottom)

        return grad_bottom


class ReorgLayerC(torch.nn.Module):
    """
    Restructure

    Example:
        >>> x = torch.randn(5, 512, 12, 12)
        >>> self = ReorgLayerC(in_channels=512, stride=2)
        >>> out = self.forward(x)
        >>> print(tuple(out.shape))
        (5, 2048, 6, 6)
    """
    def __init__(self, in_channels, stride):
        super(ReorgLayerC, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        # stride*stride times the number of input channels
        self.out_channels = in_channels * (stride ** 2)

    def forward(self, x):
        x = ReorgFunction(self.stride)(x)
        return x


