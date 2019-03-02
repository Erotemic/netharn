# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
try:
    from .reorg_layer_c import ReorgLayerC  # NOQA
except Exception:
    pass


class ReorgLayerPy(torch.nn.Module):
    """
    TODO: is this faster?

    References:
        https://github.com/ruiminshen/yolo2-pytorch/blob/master/model/yolo2.py

    Example:
        >>> x = torch.randn(5, 512, 12, 12)
        >>> alt = ReorgLayerPy(in_channels=512, stride=2)
        >>> out2 = alt.forward(x)
        >>> print(tuple(out2.shape))

    Ignore:
        >>> C = 512
        >>> x = torch.randn(5, C, 12, 12)
        >>> alt = ReorgLayerPy(in_channels=C, stride=2)
        >>> self = ReorgLayerC(in_channels=C, stride=2)
        >>> out2 = alt.forward(x)
        >>> out1 = self.forward(x)

        from clab import xpu_device
        xpu = xpu_device.XPU('gpu')
        self = xpu.mount(self)
        alt = xpu.mount(alt)

        a = torch.randn(5, C, 12, 12, requires_grad=True)
        b = torch.randn(5, C, 12, 12, requires_grad=True)
        a, b = xpu.variables(a, b, requires_grad=True)
        x = a + b

        import ubelt
        for timer in ubelt.Timerit(1000, bestof=10, label='label torch_ext'):
            with timer:
                self.forward(x)

        import ubelt
        for timer in ubelt.Timerit(1000, bestof=10, label='pure python'):
            with timer:
                out = alt.forward(x)

        a = torch.randn(5, C, 12, 12, requires_grad=True)
        b = torch.randn(5, C, 12, 12, requires_grad=True)
        a, b = xpu.variables(a, b, requires_grad=True)
        x = a + b

        import ubelt
        for timer in ubelt.Timerit(1000, bestof=10, label='label torch_ext'):
            with timer:
                y = self.forward(x)
                y.sum().backward()

        import ubelt
        for timer in ubelt.Timerit(1000, bestof=10, label='pure python'):
            with timer:
                y = alt.forward(x)
                y.sum().backward()

        # Forward Results:
        # Timed label torch_ext for: 1000 loops, best of 10
        #     time per loop: best=237.9 µs, mean=262.4 ± 3.1 µs
        # Timed pure python for: 1000 loops, best of 10
        #     time per loop: best=44.75 µs, mean=234.0 ± 1.5e+02 µs

        # Backwards Results:
        # Timed label torch_ext for: 1000 loops, best of 10
        #     time per loop: best=826.2 µs, mean=900.9 ± 2e+01 µs
        # Timed pure python for: 1000 loops, best of 10
        #     time per loop: best=0.7154 ms, mean=1.021 ± 0.038 ms
    """
    def __init__(self, in_channels, stride):
        super(ReorgLayerPy, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.out_channels = in_channels * (stride ** 2)

    def forward(self, x):
        sh = sw = self.stride
        # batch_size, channels, height, width
        B, C, H, W = x.size()
        H2, W2 = H // sh, W // sw
        x = x.view(B, C, H2, sh, W2, sw).transpose(3, 4).contiguous()
        x = x.view(B, C, H2 * W2, sh * sw).transpose(2, 3).contiguous()
        x = x.view(B, C, sh * sw, H2, W2).transpose(1, 2).contiguous()
        x = x.view(B, -1, H2, W2)
        return x

# The python version is just as fast as the C version
# ReorgLayer = ReorgLayerC
ReorgLayer = ReorgLayerPy


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m xdoctest clab.models.yolo2.layers.reorg.reorg_layer all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
