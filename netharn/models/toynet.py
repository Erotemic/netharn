import torch
import numpy as np


class ToyNet1d(torch.nn.Module):
    """
    Demo model for a simple 2 class learning problem

    Example:
        >>> self = ToyNet1d()
        >>> dset = ToyNet1d.demodata()
        >>> inputs = torch.Tensor(data)
        >>> prob = self(inputs)
        >>> conf, pred = probs.max(dim=1)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.layers = torch.nn.Sequential(*[
            torch.nn.Linear(2, 8),
            # torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8, 8),
            # torch.nn.BatchNorm1d(8),
            torch.nn.Linear(8, num_classes),
            torch.nn.Softmax(dim=1)
        ])

    def forward(self, inputs):
        return self.layers(inputs)

    @classmethod
    def demodata(ToyNet1d, rng=None):
        from netharn.data.toydata import ToyData1d
        dset = ToyData1d(rng)
        return dset


class ToyNet2D(torch.nn.Module):
    """
    Demo model for a simple 2 class learning problem

    Example:
        >>> self = ToyNet2D()
        >>> data, true = ToyNet2D.demodata()[0]
        >>> inputs = torch.Tensor(data[None, :])
        >>> prob = self(inputs)
        >>> conf, pred = probs.max(dim=1)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.layers = torch.nn.Sequential(*[
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, num_classes, kernel_size=3, padding=1, bias=False),
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        spatial_out = self.layers(inputs)
        num = float(np.prod(spatial_out.shape[-2:]))
        averaged = spatial_out.sum(dim=2).sum(dim=2) / num
        probs = self.softmax(averaged)
        return probs

    @classmethod
    def demodata(ToyNet2d, rng=None):
        from netharn.data import ToyData2d
        dset = ToyData2d(rng)
        return dset
