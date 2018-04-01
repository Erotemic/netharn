import torch
import numpy as np


class ToyNet1d(torch.nn.Module):
    """
    Demo model for a simple 2 class learning problem

    Example:
        >>> self = ToyNet1d()
        >>> data, true = ToyNet1d.demodata()
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
        """
        Math:
            # demodata equation
            x(t) = r(t) * cos(t)
            y(t) = r(t) * sin(t)

        Example:
            >>> data, labels = ToyNet1d.demodata()
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()
            >>> from matplotlib import pyplot as plt
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> cls1 = data[labels == 0]
            >>> cls2 = data[labels == 1]
            >>> plt.plot(*cls1.T, 'rx')
            >>> plt.plot(*cls2.T, 'bx')
        """
        import numpy as np
        # rng = util.ensure_rng(rng)
        if rng is None:
            rng = np.random.RandomState()

        # class 1
        n = 1000
        theta1 = rng.rand(n) * 10
        x1 = theta1 * np.cos(theta1)
        y1 = theta1 * np.sin(theta1)

        theta2 = rng.rand(n) * 10
        x2 = -theta2 * np.cos(theta2)
        y2 = -theta2 * np.sin(theta2)

        data = []
        labels = []

        data.extend(list(zip(x1, y1)))
        labels.extend([0] * n)

        data.extend(list(zip(x2, y2)))
        labels.extend([1] * n)

        data = np.array(data)
        labels = np.array(labels)

        sortx = np.arange(len(data))
        rng.shuffle(sortx)
        data = data[sortx]
        labels = labels[sortx]

        return data, labels


class ToyNet2D(torch.nn.Module):
    """
    Demo model for a simple 2 class learning problem

    Example:
        >>> self = ToyNet2D()
        >>> data, true = ToyNet2D.demodata()
        >>> inputs = torch.Tensor(data)
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
        """
        Math:
            # demodata equation
            x(t) = r(t) * cos(t)
            y(t) = r(t) * sin(t)

        Example:
            >>> data, labels = ToyNet2d.demodata()
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()
            >>> from matplotlib import pyplot as plt
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> mplutil.imshow(data1[0, 0], pnum=(1, 2, 1))
            >>> mplutil.imshow(data2[0, 0], pnum=(1, 2, 2))
        """
        import numpy as np
        if rng is None:
            rng = np.random.RandomState()
        n = 100

        whiteish = rng.rand(n, 1, 8, 8) * .9
        blackish = rng.rand(n, 1, 8, 8) * .2
        # class 0 is white block inside a black frame
        # class 1 is black block inside a white frame

        import itertools as it
        fw = 2  # frame width
        slices = [slice(None, fw), slice(-fw, None)]

        data1 = whiteish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data1[..., sl1, :] = blackish[..., sl1, :]
            data1[..., :, sl2] = blackish[..., :, sl2]

        data2 = blackish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data2[..., sl1, :] = whiteish[..., sl1, :]
            data2[..., :, sl2] = whiteish[..., :, sl2]

        data = np.concatenate([data1, data2], axis=0)
        labels = np.array(([0] * n) + ([1] * n))

        return data, labels
