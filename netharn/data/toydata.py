from torch.utils import data as torch_data
import numpy as np


class DataMixin(object):
    def make_loader(self, *args, **kwargs):
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader


class ToyData1d(torch_data.Dataset, DataMixin):
    def __init__(self, rng=None):
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

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        import torch
        data = torch.FloatTensor(self.data[index])
        label = int(self.labels[index])
        return data, label


class ToyData2d(torch_data.Dataset, DataMixin):
    """
    CommandLine:
        python ~/code/netharn/netharn/data/toydata.py ToyData2D --show

    Example:
        >>> self = ToyData2D()
        >>> data1, label1 = self[0]
        >>> data2, label2 = self[-1]
        >>> # xdoctest: +REQUIRES(--show)
        >>> from netharn.util import mplutil
        >>> mplutil.qtensure()
        >>> mplutil.figure(fnum=1, doclf=True)
        >>> mplutil.imshow(data1.numpy().squeeze(), pnum=(1, 2, 1))
        >>> mplutil.imshow(data2.numpy().squeeze(), pnum=(1, 2, 2))
        >>> mplutil.show_if_requested()
    """
    def __init__(self, rng=None):
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

        self.data = np.concatenate([data1, data2], axis=0)
        self.labels = np.array(([0] * n) + ([1] * n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        import torch
        data = torch.FloatTensor(self.data[index])
        label = int(self.labels[index])
        return data, label


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/data/toydata.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
