import torch
import numpy as np
import itertools as it
from torch.utils import data as torch_data
from netharn.data import base
from netharn import util
import ubelt as ub


class ToyData1d(torch_data.Dataset, base.DataMixin):
    def __init__(self, rng=None):
        """
        Spiral 2d data points

        CommandLine:
            python ~/code/netharn/netharn/data/toydata.py ToyData1d --show

        Example:
            >>> dset = ToyData1d()
            >>> data, labels = next(iter(dset.make_loader(batch_size=2000)))
            >>> # xdoctest: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> cls1 = data[labels == 0]
            >>> cls2 = data[labels == 1]
            >>> from matplotlib import pyplot as plt
            >>> a, b = cls1.T.numpy()
            >>> c, d = cls2.T.numpy()
            >>> plt.plot(a, b, 'rx')
            >>> plt.plot(c, d, 'bx')
            >>> mplutil.show_if_requested()
        """
        rng = util.ensure_rng(rng)

        # spiral equation in parameteric form:
        # x(t) = r(t) * cos(t)
        # y(t) = r(t) * sin(t)

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

        suffix = ub.hash_data([rng], base='abc', hasher='sha1')[0:16]
        self.input_id = 'TD1D_{}_'.format(n) + suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])
        label = int(self.labels[index])
        return data, label


class ToyData2d(torch_data.Dataset, base.DataMixin):
    """
    CommandLine:
        python ~/code/netharn/netharn/data/toydata.py ToyData2d --show

    Example:
        >>> self = ToyData2d()
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
    def __init__(self, size=4, border=1, n=100, rng=None):
        rng = util.ensure_rng(rng)

        h = w = size

        whiteish = 1 - (np.abs(rng.randn(n, 1, h, w) / 4) % 1)
        blackish = (np.abs(rng.randn(n, 1, h, w) / 4) % 1)

        fw = border
        slices = [slice(None, fw), slice(-fw, None)]

        # class 0 is white block inside a black frame
        data1 = whiteish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data1[..., sl1, :] = blackish[..., sl1, :]
            data1[..., :, sl2] = blackish[..., :, sl2]

        # class 1 is black block inside a white frame
        data2 = blackish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data2[..., sl1, :] = whiteish[..., sl1, :]
            data2[..., :, sl2] = whiteish[..., :, sl2]

        self.data = np.concatenate([data1, data2], axis=0)
        self.labels = np.array(([0] * n) + ([1] * n))

        suffix = ub.hash_data([
            size, border, n, rng
        ], base='abc', hasher='sha1')[0:16]
        self.input_id = 'TD2D_{}_'.format(n) + suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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
