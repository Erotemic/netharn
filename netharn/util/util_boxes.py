import numpy as np
import torch
import ubelt as ub

try:
    from netharn.util.cython_boxes import bbox_ious_c
except ImportError:
    bbox_ious_c = None


def box_ious(boxes1, boxes2, bias=0, mode=None):
    """
    Args:
        boxes1 (ndarray): (N, 4) tlbr format
        boxes2 (ndarray): (K, 4) tlbr format
        bias (int): either 0 or 1, does tl=br have area of 0 or 1?

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> boxes1 = Boxes.random(5, scale=10.0, rng=0, format='tlbr').data
        >>> boxes2 = Boxes.random(7, scale=10.0, rng=1, format='tlbr').data
        >>> ious = box_ious(boxes1, boxes2)
        >>> print(ub.repr2(ious.tolist(), precision=2))
        [
            [0.00, 0.00, 0.28, 0.00, 0.00, 0.20, 0.01],
            [0.00, 0.00, 0.50, 0.00, 0.04, 0.06, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.19, 0.03, 0.00, 0.00, 0.00],
        ]

    Example:
        >>> boxes1 = Boxes.random(5, scale=10.0, rng=0, format='tlbr').data
        >>> boxes2 = Boxes.random(7, scale=10.0, rng=1, format='tlbr').data
        >>> ious_c = box_ious(boxes1, boxes2, bias=0, mode='c')
        >>> ious_py = box_ious(boxes1, boxes2, bias=0, mode='py')
        >>> assert np.all(np.isclose(ious_c, ious_py))
        >>> ious_c = box_ious(boxes1, boxes2, bias=1, mode='c')
        >>> ious_py = box_ious(boxes1, boxes2, bias=1, mode='py')
        >>> assert np.all(np.isclose(ious_c, ious_py))
    """
    if mode is None:
        mode = 'py' if bbox_ious_c is None else 'c'
    if mode == 'c':
        return bbox_ious_c(boxes1.astype(np.float32),
                           boxes2.astype(np.float32), bias)
    elif mode == 'py':
        return box_ious_py(boxes1, boxes2, bias)
    else:
        raise KeyError(mode)


def box_ious_torch(boxes1, boxes2, bias=1):
    """
    Example:
        >>> boxes1 = Boxes.random(5, scale=10.0, rng=0, format='tlbr', tensor=True).data
        >>> boxes2 = Boxes.random(7, scale=10.0, rng=1, format='tlbr', tensor=True).data
        >>> bias = 1
        >>> ious = box_ious_torch(boxes1, boxes2, bias)
        >>> ious_np = box_ious_py(boxes1.numpy(), boxes2.numpy(), bias)
        >>> assert np.all(ious_np == ious.numpy())

    Benchmark:

        import ubelt
        import netharn as nh

        N = 100

        ydata = ub.ddict(list)
        xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700]

        for num in xdata:
            boxes1 = nh.util.Boxes.random(num, scale=10.0, rng=0, format='tlbr', tensor=True).data
            boxes2 = nh.util.Boxes.random(num + 1, scale=10.0, rng=1, format='tlbr', tensor=True).data
            t1 = ubelt.Timerit(N, bestof=10, label='time-torch-cpu')
            for timer in t1:
                with timer:
                    box_ious_torch(boxes1, boxes2, bias)
            ydata['cpu'].append(t1.ave_secs)

            boxes1 = boxes1.cuda()
            boxes2 = boxes2.cuda()
            t2 = ubelt.Timerit(N, bestof=10, label='time-torch-gpu')
            for timer in t2:
                with timer:
                    box_ious_torch(boxes1, boxes2, bias)
                    torch.cuda.synchronize()
            ydata['gpu'].append(t2.ave_secs)

            boxes1 = boxes1.cpu().numpy()
            boxes2 = boxes2.cpu().numpy()
            t3 = ubelt.Timerit(N, bestof=10, label='time-numpy')
            for timer in t3:
                with timer:
                    box_ious_py(boxes1, boxes2, bias)
            ydata['numpy'].append(t3.ave_secs)

        nh.util.mplutil.qtensure()
        nh.util.mplutil.multi_plot(xdata, ydata, xlabel='num boxes', ylabel='seconds')



    """
    w1 = boxes1[:, 2] - boxes1[:, 0] + bias
    h1 = boxes1[:, 3] - boxes1[:, 1] + bias
    w2 = boxes2[:, 2] - boxes2[:, 0] + bias
    h2 = boxes2[:, 3] - boxes2[:, 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = torch.min(boxes1[:, 2][:, None], boxes2[:, 2])
    x_mins = torch.max(boxes1[:, 0][:, None], boxes2[:, 0])

    iws = (x_maxs - x_mins + bias).clamp(0, float('inf'))

    y_maxs = torch.min(boxes1[:, 3][:, None], boxes2[:, 3])
    y_mins = torch.max(boxes1[:, 1][:, None], boxes2[:, 1])

    ihs = (y_maxs - y_mins + bias).clamp(0, float('inf'))

    areas_sum = (areas1[:, None] + areas2)

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    ious = inter_areas / union_areas
    return ious


def box_ious_py(boxes1, boxes2, bias=1):
    """
    This is the fastest python implementation of bbox_ious I found
    """
    w1 = boxes1[:, 2] - boxes1[:, 0] + bias
    h1 = boxes1[:, 3] - boxes1[:, 1] + bias
    w2 = boxes2[:, 2] - boxes2[:, 0] + bias
    h2 = boxes2[:, 3] - boxes2[:, 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = np.minimum(boxes1[:, 2][:, None], boxes2[:, 2])
    x_mins = np.maximum(boxes1[:, 0][:, None], boxes2[:, 0])

    iws = np.maximum(x_maxs - x_mins + bias, 0)
    # note: it would be possible to significantly reduce the computation by
    # filtering any box pairs where iws <= 0. Not sure how to do with numpy.

    y_maxs = np.minimum(boxes1[:, 3][:, None], boxes2[:, 3])
    y_mins = np.maximum(boxes1[:, 1][:, None], boxes2[:, 1])

    ihs = np.maximum(y_maxs - y_mins + bias, 0)

    areas_sum = (areas1[:, None] + areas2)

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    ious = inter_areas / union_areas
    return ious


class Boxes(ub.NiceRepr):
    """
    Converts boxes between different formats as long as the last dimension
    contains 4 coordinates and the format is specified.

    This is a convinience class, and should not not store the data for very
    long. The general idiom should be create class, convert data, and then get
    the raw data and let the class be garbage collected. This will help ensure
    that your code is portable and understandable if this class is not
    available.

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> Boxes([25, 30, 15, 10], 'xywh')
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_xywh()
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_cxywh()
        <Boxes(cxywh, array([32.5, 35. , 15. , 10. ]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').to_tlbr()
        <Boxes(tlbr, array([25, 30, 40, 40]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').scale(2).to_tlbr()
        <Boxes(tlbr, array([50., 60., 80., 80.]))>
        >>> Boxes(torch.FloatTensor([[25, 30, 15, 20]]), 'xywh').scale(.1).to_tlbr()
        <Boxes(tlbr,
             2.5000  3.0000  4.0000  5.0000
            [torch.FloatTensor of size ...

    Example:
        >>> datas = [
        >>>     [1, 2, 3, 4],
        >>>     [[1, 2, 3, 4], [4, 5, 6, 7]],
        >>>     [[[1, 2, 3, 4], [4, 5, 6, 7]]],
        >>> ]
        >>> formats = ['xywh', 'cxywh', 'tlbr']
        >>> for format1 in formats:
        >>>     for data in datas:
        >>>         self = box1 = Boxes(data, format1)
        >>>         for format2 in formats:
        >>>             box2 = box1.toformat(format2)
        >>>             back = box2.toformat(format1)
        >>>             assert box1 == back
    """
    def __init__(self, data, format='xywh'):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        self.data = data
        self.format = format

    def __eq__(self, other):
        return np.all(self.data == other.data) and self.format == other.format

    def __nice__(self):
        # return self.format + ', shape=' + str(list(self.data.shape))
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        return '{}, {}'.format(self.format, data_repr)

    __repr__ = ub.NiceRepr.__str__

    @classmethod
    def random(Boxes, num=1, scale=1.0, format='xywh', rng=None, tensor=False):
        """
        Makes random boxes

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes.random(3, rng=0, scale=100)
            <Boxes(xywh,
                array([[27, 35, 30, 27],
                       [21, 32, 21, 44],
                       [48, 19, 39, 26]]))>
            >>> Boxes.random(3, rng=0, scale=100, tensor=True)
            <Boxes(xywh,
                 27  35  30  27
                 21  32  21  44
                 48  19  39  26
                [torch.LongTensor of size ...
        """
        from netharn import util
        rng = util.ensure_rng(rng)

        xywh = (rng.rand(num, 4) * scale / 2)
        as_integer = isinstance(scale, int)
        if as_integer:
            xywh = xywh.astype(np.int)
        if tensor:
            if as_integer:
                xywh = torch.LongTensor(xywh)
            else:
                xywh = torch.FloatTensor(xywh)
        boxes = Boxes(xywh, format='xywh').toformat(format, copy=False)
        return boxes

    def copy(self):
        if torch.is_tensor(self.data):
            new_data = self.data.clone()
        else:
            new_data = self.data.copy()
        return Boxes(new_data, self.format)

    def scale(self, factor):
        r"""
        works with tlbr, cxywh, xywh, xy, or wh formats

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes(np.array([1, 1, 10, 10])).scale(2).data
            array([ 2.,  2., 20., 20.])
            >>> Boxes(np.array([[1, 1, 10, 10]])).scale((2, .5)).data
            array([[ 2. ,  0.5, 20. ,  5. ]])
            >>> Boxes(np.array([[10, 10]])).scale(.5).data
            array([[5., 5.]])
        """
        boxes = self.data
        sx, sy = factor if ub.iterable(factor) else (factor, factor)
        if torch.is_tensor(boxes):
            new_data = boxes.float().clone()
        else:
            if boxes.dtype.kind != 'f':
                new_data = boxes.astype(np.float)
            else:
                new_data = boxes.copy()
        new_data[..., 0:4:2] *= sx
        new_data[..., 1:4:2] *= sy
        return Boxes(new_data, self.format)

    def shift(self, amount):
        """
        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes([25, 30, 15, 10], 'xywh').shift(10)
            <Boxes(xywh, array([35., 40., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'xywh').shift((10, 0))
            <Boxes(xywh, array([35., 30., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'tlbr').shift((10, 5))
            <Boxes(tlbr, array([35., 35., 25., 15.]))>
        """
        boxes = self.data
        tx, ty = amount if ub.iterable(amount) else (amount, amount)
        if torch.is_tensor(boxes):
            new_data = boxes.float().clone()
        else:
            new_data = boxes.astype(np.float).copy()
        if self.format in ['xywh', 'cxywh']:
            new_data[..., 0] += tx
            new_data[..., 1] += ty
        elif self.format in ['tlbr']:
            new_data[..., 0:4:2] += tx
            new_data[..., 1:4:2] += ty
        else:
            raise KeyError(self.format)
        return Boxes(new_data, self.format)

    @property
    def shape(self):
        return self.data.shape

    @property
    def area(self):
        """
        Example:
            >>> Boxes([25, 30, 15, 10], 'xywh').area
            array([150])
            >>> Boxes([[25, 30, 0, 0]], 'xywh').area
            array([[0]])
        """
        w, h = self.to_xywh().components[-2:]
        return w * h

    @property
    def components(self):
        a = self.data[..., 0:1]
        b = self.data[..., 1:2]
        c = self.data[..., 2:3]
        d = self.data[..., 3:4]
        return [a, b, c, d]

    @classmethod
    def _cat(cls, datas):
        if torch.is_tensor(datas[0]):
            return torch.cat(datas, dim=-1)
        else:
            return np.concatenate(datas, axis=-1)

    def toformat(self, format, copy=True):
        if format == 'xywh':
            return self.to_xywh(copy=copy)
        elif format == 'tlbr':
            return self.to_tlbr(copy=copy)
        elif format == 'cxywh':
            return self.to_cxywh(copy=copy)
        elif format == 'extent':
            return self.to_extent(copy=copy)
        else:
            raise KeyError('Cannot convert {} to {}'.format(self.format, format))

    def to_extent(self, copy=True):
        if self.format == 'extent':
            return self.copy() if copy else self
        else:
            # Only difference between tlbr and extent is the column order
            # extent is x1, x2, y1, y2
            tlbr = self.to_tlbr().data
            extent = tlbr[..., [0, 2, 1, 3]]
        return Boxes(extent, 'extent')

    def to_xywh(self, copy=True):
        if self.format == 'xywh':
            return self.copy() if copy else self
        elif self.format == 'tlbr':
            x1, y1, x2, y2 = self.components
            w = x2 - x1
            h = y2 - y1
        elif self.format == 'cxywh':
            cx, cy, w, h = self.components
            x1 = cx - w / 2
            y1 = cy - h / 2
        else:
            raise KeyError(self.format)
        xywh = self._cat([x1, y1, w, h])
        return Boxes(xywh, 'xywh')

    def to_cxywh(self, copy=True):
        if self.format == 'cxywh':
            return self.copy() if copy else self
        elif self.format == 'tlbr':
            x1, y1, x2, y2 = self.components
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        elif self.format == 'xywh':
            x1, y1, w, h = self.components
            cx = x1 + (w / 2)
            cy = y1 + (h / 2)
        else:
            raise KeyError(self.format)
        cxywh = self._cat([cx, cy, w, h])
        return Boxes(cxywh, 'cxywh')

    def to_tlbr(self, copy=True):
        if self.format == 'tlbr':
            return self.copy() if copy else self
        elif self.format == 'cxywh':
            cx, cy, w, h = self.components
            half_w = (w / 2)
            half_h = (h / 2)
            x1 = cx - half_w
            x2 = cx + half_w
            y1 = cy - half_h
            y2 = cy + half_h
        elif self.format == 'xywh':
            x1, y1, w, h = self.components
            x2 = x1 + w
            y2 = y1 + h
        else:
            raise KeyError(self.format)
        tlbr = self._cat([x1, y1, x2, y2])
        return Boxes(tlbr, 'tlbr')

    def to_imgaug(self, shape):
        """
        Args:
            shape (tuple): shape of image that boxes belong to

        Example:
            >>> self = Boxes([[25, 30, 15, 10]], 'tlbr')
            >>> bboi = self.to_imgaug((10, 10))
        """
        import imgaug
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))

        tlbr = self.to_tlbr(copy=False).data
        bboi = imgaug.BoundingBoxesOnImage(
            [imgaug.BoundingBox(x1, y1, x2, y2)
             for x1, y1, x2, y2 in tlbr], shape=shape)
        return bboi

    @classmethod
    def from_imgaug(Boxes, bboi):
        """
        Args:
            bboi (ia.BoundingBoxesOnImage):

        Example:
            >>> orig = Boxes.random(5, format='tlbr')
            >>> bboi = orig.to_imgaug(shape=(500, 500))
            >>> self = Boxes.from_imgaug(bboi)
            >>> assert np.all(self.data == orig.data)
        """
        tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                         for bb in bboi.bounding_boxes])
        tlbr = tlbr.reshape(-1, 4)
        return Boxes(tlbr, format='tlbr')

    def clip(self, x_min, y_min, x_max, y_max, inplace=False):
        """
        Clip boxes to image boundaries.  If box is in tlbr format, inplace
        operation is an option.

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> boxes = Boxes(np.array([[-10, -10, 120, 120], [1, -2, 30, 50]]), 'tlbr')
            >>> clipped = boxes.clip(0, 0, 110, 100, inplace=False)
            >>> assert np.any(boxes.data != clipped.data)
            >>> clipped2 = boxes.clip(0, 0, 110, 100, inplace=True)
            >>> assert clipped2.data is boxes.data
            >>> assert np.all(clipped2.data == clipped.data)
            >>> print(clipped)
            <Boxes(tlbr,
                array([[  0,   0, 110, 100],
                       [  1,   0,  30,  50]]))>
        """
        if inplace:
            if self.format != 'tlbr':
                raise ValueError('Must be in tlbr format to operate inplace')
            self2 = self
        else:
            self2 = self.to_tlbr(copy=True)
        x1, y1, x2, y2 = self2.data.T
        np.clip(x1, x_min, x_max, out=x1)
        np.clip(y1, y_min, y_max, out=y1)
        np.clip(x2, x_min, x_max, out=x2)
        np.clip(y2, y_min, y_max, out=y2)
        return self2

    def transpose(self):
        x, y, w, h = self.to_xywh().components
        self2 = self.__class__(self._cat([y, x, h, w]), format='xywh')
        self2 = self2.toformat(self.format)
        return self2

    def compress(self, flags, axis=0, inplace=False):
        """
        Filters boxes based on a boolean criterion

        Example:
            >>> self = Boxes([[25, 30, 15, 10]], 'tlbr')
            >>> flags = [False]
        """
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))
        self2 = self if inplace else self.copy()
        self2.data = self2.data.compress(flags, axis=axis)
        return self2


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_boxes all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
