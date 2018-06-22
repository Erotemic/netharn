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
        >>> if bbox_ious_c is not None:
        >>>     ious_c = box_ious(boxes1, boxes2, bias=0, mode='c')
        >>>     ious_py = box_ious(boxes1, boxes2, bias=0, mode='py')
        >>>     assert np.all(np.isclose(ious_c, ious_py))
        >>>     ious_c = box_ious(boxes1, boxes2, bias=1, mode='c')
        >>>     ious_py = box_ious(boxes1, boxes2, bias=1, mode='py')
        >>>     assert np.all(np.isclose(ious_c, ious_py))
    """
    if mode is None:
        if torch.is_tensor(boxes1):
            mode = 'torch'
        else:
            mode = 'py' if bbox_ious_c is None else 'c'

    if mode == 'torch' or torch.is_tensor(boxes1):
        # TODO: add tests for equality with other methods or show why it should
        # be different.
        # NOTE: this is done in boxes.ious
        return box_ious_torch(boxes1, boxes2, bias)
    elif mode == 'c':
        return bbox_ious_c(boxes1.astype(np.float32),
                           boxes2.astype(np.float32), bias)
    elif mode == 'py':
        return box_ious_py(boxes1, boxes2, bias)
    else:
        raise KeyError(mode)


# def light_bbox_ious_cxywh_bias0(boxes1, boxes2):
#     """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

#     Notes:
#         Accepts cxywh format and the bias is 0

#     Args:
#         boxes1 (torch.Tensor): List of bounding boxes
#         boxes2 (torch.Tensor): List of bounding boxes

#     Note:
#         List format: [[xc, yc, w, h],...]

#     Example:
#         >>> boxes1 = Boxes.random(5, scale=10.0, rng=0, format='cxywh', tensor=True).data
#         >>> boxes2 = Boxes.random(7, scale=10.0, rng=1, format='cxywh', tensor=True).data
#         >>> bias = 1
#         >>> ious = light_bbox_ious(boxes1, boxes2)
#     """
#     b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
#     b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
#     b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
#     b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

#     dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
#     dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
#     intersections = dx * dy

#     areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
#     areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
#     unions = (areas1 + areas2.t()) - intersections
#     return intersections / unions


def box_ious_torch(boxes1, boxes2, bias=1):
    """
    Example:
        >>> boxes1 = Boxes.random(5, scale=10.0, rng=0, format='tlbr', tensor=True).data
        >>> boxes2 = Boxes.random(7, scale=10.0, rng=1, format='tlbr', tensor=True).data
        >>> bias = 0
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
    # boxes1 = boxes1.view(-1, 4)
    # boxes2 = boxes2.view(-1, 4)

    w1 = boxes1[..., 2] - boxes1[..., 0] + bias
    h1 = boxes1[..., 3] - boxes1[..., 1] + bias
    w2 = boxes2[..., 2] - boxes2[..., 0] + bias
    h2 = boxes2[..., 3] - boxes2[..., 1] + bias

    areas1 = w1 * h1
    areas2 = w2 * h2

    x_maxs = torch.min(boxes1[..., 2][..., None], boxes2[..., 2])
    x_mins = torch.max(boxes1[..., 0][..., None], boxes2[..., 0])

    iws = (x_maxs - x_mins + bias).clamp(0, float('inf'))

    y_maxs = torch.min(boxes1[..., 3][..., None], boxes2[..., 3])
    y_mins = torch.max(boxes1[..., 1][..., None], boxes2[..., 1])

    ihs = (y_maxs - y_mins + bias).clamp(0, float('inf'))

    areas_sum = (areas1[..., None] + areas2)

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


class _BoxConversionMixins(object):
    """
    Methods for converting between different bounding box formats
    """

    def toformat(self, format, copy=True):
        """
        Changes the internal representation of the bounding box
        """
        if format == 'tlwh':
            return self.to_tlwh(copy=copy)
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

    def to_tlwh(self, copy=True):
        if self.format == 'tlwh':
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
        tlwh = self._cat([x1, y1, w, h])
        return Boxes(tlwh, 'tlwh')

    def to_cxywh(self, copy=True):
        if self.format == 'cxywh':
            return self.copy() if copy else self
        elif self.format == 'tlbr':
            x1, y1, x2, y2 = self.components
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        elif self.format == 'tlwh':
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
        elif self.format == 'tlwh':
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


class _BoxPropertyMixins(object):

    @property
    def xy_center(self):
        """ Returns the xy coordinates of the box centers """
        xy = self.to_cxywh(copy=False).data[..., 0:2]
        return xy

    @property
    def components(self):
        a = self.data[..., 0:1]
        b = self.data[..., 1:2]
        c = self.data[..., 2:3]
        d = self.data[..., 3:4]
        return [a, b, c, d]

    @property
    def shape(self):
        return self.data.shape

    @property
    def area(self):
        """
        Example:
            >>> Boxes([25, 30, 15, 10], 'tlwh').area
            array([150])
            >>> Boxes([[25, 30, 0, 0]], 'tlwh').area
            array([[0]])
        """
        w, h = self.to_tlwh().components[-2:]
        return w * h


def _numel(data):
    """ compatable API between torch and numpy """
    if isinstance(data, np.ndarray):
        return data.size
    else:
        return data.numel()


class _BoxTransformMixins(object):
    """
    methods for transforming bounding boxes
    """
    def scale(self, factor):
        r"""
        works with tlbr, cxywh, tlwh, xy, or wh formats

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

    def translate(self, amount):
        """
        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes([25, 30, 15, 10], 'tlwh').shift(10)
            <Boxes(tlwh, array([35., 40., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'tlwh').shift((10, 0))
            <Boxes(tlwh, array([35., 30., 15., 10.]))>
            >>> Boxes([25, 30, 15, 10], 'tlbr').shift((10, 5))
            <Boxes(tlbr, array([35., 35., 25., 15.]))>
        """
        boxes = self.data
        if not ub.iterable(amount):
            tx = ty = amount
        elif isinstance(amount, (list, tuple)):
            tx, ty = amount
        else:
            tx = amount[..., 0]
            ty = amount[..., 1]
        if torch.is_tensor(boxes):
            new_data = boxes.float().clone()
        else:
            new_data = boxes.astype(np.float).copy()
        if _numel(new_data) > 0:
            if self.format in ['tlwh', 'cxywh']:
                new_data[..., 0] += tx
                new_data[..., 1] += ty
            elif self.format in ['tlbr']:
                new_data[..., 0:4:2] += tx
                new_data[..., 1:4:2] += ty
            else:
                raise KeyError(self.format)
        return Boxes(new_data, self.format)

    shift = translate

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
        """
        Flips the box itself in data coordinates
        """
        x, y, w, h = self.to_tlwh().components
        self2 = self.__class__(self._cat([y, x, h, w]), format='tlwh')
        self2 = self2.toformat(self.format)
        return self2


class Boxes(ub.NiceRepr, _BoxConversionMixins, _BoxPropertyMixins, _BoxTransformMixins):
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
        >>> Boxes([25, 30, 15, 10], 'tlwh')
        <Boxes(tlwh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'tlwh').to_tlwh()
        <Boxes(tlwh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'tlwh').to_cxywh()
        <Boxes(cxywh, array([32.5, 35. , 15. , 10. ]))>
        >>> Boxes([25, 30, 15, 10], 'tlwh').to_tlbr()
        <Boxes(tlbr, array([25, 30, 40, 40]))>
        >>> Boxes([25, 30, 15, 10], 'tlwh').scale(2).to_tlbr()
        <Boxes(tlbr, array([50., 60., 80., 80.]))>
        >>> Boxes(torch.FloatTensor([[25, 30, 15, 20]]), 'tlwh').scale(.1).to_tlbr()
        <Boxes(tlbr, tensor([[ 2.5000,  3.0000,  4.0000,  5.0000]]))>

    Example:
        >>> datas = [
        >>>     [1, 2, 3, 4],
        >>>     [[1, 2, 3, 4], [4, 5, 6, 7]],
        >>>     [[[1, 2, 3, 4], [4, 5, 6, 7]]],
        >>> ]
        >>> formats = ['tlwh', 'cxywh', 'tlbr']
        >>> for format1 in formats:
        >>>     for data in datas:
        >>>         self = box1 = Boxes(data, format1)
        >>>         for format2 in formats:
        >>>             box2 = box1.toformat(format2)
        >>>             back = box2.toformat(format1)
        >>>             assert box1 == back
    """
    def __init__(self, data, format=None):
        if format is None:
            print('WARNING: format for Boxes not specified, default to tlwh')
            format = 'tlwh'
        CHECKS = False
        if CHECKS:
            if _numel(data) > 0 and data.shape[-1] == 4:
                raise ValueError('trailing dimension of boxes must be 4')

        if isinstance(data, (list, tuple)):
            data = np.array(data)
        self.data = data
        self.format = format

    def __getitem__(self, index):
        cls = self.__class__
        subset = cls(self.data[index], self.format)
        return subset

    def __eq__(self, other):
        """
        Tests equality of two Boxes objects

        Example:
            >>> box0 = box1 = Boxes([[1, 2, 3, 4]], 'tlwh')
            >>> box2 = Boxes(box0.data, 'tlbr')
            >>> box3 = Boxes([[0, 2, 3, 4]], box0.format)
            >>> box4 = Boxes(box0.data, box2.format)
            >>> assert box0 == box1
            >>> assert not box0 == box2
            >>> assert not box2 == box3
            >>> assert box2 == box4
        """
        return np.array_equal(self.data, other.data) and self.format == other.format

    def __nice__(self):
        # return self.format + ', shape=' + str(list(self.data.shape))
        data_repr = repr(self.data)
        if '\n' in data_repr:
            data_repr = ub.indent('\n' + data_repr.lstrip('\n'), '    ')
        return '{}, {}'.format(self.format, data_repr)

    __repr__ = ub.NiceRepr.__str__

    @classmethod
    def random(Boxes, num=1, scale=1.0, format='tlwh', rng=None, tensor=False):
        """
        Makes random boxes

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Boxes.random(3, rng=0, scale=100)
            <Boxes(tlwh,
                array([[27, 35, 30, 27],
                       [21, 32, 21, 44],
                       [48, 19, 39, 26]]))>
            >>> Boxes.random(3, rng=0, scale=100, tensor=True)
            <Boxes(tlwh,
                tensor([[ 27,  35,  30,  27],
                        [ 21,  32,  21,  44],
                        [ 48,  19,  39,  26]]))>
        """
        from netharn import util
        rng = util.ensure_rng(rng)

        tlwh = (rng.rand(num, 4) * scale / 2)
        as_integer = isinstance(scale, int)
        if as_integer:
            tlwh = tlwh.astype(np.int)
        if tensor:
            if as_integer:
                tlwh = torch.LongTensor(tlwh)
            else:
                tlwh = torch.FloatTensor(tlwh)
        boxes = Boxes(tlwh, format='tlwh').toformat(format, copy=False)
        return boxes

    def copy(self):
        if torch.is_tensor(self.data):
            new_data = self.data.clone()
        else:
            new_data = self.data.copy()
        return Boxes(new_data, self.format)

    @classmethod
    def _cat(cls, datas):
        if torch.is_tensor(datas[0]):
            return torch.cat(datas, dim=-1)
        else:
            return np.concatenate(datas, axis=-1)

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

    def numpy(self):
        """ converts tensors to numpy """
        new_self = self.copy()
        if torch.is_tensor(self.data):
            new_self.data = new_self.data.cpu().numpy()
        return new_self

    def ious(self, other, bias=0, mode=None):
        """
        Compute IOUs between these boxes and another set of boxes

        Examples:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> self = Boxes(np.array([[ 0,  0, 10, 10],
            >>>                        [10,  0, 20, 10],
            >>>                        [20,  0, 30, 10]]), 'tlbr')
            >>> other = Boxes(np.array([6, 2, 20, 10]), 'tlbr')
            >>> overlaps = self.ious(other, bias=1).round(2)
            >>> assert np.all(np.isclose(overlaps, [0.21, 0.63, 0.04])), repr(overlaps)

        Examples:
            >>> formats = ['cxywh', 'tlwh', 'tlbr']
            >>> istensors = [False, True]
            >>> results = {}
            >>> for format in formats:
            >>>     for tensor in istensors:
            >>>         boxes1 = Boxes.random(5, scale=10.0, rng=0, format=format, tensor=tensor)
            >>>         boxes2 = Boxes.random(7, scale=10.0, rng=1, format=format, tensor=tensor)
            >>>         ious = boxes1.ious(boxes2)
            >>>         results[(format, tensor)] = ious
            >>> results = {k: v.numpy() if torch.is_tensor(v) else v for k, v in results.items() }
            >>> results = {k: v.tolist() for k, v in results.items()}
            >>> print(ub.repr2(results, sk=True, precision=3, nl=2))
            >>> from functools import partial
            >>> assert ub.allsame(results.values(), partial(np.allclose, atol=1e-07))
        """
        other_is_1d = (len(other.shape) == 1)
        if other_is_1d:
            # `box_ious` expect 2d input
            other = other[None, :]

        self_tlbr = self.to_tlbr(copy=False)
        other_tlbr = other.to_tlbr(copy=False)
        ious = box_ious(self_tlbr.data, other_tlbr.data, bias=bias, mode=mode)

        if other_is_1d:
            ious = ious[..., 0]
        return ious

    def view(self, *shape):
        """
        Passthrough method to view or reshape

        Example:
            >>> self = Boxes.random(6, scale=10.0, rng=0, format='tlwh', tensor=True)
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
            >>> self = Boxes.random(6, scale=10.0, rng=0, format='tlbr', tensor=False)
            >>> assert list(self.view(3, 2, 4).data.shape) == [3, 2, 4]
        """
        if torch.is_tensor(self.data):
            data_ = self.data.view(*shape)
        else:
            data_ = self.data.reshape(*shape)
        return self.__class__(data_, self.format)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_boxes all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
