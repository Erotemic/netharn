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
        >>> boxes1 = Boxes(random_boxes(3, scale=100.0).numpy(), 'tlbr').data
        >>> boxes2 = Boxes(random_boxes(2, scale=100.0).numpy(), 'tlbr').data
        >>> ious_c = bbox_ious_c(boxes1, boxes2, bias=0)
        >>> ious_py1 = box_ious_py1(boxes1, boxes2, bias=0)
        >>> ious_py2 = box_ious_py2(boxes1, boxes2)
        >>> ious_py3 = box_ious_py3(boxes1, boxes2)
        >>> assert np.all(np.isclose(ious_c, ious_py1))
        >>> assert np.all(np.isclose(ious_c, ious_py2))
        >>> assert np.all(np.isclose(ious_c, ious_py3))
    """
    mode = 'py' if mode is None and bbox_ious_c is None else 'c'
    if mode == 'c':
        return bbox_ious_c(boxes1, boxes2, bias)
    elif mode == 'c':
        return box_ious_py1(boxes1, boxes2, bias)
    else:
        raise KeyError(mode)


def bboxes_iou_light(boxes1, boxes2):
    import itertools as it
    results = []
    for box1, box2 in it.product(boxes1, boxes2):
        result = bbox_iou_light(box1, box2)
        results.append(result)
    return results

def bbox_iou_light(box1, box2):
    """ Compute IOU between 2 bounding boxes
        Box format: [xc, yc, w, h]

    from netharn import util
    boxes1_ = util.Boxes(boxes1, 'tlbr').to_cxywh().data
    boxes2_ = util.Boxes(boxes2, 'tlbr').to_cxywh().data
    bboxes_iou_light(boxes1_, boxes2_)

    ious_c = bbox_ious_c(boxes1, boxes2, bias=0)

    """
    mx = min(box1[0]-box1[2]/2, box2[0]-box2[2]/2)
    Mx = max(box1[0]+box1[2]/2, box2[0]+box2[2]/2)
    my = min(box1[1]-box1[3]/2, box2[1]-box2[3]/2)
    My = max(box1[1]+box1[3]/2, box2[1]+box2[3]/2)
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]

    uw = Mx - mx
    uh = My - my
    iw = w1 + w2 - uw
    ih = h1 + h2 - uh
    if iw <= 0 or ih <= 0:
        return 0

    area1 = w1 * h1
    area2 = w2 * h2
    iarea = iw * ih
    uarea = area1 + area2 - iarea
    return iarea/uarea


def box_ious_py1(boxes1, boxes2, bias=1):
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


def box_ious_py3(boxes1, boxes2):
    N = boxes1.shape[0]
    K = boxes2.shape[0]

    # Preallocate output
    intersec = np.zeros((N, K), dtype=np.float32)

    inter_areas3 = np.zeros((N, K), dtype=np.float32)
    union_areas3 = np.zeros((N, K), dtype=np.float32)
    iws3 = np.zeros((N, K), dtype=np.float32)
    ihs3 = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        qbox_area = (
            (boxes2[k, 2] - boxes2[k, 0] + 1) *
            (boxes2[k, 3] - boxes2[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes1[n, 2], boxes2[k, 2]) -
                max(boxes1[n, 0], boxes2[k, 0]) + 1
            )
            iw = max(iw, 0)

            # if iw > 0:
            ih = (
                min(boxes1[n, 3], boxes2[k, 3]) -
                max(boxes1[n, 1], boxes2[k, 1]) + 1
            )
            ih = max(ih, 0)
            # if ih > 0:
            box_area = (
                (boxes1[n, 2] - boxes1[n, 0] + 1) *
                (boxes1[n, 3] - boxes1[n, 1] + 1)
            )
            inter_area = iw * ih
            union_area = (qbox_area + box_area - inter_area)

            ihs3[n, k] = ih
            iws3[n, k] = iw
            inter_areas3[n, k] = inter_area
            union_areas3[n, k] = union_area

            intersec[n, k] = inter_area / union_area
    return intersec


def box_ious_py2(boxes1, boxes2):
    """
    Implementation using 2d index based filtering.
    It turns out that this is slower than the dense version.

    for a 5x7:
        %timeit box_ious_py2(boxes1, boxes2)
        %timeit box_ious_py(boxes1, boxes2)
        101 µs ± 1.47 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        42.5 µs ± 298 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    for a 45x7:
        boxes1 = Boxes(random_boxes(100, scale=100.0).numpy(), 'tlbr').data
        boxes2 = Boxes(random_boxes(80, scale=100.0).numpy(), 'tlbr').data
        %timeit box_ious_py2(boxes1, boxes2)
        %timeit box_ious_py(boxes1, boxes2)
        %timeit bbox_ious_c(boxes1, boxes2)

        116 µs ± 962 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        49.2 µs ± 824 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

        from netharn import util
        _ = util.profile_onthefly(box_ious_py2)(boxes1, boxes2)
        _ = util.profile_onthefly(box_ious_py)(boxes1, boxes2)

    Benchmark:
        from netharn import util
        boxes1 = Boxes(random_boxes(45, scale=100.0).numpy(), 'tlbr').data
        boxes2 = Boxes(random_boxes(7, scale=100.0).numpy(), 'tlbr').data

        import ubelt as ub
        for timer in ub.Timerit(100, bestof=10, label='c'):
            bbox_ious_c(boxes1, boxes2)

        # from netharn.util.cython_boxes import bbox_ious_c_par
        # import ubelt as ub
        # for timer in ub.Timerit(100, bestof=10, label='c'):
        #     bbox_ious_c_par(boxes1, boxes2)

        for timer in ub.Timerit(100, bestof=10, label='py1'):
            box_ious_py1(boxes1, boxes2)

        for timer in ub.Timerit(100, bestof=10, label='py2'):
            box_ious_py2(boxes1, boxes2)

        for timer in ub.Timerit(100, bestof=10, label='py3'):
            box_ious_py3(boxes1, boxes2)

        boxes1_ = util.Boxes(boxes1, 'tlbr').to_cxywh().data
        boxes2_ = util.Boxes(boxes2, 'tlbr').to_cxywh().data

        for timer in ub.Timerit(100, bestof=10, label='py3'):
            bboxes_iou_light(boxes1_, boxes2_)
    """
    N = len(boxes1)
    K = len(boxes2)

    ax1, ay1, ax2, ay2 = (boxes1.T)
    bx1, by1, bx2, by2 = (boxes2.T)
    aw, ah = (ax2 - ax1 + 1), (ay2 - ay1 + 1)
    bw, bh = (bx2 - bx1 + 1), (by2 - by1 + 1)

    areas1 = aw * ah
    areas2 = bw * bh

    # Create all pairs of boxes
    ns = np.repeat(np.arange(N), K, axis=0)
    ks = np.repeat(np.arange(K)[None, :], N, axis=0).ravel()

    ex_ax1 = np.repeat(ax1, K, axis=0)
    ex_ay1 = np.repeat(ay1, K, axis=0)
    ex_ax2 = np.repeat(ax2, K, axis=0)
    ex_ay2 = np.repeat(ay2, K, axis=0)

    ex_bx1 = np.repeat(bx1[None, :], N, axis=0).ravel()
    ex_by1 = np.repeat(by1[None, :], N, axis=0).ravel()
    ex_bx2 = np.repeat(bx2[None, :], N, axis=0).ravel()
    ex_by2 = np.repeat(by2[None, :], N, axis=0).ravel()

    x_maxs = np.minimum(ex_ax2, ex_bx2)
    x_mins = np.maximum(ex_ax1, ex_bx1)

    iws = (x_maxs - x_mins + 1)

    # Remove pairs of boxes that don't intersect in the x dimension
    flags = iws > 0
    ex_ay1 = ex_ay1.compress(flags, axis=0)
    ex_ay2 = ex_ay2.compress(flags, axis=0)
    ex_by1 = ex_by1.compress(flags, axis=0)
    ex_by2 = ex_by2.compress(flags, axis=0)
    ns = ns.compress(flags, axis=0)
    ks = ks.compress(flags, axis=0)
    iws = iws.compress(flags, axis=0)

    y_maxs = np.minimum(ex_ay2, ex_by2)
    y_mins = np.maximum(ex_ay1, ex_by1)

    ihs = (y_maxs - y_mins + 1)

    # Remove pairs of boxes that don't intersect in the x dimension
    flags = ihs > 0
    ns = ns.compress(flags, axis=0)
    ks = ks.compress(flags, axis=0)
    iws = iws.compress(flags, axis=0)
    ihs = ihs.compress(flags, axis=0)

    areas_sum = areas1[ns] + areas2[ks]

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    expanded_ious = inter_areas / union_areas

    ious = np.zeros((N, K), dtype=np.float32)
    ious[ns, ks] = expanded_ious
    return ious


class Boxes(ub.NiceRepr):
    """
    Converts boxes between different formats as long as the last dimension
    contains 4 coordinates and the format is specified.

    Example:
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
            [torch.FloatTensor of size (1,4)]
            )>

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
                [torch.LongTensor of size (3,4)]
                )>
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
        """
        works with tlbr, cxywh, xywh, xy, or wh formats

        Example:
            >>> Boxes(np.array([1, 1, 10, 10])).scale(2).data
            array([ 2.,  2., 20., 20.])
            >>> Boxes(np.array([[1, 1, 10, 10]])).scale((2, .5)).data
            array([[ 2. ,  0.5, 20. ,  5. ]])
            >>> Boxes(scale_boxes(np.array([[10, 10]])).scale(.5).data
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
        w, h = self.to_xywh().components[-2:]
        return w * h

    @property
    def components(self):
        a = self.data[..., 0:1]
        b = self.data[..., 1:2]
        c = self.data[..., 2:3]
        d = self.data[..., 3:4]
        return [a, b, c, d]

    def toformat(self, format, copy=True):
        if format == 'xywh':
            return self.to_xywh(copy=copy)
        elif format == 'tlbr':
            return self.to_tlbr(copy=copy)
        elif format == 'cxywh':
            return self.to_cxywh(copy=copy)
        else:
            raise KeyError(self.format)

    @classmethod
    def _cat(cls, datas):
        if torch.is_tensor(datas[0]):
            return torch.cat(datas, dim=-1)
        else:
            return np.concatenate(datas, axis=-1)

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
            >>> self = Boxes([25, 30, 15, 10], 'tlbr')
            >>> bboi = self.to_imgaug()
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
            >>> orig = Boxes(random_boxes(5).numpy(), 'tlbr')
            >>> bboi = orig.to_imgaug(shape=(500, 500))
            >>> self = Boxes.from_imgaug(bboi)
            >>> assert np.all(self.data == orig.data)
        """
        tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                         for bb in bboi.bounding_boxes])
        tlbr = tlbr.reshape(-1, 4)
        return Boxes(tlbr, format='tlbr')

    def to_brambox(self):
        """
        Notes:
            pip install git+https://gitlab.com/EAVISE/brambox.git

        Example:
            >>> xywh = Boxes(random_boxes(5, scale=100.0).numpy(), 'xywh')
            >>> xywh.to_brambox()
        """
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))
        from brambox.boxes.box import Box
        xywh_boxes = self.to_xywh(copy=False).data
        boxes = []
        for x, y, w, h in xywh_boxes:
            box = Box()
            box.x_top_left = x
            box.y_top_left = y
            box.width = w
            box.height = h
            boxes.append(box)
        return boxes

    def clip(self, x_min, y_min, x_max, y_max, inplace=False):
        """
        Clip boxes to image boundaries.  If box is in tlbr format, inplace
        operation is an option.

        Example:
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


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries inplace.

    Args:
        boxes (ndarray): multiple boxes in tlbr format
        im_shape (tuple): (H, W) of original image

    Example:
        >>> boxes = np.array([[-10, -10, 120, 120], [1, -2, 30, 50]])
        >>> im_shape = (100, 110)  # H, W
        >>> clip_boxes(boxes, im_shape)
        array([[  0,   0, 109,  99],
               [  1,   0,  30,  50]])
    """
    if boxes.shape[0] == 0:
        return boxes

    if not isinstance(boxes, (np.ndarray, list)):
        raise TypeError('got boxes={!r}'.format(boxes))

    im_h, im_w = im_shape
    x1, y1, x2, y2 = boxes.T
    np.minimum(x1, im_w - 1, out=x1)  # x1 < im_shape[1]
    np.minimum(y1, im_h - 1, out=y1)  # y1 < im_shape[0]
    np.minimum(x2, im_w - 1, out=x2)  # x2 < im_shape[1]
    np.minimum(y2, im_h - 1, out=y2)  # y2 < im_shape[0]
    # y1 >= 0 and  x1 >= 0
    boxes = np.maximum(boxes, 0, out=boxes)
    return boxes


def random_boxes(num, box_format='tlbr', scale=100):
    if num:
        xywh = (torch.rand(num, 4) * scale)
        if isinstance(scale, int):
            xywh = xywh.long()
        if box_format == 'xywh':
            return xywh
        elif box_format == 'tlbr':
            return Boxes(xywh, 'xywh').to_tlbr().data
        else:
            raise KeyError(box_format)
    else:
        if isinstance(scale, int):
            return torch.LongTensor()
        else:
            return torch.FloatTensor()


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_boxes all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
