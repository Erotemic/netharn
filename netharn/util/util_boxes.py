import numpy as np
import torch
import ubelt as ub


def scale_boxes(boxes, factor):
    """
    works with tlbr, cxywh, xywh, xy, or wh formats

    Example:
        >>> scale_boxes(np.array([1, 1, 10, 10]), 2)
        array([ 2.,  2., 20., 20.])
        >>> scale_boxes(np.array([[1, 1, 10, 10]]), (2, .5))
        array([[ 2. ,  0.5, 20. ,  5. ]])
        >>> scale_boxes(np.array([[10, 10]]), .5)
        array([[5., 5.]])
    """
    sx, sy = factor if ub.iterable(factor) else (factor, factor)
    if torch.is_tensor(boxes):
        boxes = boxes.float().clone()
    else:
        boxes = boxes.astype(np.float).copy()
    boxes[..., 0:4:2] *= sx
    boxes[..., 1:4:2] *= sy
    return boxes


class Boxes(ub.NiceRepr):
    """
    Converts boxes between different formats as long as the last dimension
    contains 4 coordinates and the format is specified.

    Example:
        >>> Boxes([25, 30, 15, 10], 'xywh')
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').as_xywh()
        <Boxes(xywh, array([25, 30, 15, 10]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').as_cxywh()
        <Boxes(cxywh, array([32.5, 35. , 15. , 10. ]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').as_tlbr()
        <Boxes(tlbr, array([25, 30, 40, 40]))>
        >>> Boxes([25, 30, 15, 10], 'xywh').scale(2).as_tlbr()
        <Boxes(tlbr, array([50., 60., 80., 80.]))>
        >>> Boxes(torch.FloatTensor([[25, 30, 15, 20]]), 'xywh').scale(.1).as_tlbr()
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
        >>>             box2 = box1.asformat(format2)
        >>>             back = box2.asformat(format1)
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

    def copy(self):
        new_data = self.data.copy()
        return Boxes(new_data, self.format)

    def scale(self, factor):
        boxes = self.data
        sx, sy = factor if ub.iterable(factor) else (factor, factor)
        if torch.is_tensor(boxes):
            new_data = boxes.float().clone()
        else:
            new_data = boxes.astype(np.float).copy()
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
        w, h = self.as_xywh().components[-2:]
        return w * h

    @property
    def components(self):
        a = self.data[..., 0:1]
        b = self.data[..., 1:2]
        c = self.data[..., 2:3]
        d = self.data[..., 3:4]
        return [a, b, c, d]

    def asformat(self, format):
        if format == 'xywh':
            return self.as_xywh()
        elif format == 'tlbr':
            return self.as_tlbr()
        elif format == 'cxywh':
            return self.as_cxywh()
        else:
            raise KeyError(self.format)

    @classmethod
    def _cat(cls, datas):
        if torch.is_tensor(datas[0]):
            return torch.cat(datas, dim=-1)
        else:
            return np.concatenate(datas, axis=-1)

    def as_xywh(self, copy=True):
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

    def as_cxywh(self, copy=True):
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

    def as_tlbr(self, copy=True):
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

    def as_imgaug(self, shape):
        """
        Args:
            shape (tuple): shape of image that boxes belong to

        Example:
            >>> self = Boxes([25, 30, 15, 10], 'tlbr')
            >>> bboi = self.as_imgaug()
        """
        import imgaug
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))

        tlbr = self.as_tlbr(copy=False).data
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
            >>> bboi = orig.as_imgaug(shape=(500, 500))
            >>> self = Boxes.from_imgaug(bboi)
            >>> assert np.all(self.data == orig.data)
        """
        tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                         for bb in bboi.bounding_boxes])
        tlbr = tlbr.reshape(-1, 4)
        return Boxes(tlbr, format='tlbr')

    def as_brambox(self):
        """
        Notes:
            pip install git+https://gitlab.com/EAVISE/brambox.git

        Example:
            >>> xywh = Boxes(random_boxes(5).numpy(), 'xywh')
            >>> xywh.as_brambox()
        """
        if len(self.data.shape) != 2:
            raise ValueError('data must be 2d got {}d'.format(len(self.data.shape)))
        from brambox.boxes.box import Box
        xywh_boxes = self.as_xywh(copy=False).data
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
            self2 = self.as_tlbr(copy=True)
        x1, y1, x2, y2 = self2.data.T
        np.clip(x1, x_min, x_max, out=x1)
        np.clip(y1, y_min, y_max, out=y1)
        np.clip(x2, x_min, x_max, out=x2)
        np.clip(y2, y_min, y_max, out=y2)
        return self2

    def transpose(self):
        x, y, w, h = self.as_xywh().components
        self2 = self.__class__(self._cat([y, x, h, w]), format='xywh')
        self2 = self2.asformat(self.format)
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
            return Boxes(xywh, 'xywh').as_tlbr().data
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
