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
        new_data = scale_boxes(self.data, factor)
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

    def _cat(self, datas):
        if torch.is_tensor(datas[0]):
            return torch.cat(datas, dim=-1)
        else:
            return np.concatenate(datas, axis=-1)

    def as_xywh(self):
        if self.format == 'xywh':
            return self.copy()
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

    def as_cxywh(self):
        if self.format == 'cxywh':
            return self.copy()
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

    def as_tlbr(self):
        if self.format == 'tlbr':
            return self.copy()
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
        tlbr = self._cat([x1, y1, x2, y2])
        return Boxes(tlbr, 'tlbr')


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
    np.minimum(x1, im_w - 1, out=x1)  # x1 >= 0
    np.minimum(y1, im_h - 1, out=y1)  # y1 >= 0
    np.minimum(x2, im_w - 1, out=x2)  # x2 < im_shape[1]
    np.minimum(y2, im_h - 1, out=y2)  # y2 < im_shape[0]
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
