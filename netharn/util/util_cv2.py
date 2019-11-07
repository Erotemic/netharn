# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import numpy as np  # NOQA


def draw_boxes_on_image(img, boxes, color='blue', thickness=1,
                        box_format=None, colorspace='bgr'):
    """
    Draws boxes on an image.

    Args:
        img (ndarray): image to copy and draw on
        boxes (nh.util.Boxes): boxes to draw
        colorspace (str): string code of the input image colorspace

    Example:
        >>> from netharn import util
        >>> img = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> color = 'dodgerblue'
        >>> thickness = 1
        >>> boxes = util.Boxes([[1, 1, 8, 8]], 'tlbr')
        >>> img2 = draw_boxes_on_image(img, boxes, color, thickness)
        >>> assert tuple(img2[1, 1]) == (255, 144, 30)
        >>> # xdoc: +REQUIRES(--show)
        >>> from netharn.util import mplutil
        >>> mplutil.autompl()  # xdoc: +SKIP
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> mplutil.imshow(img2)
    """
    from netharn import util
    if not isinstance(boxes, util.Boxes):
        if box_format is None:
            raise ValueError('specify box_format')
        boxes = util.Boxes(boxes, box_format)

    color = tuple(util.Color(color).as255(colorspace))
    tlbr = boxes.to_tlbr().data
    img2 = img.copy()
    for x1, y1, x2, y2 in tlbr:
        # pt1 = (int(round(x1)), int(round(y1)))
        # pt2 = (int(round(x2)), int(round(y2)))
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        img2 = cv2.rectangle(img2, pt1, pt2, color, thickness=thickness)
    return img2


def draw_text_on_image(img, text, org, **kwargs):
    """
    Draws multiline text on an image using opencv

    Args:
        img (ndarray): image to draw on
        text (str): text to draw
        org (tuple): x, y Bottom-left corner of the text string in the image
        **kwargs:
            color (tuple): default blue
            thickneess (int): defaults to 2
            fontFace (int): defaults to cv2.FONT_HERSHEY_SIMPLEX
            fontScale (float): defaults to 1.0

    References:
        https://stackoverflow.com/questions/27647424/

    Example:
        >>> import netharn as nh
        >>> img = nh.util.grab_test_image(space='bgr')
        >>> img2 = nh.util.draw_text_on_image(img, 'FOOBAR', org=(0, 100))
        >>> # xdoc: +REQUIRES(--show)
        >>> nh.util.autompl()
        >>> nh.util.imshow(img2, fontScale=10)
        >>> nh.util.show_if_requested()
    """
    if 'color' not in kwargs:
        kwargs['color'] = (255, 0, 0)

    if 'thickness' not in kwargs:
        kwargs['thickness'] = 2

    if 'fontFace' not in kwargs:
        kwargs['fontFace'] = cv2.FONT_HERSHEY_SIMPLEX

    if 'fontScale' not in kwargs:
        kwargs['fontScale'] = 1.0

    if 'lineType' not in kwargs:
        kwargs['lineType'] = cv2.LINE_AA

    getsize_kw = {
        k: kwargs[k]
        for k in ['fontFace', 'fontScale', 'thickness']
        if k in kwargs
    }
    x0, y0 = org
    ypad = kwargs.get('thickness', 2) + 4
    y = y0
    for i, line in enumerate(text.split('\n')):
        (w, h), text_sz = cv2.getTextSize(text, **getsize_kw)
        img = cv2.putText(img, line, (x0, y), **kwargs)
        y += (h + ypad)
    return img


def putMultiLineText(*args, **kw):
    # DEPRICATED
    import warnings
    warnings.warn('putMultiLineText is depricated. Use draw_text_on_image', DeprecationWarning)
    return draw_text_on_image(*args, **kw)
