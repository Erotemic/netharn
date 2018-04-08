import cv2


def draw_boxes_on_image(img, boxes, color='blue', thickness=1,
                        box_format=None):
    """

    Example:
        >>> from netharn import util
        >>> img = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> color = 'blue'
        >>> thickness = 1
        >>> boxes = util.Boxes([[1, 1, 8, 8]], 'tlbr')
        >>> img2 = draw_boxes_on_image(img, boxes, color, thickness)
        >>> # xdoc: +REQUIRES(--show)
        >>> from netharn.util import mplutil
        >>> mplutil.qtensure()  # xdoc: +SKIP
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> mplutil.imshow(img2)
    """
    from netharn import util
    if not isinstance(boxes, util.Boxes):
        if box_format is None:
            raise ValueError('specify box_format')
        boxes = util.Boxes(boxes, box_format)

    color = tuple(util.Color(color).as255('bgr'))
    tlbr = boxes.as_tlbr().data
    img2 = img.copy()
    for x1, y1, x2, y2 in tlbr:
        # pt1 = (int(round(x1)), int(round(y1)))
        # pt2 = (int(round(x2)), int(round(y2)))
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        img2 = cv2.rectangle(img2, pt1, pt2, color, thickness=thickness)
    return img2
