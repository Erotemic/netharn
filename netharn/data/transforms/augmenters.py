import numpy as np
import cv2
from imgaug.parameters import (Uniform, Binomial)
from netharn.data.transforms import augmenter_base
from netharn.util import profiler  # NOQA
from netharn import util
import imgaug


def demodata_hsv_image(w=200, h=200):
    """
    Example:
        >>> rgb255 = demodata_hsv_image()
        >>> from netharn.util import mplutil
        >>> mplutil.autompl()
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> mplutil.imshow(rgb255, colorspace='rgb')
        >>> mplutil.show_if_requested()
    """

    hsv = np.zeros((h, w, 3), dtype=np.float32)

    hue = np.linspace(0, 360, num=w)
    hsv[:, :, 0] = hue[None, :]

    sat = np.linspace(0, 1, num=h)
    hsv[:, :, 1] = sat[:, None]

    val = np.linspace(0, 1, num=3)
    parts = []
    for v in val:
        p = hsv.copy()
        p[:, :, 2] = v
        parts.append(p)
    final_hsv = np.hstack(parts)
    rgb01 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    rgb255 = (rgb01 * 255).astype(np.uint8)
    return rgb255


class HSVShift(augmenter_base.ParamatarizedAugmenter):
    r"""
    Perform random HSV shift on the RGB data.

    MODIFIED FROM LIGHTNET YOLO into imgaug format

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue.
            The number is specified as a percentage of the available hue space
            (e.g. hue * 255 or hue * 360).
        saturation (Number): Random number between 1,saturation is used to
            shift the saturation; 50% chance to get 1/dSaturation instead of
            dSaturation
        value (Number): Random number between 1,value is used to shift the
            value; 50% chance to get 1/dValue in stead of dValue

    CommandLine:
        python -m netharn.data.transforms.augmenters HSVShift --show

    Example:
        >>> self = HSVShift(0.1, 1.5, 1.5)
        >>> img = demodata_hsv_image()
        >>> aug = self.augment_image(img)
        >>> det = self.to_deterministic()
        >>> assert np.all(det.augment_image(img) == det.augment_image(img))
        >>> # xdoc: +REQUIRES(--show)
        >>> from netharn.util import mplutil
        >>> import ubelt as ub
        >>> mplutil.autompl()
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> self = HSVShift(0.5, 1.5, 1.5)
        >>> pnums = mplutil.PlotNums(5, 5)
        >>> random_state = self.random_state
        >>> mplutil.imshow(img, colorspace='rgb', pnum=pnums[0], title='orig')
        >>> for i in range(1, len(pnums)):
        >>>     aug = self.augment_image(img)
        >>>     title = 'aug: {}'.format(ub.repr2(self._prev_params, nl=0, precision=3))
        >>>     mplutil.imshow(aug, colorspace='rgb', pnum=pnums[i], title=title)
        >>> mplutil.show_if_requested()
    """
    def __init__(self, hue, sat, val, input_colorspace='rgb'):
        super().__init__()
        self.input_colorspace = input_colorspace
        self.hue = Uniform(-hue, hue)
        self.sat = Uniform(1, sat)
        self.val = Uniform(1, val)

        self.flip_val = Binomial(.5)
        self.flip_sat = Binomial(.5)

    def _augment_images(self, images, random_state, parents, hooks):
        return [self.forward(img, random_state) for img in images]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    @profiler.profile
    def forward(self, img, random_state=None):
        assert self.input_colorspace == 'rgb'
        assert img.dtype.kind == 'u' and img.dtype.itemsize == 1

        dh = self.hue.draw_sample(random_state)
        ds = self.sat.draw_sample(random_state)
        dv = self.val.draw_sample(random_state)

        if self.flip_sat.draw_sample(random_state):
            ds = 1.0 / ds
        if self.flip_val.draw_sample(random_state):
            dv = 1.0 / dv

        self._prev_params = (dh, ds, dv)

        # Note the cv2 conversion to HSV does not go into the 0-1 range,
        # instead it goes into (0-360, 0-1, 0-1) for hue, sat, and val.
        img01 = img.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img01, cv2.COLOR_RGB2HSV)

        hue_bound = 360.0
        sat_bound = 1.0
        val_bound = 1.0

        def wrap_hue(new_hue, hue_bound):
            """ This is about 10x faster than using modulus """
            out = new_hue
            out[out >= hue_bound] -= hue_bound
            out[out < 0] += hue_bound
            return out

        # add to hue
        hsv[:, :, 0] = wrap_hue(hsv[:, :, 0] + (hue_bound * dh), hue_bound)
        # scale saturation and value
        hsv[:, :, 1] = np.clip(ds * hsv[:, :, 1], 0.0, sat_bound)
        hsv[:, :, 2] = np.clip(dv * hsv[:, :, 2], 0.0, val_bound)

        img01 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        img255 = (img01 * 255).astype(np.uint8)
        return img255


class Resize(augmenter_base.ParamatarizedAugmenter):
    """
    Transform images and annotations to the right network dimensions.

    Args:
        target_size (tuple): Scale images to this size (w, h) keeping the
        aspect ratio using a letterbox.

    Example:
        >>> img = demodata_hsv_image()
        >>> box = util.Boxes([[.45, .05, .10, .05], [0., 0.0, .199, .199], [.24, .05, .01, .05]], format='tlwh').to_tlbr()
        >>> bboi = box.to_imgaug(shape=img.shape)
        >>> self = Resize((40, 30))
        >>> aug1  = self.augment_image(img)
        >>> bboi1 = self.augment_bounding_boxes([bboi])[0]

    Example:
        >>> img = demodata_hsv_image()
        >>> box = util.Boxes([[450, 50, 100, 50], [0.0, 0, 199, 199], [240, 50, 10, 50]], format='tlwh').to_tlbr()
        >>> bboi = box.to_imgaug(shape=img.shape)
        >>> imgT = np.ascontiguousarray(img.transpose(1, 0, 2))
        >>> bboiT = box.transpose().to_imgaug(shape=imgT.shape)
        >>> self = Resize((40, 30))
        >>> self2 = Resize((1000, 1000))
        >>> # ---------------------------
        >>> aug1  = self.augment_image(img)
        >>> bboi1 = self.augment_bounding_boxes([bboi])[0]
        >>> aug2  = self.augment_image(imgT)
        >>> bboi2 = self.augment_bounding_boxes([bboiT])[0]
        >>> aug3  = self2.augment_image(img)
        >>> bboi3 = self2.augment_bounding_boxes([bboi])[0]
        >>> aug4  = self2.augment_image(imgT)
        >>> bboi4 = self2.augment_bounding_boxes([bboiT])[0]
        >>> # ---------------------------
        >>> # xdoc: +REQUIRES(--show)
        >>> from netharn import util
        >>> from netharn.util import mplutil
        >>> mplutil.autompl()
        >>> #mplutil.figure(doclf=True, fnum=1)
        >>> #pnum_ = mplutil.PlotNums(3, 2)
        >>> #mplutil.imshow(util.draw_boxes_on_image(img, util.Boxes.from_imgaug(bboi)), pnum=pnum_(), title='orig')
        >>> #mplutil.imshow(util.draw_boxes_on_image(imgT, util.Boxes.from_imgaug(bboiT)), pnum=pnum_(), title='origT')
        >>> #mplutil.imshow(util.draw_boxes_on_image(aug1, util.Boxes.from_imgaug(bboi1)), pnum=pnum_())
        >>> #mplutil.imshow(util.draw_boxes_on_image(aug2, util.Boxes.from_imgaug(bboi2)), pnum=pnum_())
        >>> #mplutil.imshow(util.draw_boxes_on_image(aug3, util.Boxes.from_imgaug(bboi3)), pnum=pnum_())
        >>> #mplutil.imshow(util.draw_boxes_on_image(aug4, util.Boxes.from_imgaug(bboi4)), pnum=pnum_())
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> pnum_ = mplutil.PlotNums(3, 2)
        >>> mplutil.imshow(img, colorspace='rgb', pnum=pnum_(), title='orig')
        >>> mplutil.draw_boxes(util.Boxes.from_imgaug(bboi))
        >>> mplutil.imshow(imgT, colorspace='rgb', pnum=pnum_(), title='origT')
        >>> mplutil.draw_boxes(util.Boxes.from_imgaug(bboiT))
        >>> # ----
        >>> mplutil.imshow(aug1, colorspace='rgb', pnum=pnum_())
        >>> #mplutil.draw_boxes(util.Boxes.from_imgaug(bboi1))
        >>> x = util.Boxes.from_imgaug(bboi1).shift((-0.5, -0.5))
        >>> mplutil.draw_boxes(x)
        >>> mplutil.imshow(aug2, colorspace='rgb', pnum=pnum_())
        >>> x = util.Boxes.from_imgaug(bboi2).shift((-0.5, -0.5))
        >>> mplutil.draw_boxes(x)
        >>> #mplutil.draw_boxes(util.Boxes.from_imgaug(bboi2))
        >>> # ----
        >>> mplutil.imshow(aug3, colorspace='rgb', pnum=pnum_())
        >>> mplutil.draw_boxes(util.Boxes.from_imgaug(bboi3))
        >>> mplutil.imshow(aug4, colorspace='rgb', pnum=pnum_())
        >>> mplutil.draw_boxes(util.Boxes.from_imgaug(bboi4))

    Ignore:
        image = img
        target_size = np.array(self.target_size)
        orig_size = np.array(img.shape[0:2][::-1])
        shift, scale, embed_size = self._letterbox_transform(orig_size,
                                                             target_size)
    """
    def __init__(self, target_size, fill_color=127, mode='letterbox'):
        super().__init__()
        self.target_size = None if target_size is None else np.array(target_size)
        self.fill_color = fill_color
        self.mode = mode
        assert self.mode == 'letterbox', 'thats all folks'

    def forward(self, img, random_state=None):
        orig_size = np.array(img.shape[0:2][::-1])
        assert self.mode == 'letterbox', 'thats all folks'
        shift, scale, embed_size = self._letterbox_transform(orig_size,
                                                             self.target_size)
        new_img = self._img_letterbox_apply(img, embed_size, shift,
                                            self.target_size)
        return new_img

    def _augment_images(self, images, random_state, parents, hooks):
        self.target_size = None if self.target_size is None else np.array(self.target_size)
        return [self.forward(img, random_state) for img in images]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        """
        Example:
            >>> import imgaug
            >>> tlbr = [[0, 0, 10, 10], [1, 2, 8, 9]]
            >>> shape = (20, 40, 3)
            >>> bboi = util.Boxes(tlbr, 'tlbr').to_imgaug(shape)
            >>> bounding_boxes_on_images = [bboi]
            >>> kps_ois = []
            >>> for bbs_oi in bounding_boxes_on_images:
            >>>     kps = []
            >>>     for bb in bbs_oi.bounding_boxes:
            >>>         kps.extend(bb.to_keypoints())
            >>>     kps_ois.append(imgaug.KeypointsOnImage(kps, shape=bbs_oi.shape))
            >>> keypoints_on_images = kps_ois
            >>> self = LetterboxResize((400, 400))
        """
        result = []
        target_size = np.array(self.target_size)
        target_shape = target_size[::-1]
        prev_size = None
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            orig_size = (keypoints_on_image.width, keypoints_on_image.height)

            if prev_size != orig_size:
                # Cache previously computed values
                shift, scale, embed_size = self._letterbox_transform(
                    orig_size, target_size)
                prev_size = orig_size

            xy = keypoints_on_image.get_coords_array()
            xy_aug = (xy * scale) + shift

            new_keypoint = imgaug.KeypointsOnImage.from_coords_array(
                xy_aug, shape=target_shape)
            result.append(new_keypoint)
        return result

    def _img_letterbox_invert(self, img, orig_size, target_size):
        """
        Args:
            img : the image to scale back up
            orig_size : original wh of the image
            target_size : network input wh

        Example:
            >>> orig_img = demodata_hsv_image(w=100, h=200)
            >>> orig_size = np.array(orig_img.shape[0:2][::-1])
            >>> target_size = (416, 416)
            >>> self = Resize(target_size)
            >>> img = self.forward(orig_img)
            >>> inverted_img = self._img_letterbox_invert(img, orig_size, target_size)
            >>> assert inverted_img.shape == orig_img.shape
            >>> # xdoc: +REQUIRES(--show)
            >>> util.imshow(orig_img, fnum=1, pnum=(1, 3, 1))
            >>> util.imshow(img, fnum=1, pnum=(1, 3, 2))
            >>> util.imshow(inverted_img, fnum=1, pnum=(1, 3, 3))
        """
        shift, scale, embed_size = self._letterbox_transform(orig_size, target_size)
        top, bot, left, right = self._padding(embed_size, shift, target_size)

        # Undo padding
        h, w = img.shape[0:2]
        unpadded_img = img[top:h - bot, left:w - right]

        sf = orig_size / embed_size
        dsize = tuple(orig_size)
        # Choose INTER_AREA if we are shrinking the image
        interpolation = cv2.INTER_AREA if sf.sum() < 2 else cv2.INTER_CUBIC
        inverted_img = cv2.resize(unpadded_img, dsize, interpolation=interpolation)
        return inverted_img

    def _boxes_letterbox_apply(self, boxes, orig_size, target_size):
        """
        Apply the letterbox transform to these bounding boxes

        """
        shift, scale, embed_size = self._letterbox_transform(orig_size, target_size)
        new_boxes = boxes.scale(scale).translate(shift)
        return new_boxes

    def _boxes_letterbox_invert(self, boxes, orig_size, target_size):
        """
        Undo the letterbox transform for these bounding boxes. Moves
        the box from `target_size` coordinatse (which are probably square)
        to `orig_size` coordinates (which are probably nto square).

        Args:
            boxes (util.Boxes) : boxes to rework in `target_size` coordinates
            orig_size : original wh of the image
            target_size : network input wh (i.e. inp_size)

        Example:
            >>> target_size = (416, 416)
            >>> orig_size = (1000, 400)
            >>> cxywh_norm = util.Boxes(np.array([[.5, .5, .2, .2]]), 'cxywh')
            >>> self = Resize(target_size)
            >>> cxywh = self._boxes_letterbox_invert(cxywh_norm, orig_size, target_size)
            >>> cxywh_norm2 = self._boxes_letterbox_apply(cxywh, orig_size, target_size)
            >>> assert np.allclose(cxywh_norm2.data, cxywh_norm.data)
        """
        shift, scale, embed_size = self._letterbox_transform(orig_size, target_size)
        new_boxes = boxes.translate(-shift).scale(1.0 / scale)
        return new_boxes

    def _letterbox_transform(self, orig_size, target_size):
        """
        aspect ratio preserving scaling + extra padding to equal target size

        scale should be applied before the shift.

        Args:
            orig_size : original wh of the image
            target_size : network input wh

        Example:
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Resize(None)._letterbox_transform([5, 10], [10, 10])
            (array([2, 0]), array([1., 1.]), array([ 5, 10]))
            >>> Resize(None)._letterbox_transform([10, 5], [10, 10])
            (array([0, 2]), array([1., 1.]), array([10,  5]))
        """
        # determine if width or the height should be used as the scale factor.
        orig_size = np.array(orig_size)
        target_size = np.array(target_size)
        fw, fh = orig_size / target_size
        sf = 1 / fw if fw >= fh else 1 / fh

        # Whats the closest integer size we can resize to?
        embed_size = np.round(orig_size * sf).astype(np.int)
        # Determine how much padding we need for the top/left side
        # Note: the right/bottom side might need an extra pixel of padding
        # depending on rounding issues.
        shift = np.round((target_size - embed_size) / 2).astype(np.int)

        scale = embed_size / orig_size
        return shift, scale, embed_size

    @staticmethod
    def _padding(embed_size, shift, target_size):
        pad_lefttop = shift
        pad_rightbot = target_size - (embed_size + shift)

        left, top = pad_lefttop
        right, bot = pad_rightbot
        return top, bot, left, right

    def _img_letterbox_apply(self, img, embed_size, shift, target_size):
        top, bot, left, right = self._padding(embed_size, shift, target_size)

        orig_size = np.array(img.shape[0:2][::-1])
        channels = util.get_num_channels(img)

        sf = embed_size / orig_size
        dsize = tuple(embed_size)
        # Choose INTER_AREA if we are shrinking the image
        interpolation = cv2.INTER_AREA if sf.sum() < 2 else cv2.INTER_CUBIC
        scaled = cv2.resize(img, dsize, interpolation=interpolation)

        fill_color = self.fill_color
        hwc255 = cv2.copyMakeBorder(scaled, top, bot, left, right,
                                    cv2.BORDER_CONSTANT,
                                    value=(fill_color,) * channels)
        return hwc255


LetterboxResize = Resize

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.data.transforms.augmenters all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
