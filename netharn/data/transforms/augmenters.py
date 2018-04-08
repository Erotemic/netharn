import numpy as np
import cv2
from imgaug.parameters import (Uniform, Binomial)
from netharn.data.transforms import augmenter_base
from netharn.util import profiler  # NOQA
from netharn import util
import imgaug


def demodata_hsv_image():
    """
    Example:
        >>> rgb255 = demodata_hsv_image()
        >>> from netharn.util import mplutil
        >>> mplutil.qtensure()  # xdoc: +SKIP
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> mplutil.imshow(rgb255, colorspace='rgb')
        >>> mplutil.show_if_requested()
    """

    w, h = 200, 200
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

    Example:
        >>> self = HSVShift(0.1, 1.5, 1.5)
        >>> img = demodata_hsv_image()
        >>> aug = self.augment_image(img)
        >>> det = self.to_deterministic()
        >>> assert np.all(det.augment_image(img) == det.augment_image(img))
        >>> # xdoc: +REQUIRES(--show)
        >>> from netharn.util import mplutil
        >>> mplutil.qtensure()  # xdoc: +SKIP
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> self = HSVShift(0.1, 1.5, 1.5)
        >>> pnums = mplutil.PlotNums(8, 8)
        >>> random_state = self.random_state
        >>> mplutil.imshow(img, colorspace='rgb', pnum=pnums[0], title='orig')
        >>> import ubelt as ub
        >>> for i in range(1, len(pnums)):
        >>>     aug = self.augment_image(img)
        >>>     mplutil.imshow(aug, colorspace='rgb', pnum=pnums[i], title='aug: {}'.format(ub.repr2(self._prev_params, nl=0, precision=4)))
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
        result = []
        nb_images = len(images)
        for i in range(nb_images):
            new_img = self.apply(images[i], random_state)
            result.append(new_img)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    @profiler.profile
    def apply(self, img, random_state):
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


class LetterboxResize(augmenter_base.ParamatarizedAugmenter):
    """
    Transform images and annotations to the right network dimensions.

    Args:
        target_size (tuple): Scale images to this size (w, h) keeping the
        aspect ratio using a letterbox.
    """
    def __init__(self, target_size, fill_color=127):
        super().__init__()
        self.target_size = np.array(target_size)
        self.fill_color = fill_color

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        for i in range(nb_images):
            img = images[i]
            orig_size = img.shape[0:2][::-1]
            shift, embed_size = self._letterbox_transform(
                orig_size, self.target_size)
            new_img = self._apply_img_letterbox(img, embed_size, shift,
                                                self.target_size)
            result.append(new_img)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        """

        Example:
            >>> import imgaug
            >>> tlbr = [[0, 0, 10, 10], [1, 2, 8, 9]]
            >>> shape = (20, 40, 3)
            >>> bbs = imgaug.BoundingBoxesOnImage(
            >>>     [imgaug.BoundingBox(x1, y1, x2, y2)
            >>>      for x1, y1, x2, y2 in tlbr], shape=shape)
            >>> bounding_boxes_on_images = [bbs]
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
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            height, width = keypoints_on_image.height, keypoints_on_image.width

            orig_size = np.array((width, height))
            shift, embed_size = self._letterbox_transform(
                orig_size, self.target_size)
            scale = embed_size / orig_size

            coords = keypoints_on_image.get_coords_array()
            coords_aug = (coords * scale) + shift

            result.append(imgaug.KeypointsOnImage.from_coords_array(
                coords_aug, shape=keypoints_on_image.shape))
        return result

    def _letterbox_transform(self, orig_size, target_size):
        """
        aspect ratio preserving scaling + extra padding to equal target size
        """
        ow, oh = orig_size
        tw, th = target_size

        factor = orig_size / target_size
        fw, fh = factor
        sf = 1 / fw if fw >= fh else 1 / fh

        embed_size = np.round(orig_size * sf).astype(np.int)
        shift = np.round((target_size - embed_size) / 2).astype(np.int)
        return shift, embed_size

    def _apply_img_letterbox(self, image, embed_size, shift, target_size):
        """
        Example:
            >>> #image = np.ones((100, 50), dtype=np.uint8) * 255
            >>> image = np.ones((100, 100), dtype=np.uint8) * 255
            >>> target_size = np.array((100, 120))
            >>> orig_size = np.array(image.shape[0:2][::-1])
            >>> shift, embed_size = letterbox_transform(orig_size, target_size)
            >>> hwc255 = _apply_img_letterbox(image, embed_size, shift, target_size)
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc255, colorspace='gray')
            >>> mplutil.show_if_requested()
        """

        pad_lefttop = shift
        pad_rightbot = target_size - (embed_size + shift)

        left, top = pad_lefttop
        right, bot = pad_rightbot

        fill_color = 127
        channels = util.get_num_channels(image)

        # NO DONT SQUISH: squish the image into network input coordinates
        # USE LETTERBOX TO MAINTAIN ASPECT RATIO
        interpolation = cv2.INTER_AREA
        scaled = cv2.resize(image, tuple(embed_size),
                            interpolation=interpolation)

        hwc255 = cv2.copyMakeBorder(scaled, top, bot, left, right,
                                    cv2.BORDER_CONSTANT,
                                    value=(fill_color,) * channels)
        return hwc255

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.data.transforms.augmenters all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
