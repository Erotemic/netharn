import numpy as np
import cv2
from imgaug.parameters import (Uniform, Binomial)
from netharn.data.transforms import augmenter_base


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
    """ Perform random HSV shift on the RGB data.

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

    Notes:
        # Note the cv2 conversion to HSV does not go into the 0-1 range,
        # instead it goes into (0-360, 0-1, 0-1) for hue, sat, and val.

    Exmaple:
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
        >>> for i in range(1, len(pnums)):
        >>>     aug = self.augment_image(img)
        >>>     mplutil.imshow(aug, colorspace='rgb', pnum=pnums[i], title='aug: {}'.format(ub.repr2(self._prev_params, nl=0, precision=4)))
        >>> mplutil.show_if_requested()

    Ignore:
        random_state = self.random_state
        a1 = iaa.AddToHueAndSaturation((-20, 20))
        a1.augment_image(img)
        d1 = a1.to_deterministic()
        d1.augment_image(img)
        a1.to_deterministic().augment_image(img)

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

        use_floats = True

        if use_floats:
            img01 = img.astype(np.float32) / 255.0
            hsv = cv2.cvtColor(img01, cv2.COLOR_RGB2HSV)
            hue_bound = 360.0
            sat_bound = 1.0
            val_bound = 1.0
        else:
            hue_bound = 255
            sat_bound = 255
            val_bound = 255
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)

        def clip_hue(new_hue, hue_bound):
            """ This is about 10x faster than using modulus """
            out = new_hue
            out[out >= hue_bound] -= hue_bound
            out[out < 0] += hue_bound
            return out

        # add to hue
        hsv[:, :, 0] = clip_hue(hsv[:, :, 0] + (hue_bound * dh), hue_bound)
        # scale saturation and value
        hsv[:, :, 1] = np.clip(ds * hsv[:, :, 1], 0.0, sat_bound)
        hsv[:, :, 2] = np.clip(dv * hsv[:, :, 2], 0.0, val_bound)

        if use_floats:
            img01 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            img255 = (img01 * 255).astype(np.uint8)
        else:
            img255 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img255
