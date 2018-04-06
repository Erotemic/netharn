from collections import OrderedDict
import numpy as np
import imgaug.augmenters as iaa
from imgaug.parameters import (StochasticParameter, Uniform, Binomial)
import cv2


class Augmenter2(iaa.Augmenter):
    """
    Helper that automatically registers stochastic parameters
    """

    def __init__(self, *args, **kwargs):
        super().__setattr__('_initialized', True)
        super().__setattr__('_registered_params', OrderedDict())
        super().__init__(*args, **kwargs)

    def _setparam(self, name, value):
        self._registered_params[name] = value
        setattr(self, name, value)

    def get_parameters(self):
        return list(self._registered_params.values())

    def __setattr__(self, key, value):
        if not getattr(self, '_initialized', False) and key != '_initialized':
            raise Exception(
                ('Must call super().__init__ in {} that inherits '
                 'from Augmenter2').format(self.__class__))
        if not key.startswith('_'):
            if key in self._registered_params:
                self._registered_params[key] = value
            elif isinstance(value, StochasticParameter):
                self._registered_params[key] = value
        super().__setattr__(key, value)


class HSVShift(Augmenter2):
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

    Exmaple:
        >>> self = HSVShift(0.1, 1.5, 1.5)
        >>> # start with a red image
        >>> img = (np.zeros((500, 500, 3)) + [[255, 0, 0]]).astype(np.uint8)
        >>> aug = self.augment_image(img)
        >>> det = self.to_deterministic()
        >>> assert np.all(det.augment_image(img) == det.augment_image(img))
        >>> # xdoc: +REQUIRES(--show)
        >>> from netharn.util import mplutil
        >>> mplutil.qtensure()  # xdoc: +SKIP
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> mplutil.imshow(aug, colorspace='rgb')
        >>> mplutil.show_if_requested()

    Ignore:
        random_state = self.random_state

        a1 = iaa.AddToHueAndSaturation((-20, 20))
        a1.augment_image(img)
        d1 = a1.to_deterministic()
        d1.augment_image(img)
        a1.to_deterministic().augment_image(img)

    """
    def __init__(self, hue, saturation, value, input_colorspace='rgb'):
        super().__init__()
        self.input_colorspace = input_colorspace
        self.hue = Uniform(-hue, hue)
        self.sat = Uniform(1, saturation)
        self.val = Uniform(1, value)

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

        # if img.dtype.kind == 'f':
        #     hue_bound = 360.0
        # elif img.dtype == np.uint8:
        hue_bound = 255
        sat_bound = 255
        val_bound = 255

        dh = self.hue.draw_sample(random_state) * hue_bound
        ds = self.sat.draw_sample(random_state)
        dv = self.val.draw_sample(random_state)

        if self.flip_sat.draw_sample(random_state):
            ds = 1.0 / ds
        if self.flip_val.draw_sample(random_state):
            dv = 1.0 / dv

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # add to hue and scale saturation and value
        hsv[:, :, 0] = ((hsv[:, :, 0] + dh) % hue_bound).astype(hsv.dtype)
        hsv[:, :, 1] = np.clip(ds * hsv[:, :, 1], 0, sat_bound).astype(hsv.dtype)
        hsv[:, :, 2] = np.clip(dv * hsv[:, :, 2], 0, val_bound).astype(hsv.dtype)

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img
