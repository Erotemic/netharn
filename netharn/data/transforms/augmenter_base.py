# -*- coding: utf-8 -*-
from collections import OrderedDict
import imgaug


class ParamatarizedAugmenter(imgaug.augmenters.Augmenter):
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
            elif isinstance(value, imgaug.parameters.StochasticParameter):
                self._registered_params[key] = value
        super().__setattr__(key, value)

    def _augment_heatmaps(self):
        raise NotImplemented
