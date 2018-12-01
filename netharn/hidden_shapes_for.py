# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch  # NOQA
import torchvision  # NOQA
import math
import torch.nn as nn
import ubelt as ub
# from netharn import util
from netharn.output_shape_for import OutputShapeFor
try:
    from netharn.device import DataSerial
except ImportError:
    DataSerial = None

REGISTERED_HIDDEN_SHAPES_TYPES = []


SHAPE_CLS = tuple  # We exepct shapes to be specified as this class


def compute_type(type):
    def _wrap(func):
        if type is not None:
            REGISTERED_HIDDEN_SHAPES_TYPES.append((type, func))
        return func
    return _wrap


class HiddenShapesFor(object):
    """
    Knows how to compute the hidden activation state for a few modules.
    Otherwise it defaults to OutputShapeFor

    Returns:
        Tuple[shapes, shape]:
            shapes: Dict: The hidden shape (can be complex due to nesting)
            shape: Tuple[int, ...]: (always a simple fixed-depth type)

    CommandLine:
        xdoctest -m ~/code/netharn/netharn/hidden_shapes_for.py HiddenShapesFor

    Example:
        >>> # Case where we have a registered func
        >>> from netharn.hidden_shapes_for import *
        >>> self = nn.Sequential(
        >>>     nn.Conv2d(2, 3, kernel_size=3),
        >>>     nn.Conv2d(3, 5, kernel_size=3),
        >>> )
        >>> shapes, shape = HiddenShapesFor(self)([1, 1, 7, 11])
        >>> print('shapes = {}'.format(ub.repr2(shapes, nl=1)))
        >>> print('shape = {}'.format(ub.repr2(shape, nl=0)))
        shapes = {
            '0': (1, 3, 5, 9),
            '1': (1, 5, 3, 7),
        }
        shape = (1, 5, 3, 7)

    Example:
        >>> # Case where we haven't registered a func
        >>> # In this case shapes is not populated (but shape is)
        >>> from netharn.hidden_shapes_for import *
        >>> self = nn.Conv2d(2, 3, kernel_size=3)
        >>> shapes, shape = HiddenShapesFor(self)([1, 1, 7, 11])
        >>> print('shapes = {}'.format(ub.repr2(shapes, nl=0)))
        >>> print('shape = {}'.format(shape))
        shapes = (1, 3, 5, 9)
        shape = (1, 3, 5, 9)

    """
    math = math  # for hacking in sympy

    def __init__(self, module):
        self.module = module
        self._func = getattr(module, 'hidden_shapes_for', None)
        if self._func is None:
            # Lookup shape func if we can't find it
            for type, _func in REGISTERED_HIDDEN_SHAPES_TYPES:
                try:
                    if module is type or isinstance(module, type):
                        self._func = _func
                except TypeError:
                    pass
            if not self._func:
                try:
                    OutputShapeFor(module)
                except TypeError:
                    raise TypeError('Unknown (hidden) module type {}'.format(module))

    def __call__(self, *args, **kwargs):
        if self._func:
            if isinstance(self.module, nn.Module):
                # bound methods dont need module
                is_bound  = hasattr(self._func, '__func__') and getattr(self._func, '__func__', None) is not None
                is_bound |= hasattr(self._func, 'im_func') and getattr(self._func, 'im_func', None) is not None
                if is_bound:
                    shapes, shape = self._func(*args, **kwargs)
                else:
                    # nn.Module with state
                    shapes, shape = self._func(self.module, *args, **kwargs)
            else:
                # a simple pytorch func
                shapes, shape = self._func(*args, **kwargs)
        else:
            shape = OutputShapeFor(self.module)(*args, **kwargs)
            shapes = shape
        return shapes, shape

    @staticmethod
    @compute_type(nn.Sequential)
    def sequential(module, input_shape):
        """
        CommandLine:
            xdoctest -m ~/code/netharn/netharn/hidden_shapes_for.py HiddenShapesFor.sequential

        Example:
            >>> self = nn.Sequential(
            >>>     nn.Conv2d(2, 3, kernel_size=3),
            >>>     nn.Conv2d(3, 5, kernel_size=3),
            >>>     nn.Conv2d(5, 7, kernel_size=3),
            >>> )
            >>> shapes, shape = HiddenShapesFor(self)([1, 1, 7, 11])
            >>> print('shape = {}'.format(ub.repr2(shape, nl=0)))
            >>> print('shapes = {}'.format(ub.repr2(shapes, nl=1)))
            shape = (1, 7, 1, 5)
            shapes = {
                '0': (1, 3, 5, 9),
                '1': (1, 5, 3, 7),
                '2': (1, 7, 1, 5),
            }
        """
        shape = input_shape
        shapes = ub.odict()
        for key, child in module._modules.items():
            # shapes, shape = HiddenShapesFor(child)(shape)
            if hasattr(child, 'hidden_shapes_for'):
                shapes[key], shape = child.hidden_shapes_for(shape)
            else:
                shapes[key], shape = HiddenShapesFor(child)(shape)
        return shapes, shape


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.hidden_shapes_for all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
