"""
Code for commonalities between "X for" objects that compute analytic properties
of networks like OutputShapeFor and ReceptiveFieldFor
"""
import ubelt as ub
from collections import OrderedDict


class Hidden(OrderedDict, ub.NiceRepr):
    """ Object for storing hidden states of analystic computation """

    def __nice__(self):
        return ub.repr2(self, nl=0)

    def __str__(self):
        return ub.NiceRepr.__str__(self)

    def __repr__(self):
        return ub.NiceRepr.__repr__(self)

    def __setitem__(self, key, value):
        if getattr(value, 'hidden', None) is not None:
            # When setting a value to an OutputShape object, if that object has
            # a hidden shape, then use that instead.
            value = value.hidden
        return OrderedDict.__setitem__(self, key, value)

    def shallow(self, n=1):
        """
        Grabs only the shallowest n layers of hidden shapes
        """
        if n == 0:
            last = self
            while hasattr(last, 'shallow'):
                values = list(last.values())
                if len(values):
                    last = values[-1]
                else:
                    break
            return last
        else:
            output = OrderedDict()
            for key, value in self.items():
                # if isinstance(value, HiddenShapes):
                if hasattr(value, 'shallow'):
                    value = value.shallow(n - 1)
                output[key] = value
            return output


class OutputFor(object):
    """
    Analytic base / identity class
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)


class Output(object):
    """
    Analytic base / identity class
    """
    @classmethod
    def coerce(cls, data=None, hidden=None):
        return data


class ForwardFor(OutputFor):
    """
    Analytic version of forward functions
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    @staticmethod
    def getitem(arr):
        """
        Wraps getitem calls

        Example:
            >>> import torch
            >>> arr = torch.rand(2, 16, 2, 2)
            >>> result = ForwardFor.getitem(arr)[:, 0:4]
            >>> assert result.shape == (2, 4, 2, 2)
        """
        return _ForwardGetItem(arr)

    @staticmethod
    def view(arr, *args):
        """
        Wraps view calls

        Example:
            >>> import torch
            >>> arr = torch.rand(2, 16, 2, 2)
            >>> result = ForwardFor.view(arr, -1)
        """
        return arr.view(*args)

    @staticmethod
    def shape(arr):
        """
        Wraps shape calls

        Example:
            >>> import torch
            >>> arr = torch.rand(2, 16, 2, 2)
            >>> result = ForwardFor.shape(arr)
        """
        return arr.shape

    @staticmethod
    def add(arr1, arr2):
        return arr1 + arr2

    @staticmethod
    def mul(arr1, arr2):
        return arr1 * arr2

    @staticmethod
    def sub(arr1, arr2):
        return arr1 - arr2

    @staticmethod
    def div(arr1, arr2):
        return arr1 - arr2


class _ForwardGetItem(object):
    def __init__(self, inp):
        self.inp = inp

    def __getitem__(self, slices):
        return self.inp.__getitem__(slices)
