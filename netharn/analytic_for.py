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
