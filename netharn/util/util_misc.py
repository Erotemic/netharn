# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import ubelt as ub


class SupressPrint():
    """
    Temporarily replace the print function in a module with a noop

    Args:
        *mods: the modules to disable print in
        **kw: only accepts "enabled"
            enabled (bool, default=True): enables or disables this context
    """
    def __init__(self, *mods, **kw):
        enabled = kw.get('enabled', True)
        self.mods = mods
        self.enabled = enabled
        self.oldprints = {}

    def __enter__(self):
        if self.enabled:
            for mod in self.mods:
                oldprint = getattr(self.mods, 'print', print)
                self.oldprints[mod] = oldprint
                mod.print = lambda *args, **kw: None
    def __exit__(self, a, b, c):
        if self.enabled:
            for mod in self.mods:
                mod.print = self.oldprints[mod]


class FlatIndexer(ub.NiceRepr):
    """
    Creates a flat "view" of a jagged nested indexable object.
    Only supports one offset level.

    TODO:
        - [ ] Move to kwarray

    Args:
        lens (list): a list of the lengths of the nested objects.

    Doctest:
        >>> self = FlatIndexer([1, 2, 3])
        >>> len(self)
        >>> self.unravel(4)
        >>> self.ravel(2, 1)
    """
    def __init__(self, lens):
        self.lens = lens
        self.cums = np.cumsum(lens)

    @classmethod
    def fromlist(cls, items):
        lens = list(map(len, items))
        return cls(lens)

    def __len__(self):
        return self.cums[-1]

    def unravel(self, index):
        """
        Args:
            index : raveled index

        Returns:
            Tuple[int, int]: outer and inner indices
        """
        outer = np.where(self.cums > index)[0][0]
        base = self.cums[outer] - self.lens[outer]
        inner = index - base
        return (outer, inner)

    def ravel(self, outer, inner):
        """
        Args:
            outer: index into outer list
            inner: index into the list referenced by outer

        Returns:
            index: the raveled index
        """
        base = self.cums[outer] - self.lens[outer]
        return base + inner
