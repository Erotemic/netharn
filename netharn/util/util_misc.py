# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


class SupressPrint():
    """
    Temporarily replace the print function in a module with a noop
    """
    def __init__(self, *mods):
        self.mods = mods
        self.oldprints = {}
    def __enter__(self):
        for mod in self.mods:
            oldprint = getattr(self.mods, 'print', print)
            self.oldprints[mod] = oldprint
            mod.print = lambda *args, **kw: None
    def __exit__(self, a, b, c):
        for mod in self.mods:
            mod.print = self.oldprints[mod]
