# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import pandas as pd
import copy


__version__ = '0.0.1'


class LocLight(object):
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, index):
        return self.parent._getrow(index)


class DataFrameLight(ub.NiceRepr):
    """
    Implements a subset of the pandas.DataFrame API

    The API is restricted to facilitate speed tradeoffs

    CommandLine:
        python -m netharn.util.util_dataframe DataFrameLight

    Example:
        >>> self = DataFrameLight({})
        >>> print('self = {!r}'.format(self))
        >>> self = DataFrameLight({'a': [0, 1, 2], 'b': [2, 3, 4]})
        >>> print('self = {!r}'.format(self))
        >>> item = self[0]
        >>> print('item = {!r}'.format(item))
    """
    def __init__(self, data=None):
        self._raw = data
        self._keys = None
        self._data = None
        self._localizer = LocLight(self)
        self.__normalize__()

    @property
    def iloc(self):
        return self._localizer

    @property
    def loc(self):
        return self._localizer

    def _pandas(self):
        """
        CommandLine:
            xdoctest netharn.util.util_dataframe DataFrameLight._demodata

            xdoctest -m netharn.util.util_dataframe DataFrameLight:._pandas:0 --bench

        Example:
            >>> from netharn.util.util_dataframe import *
            >>> _keys = ['list', 'dict', 'series', 'split', 'records', 'index']
            >>> results = []
            >>> df = DataFrameLight._demodata(num=10000)._pandas()
            >>> ti = ub.Timerit(verbose=True, unit='ms')
            >>> for key in _keys:
            >>>     result = ti.reset(key).call(lambda: df.to_dict(orient=key))
            >>>     results.append((result.mean(), result.report()))
            >>> # Custom keys
            >>> key = 'self._pandas'
            >>> self = DataFrameLight(df)
            >>> result = ti.reset(key).call(lambda: self._pandas())
            >>> results.append((result.mean(), result.report()))
            >>> key = 'light-from-pandas'
            >>> result = ti.reset(key).call(lambda: DataFrameLight(df))
            >>> results.append((result.mean(), result.report()))
            >>> key = 'light-from-dict'
            >>> result = ti.reset(key).call(lambda: DataFrameLight(self._data))
            >>> results.append((result.mean(), result.report()))
            >>> print('==============')
            >>> print('\n'.join([t[1] for t in sorted(results)]))
            >>> best = 'series'
            >>> x = df.to_dict(orient=best)


        Example:
            >>> from netharn.util.util_dataframe import *
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> df_heavy = df_light._pandas()
            >>> got = DataFrameLight(df_heavy)
            >>> assert got._data == df_light.data
        """
        return pd.DataFrame(self._data, columns=self._keys)

    @classmethod
    def _demodata(cls, num=7):
        """
        CommandLine:
            python -m netharn.util.util_dataframe DataFrameLight._demodata

            python -m netharn.util.util_dataframe DataFrameLight:1 --bench

        Example:
            >>> from netharn.util.util_dataframe import *
            >>> self = DataFrameLight._demodata(num=7)
            >>> print('self = {!r}'.format(self))
            >>> other = DataFrameLight._demodata(num=11)
            >>> print('other = {!r}'.format(other))
            >>> both = self.union(other)
            >>> print('both = {!r}'.format(both))
            >>> assert both is not self
            >>> assert other is not self

        Example:
            >>> # BENCHMARK
            >>> # xdoc: +REQUIRES(--bench)
            >>> from netharn.util.util_dataframe import *
            >>> import ubelt as ub
            >>> df_light = DataFrameLight._demodata(num=100)
            >>> df_heavy = df_light._pandas()
            >>> for timer in ub.Timerit(100, bestof=10, label='DF-heavy'):
            >>>     with timer:
            >>>         for i in range(10):
            >>>             df_heavy.iloc[i]
            >>> for timer in ub.Timerit(100, bestof=10, label='DF-Light'):
            >>>     with timer:
            >>>         for i in range(10):
            >>>             df_light.iloc[i]
        """
        demodata = {
            'foo': [0] * num,
            'bar': [1] * num,
            'baz': [2.73] * num,
        }
        self = cls(demodata)
        return self

    def __nice__(self):
        return 'keys: {}, len={}'.format(self._keys, len(self))

    def __len__(self):
        if self._data:
            return len(self._data[self._keys[0]])
        else:
            return 0

    def __normalize__(self):
        if self._raw is None:
            self._data = {}
            self.keys = []
        elif isinstance(self._raw, dict):
            self._keys = list(self._raw.keys())
            self._data = self._raw
            if __debug__:
                lens = []
                for d in self._data.values():
                    assert isinstance(d, list)
                    lens.append(len(d))
                assert ub.allsame(lens)

        elif isinstance(self._raw, pd.DataFrame):
            # self._data = self._raw.to_dict(orient='index')
            self._keys = list(self._raw.keys())
            # self._data = self._raw.to_dict(orient='series')
            self._data = self._raw.to_dict(orient='list')
        else:
            raise TypeError('Unknown _raw type')

    def keys(self):
        return iter(self._keys)

    def _getrow(self, index):
        return {key: self._data[key][index] for key in self._keys}

    def _getcol(self, key):
        return self._data[key]

    def __getitem__(self, index):
        assert self._keys is not None
        return self._getcol(index)

    def extend(self, other):
        for key in self._keys:
            vals1 = self._data[key]
            vals2 = other._data[key]
            vals1.extend(vals2)

    def union(self, other):
        both = copy.copy(self)
        both.extend(other)
        return both

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_dataframe all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
