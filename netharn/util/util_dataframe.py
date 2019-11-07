# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import pandas as pd
import numpy as np
import copy


__version__ = '0.0.1'


class LocLight(object):
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, index):
        return self.parent._getrow(index)


class DataFrameLight(ub.NiceRepr):
    r"""
    Implements a subset of the pandas.DataFrame API

    The API is restricted to facilitate speed tradeoffs

    Notes:
        pandas.DataFrame is slow. DataFrameLight is faster.
        It is a tad more restrictive though.

    CommandLine:
        python -m netharn.util.util_dataframe DataFrameLight:1 --bench

    Example:
        >>> self = DataFrameLight({})
        >>> print('self = {!r}'.format(self))
        >>> self = DataFrameLight({'a': [0, 1, 2], 'b': [2, 3, 4]})
        >>> print('self = {!r}'.format(self))
        >>> item = self.iloc[0]
        >>> print('item = {!r}'.format(item))

    Example:
        >>> # BENCHMARK
        >>> # xdoc: +REQUIRES(--bench)
        >>> from netharn.util.util_dataframe import *
        >>> import ubelt as ub
        >>> NUM = 1000
        >>> print('NUM = {!r}'.format(NUM))
        >>> # to_dict conversions
        >>> print('==============')
        >>> print('====== to_dict conversions =====')
        >>> _keys = ['list', 'dict', 'series', 'split', 'records', 'index']
        >>> results = []
        >>> df = DataFrameLight._demodata(num=NUM)._pandas()
        >>> ti = ub.Timerit(verbose=False, unit='ms')
        >>> for key in _keys:
        >>>     result = ti.reset(key).call(lambda: df.to_dict(orient=key))
        >>>     results.append((result.mean(), result.report()))
        >>> key = 'series+numpy'
        >>> result = ti.reset(key).call(lambda: {k: v.values for k, v in df.to_dict(orient='series').items()})
        >>> results.append((result.mean(), result.report()))
        >>> print('\n'.join([t[1] for t in sorted(results)]))
        >>> print('==============')
        >>> print('====== DFLight Conversions =======')
        >>> ti = ub.Timerit(verbose=True, unit='ms')
        >>> key = 'self._pandas'
        >>> self = DataFrameLight(df)
        >>> ti.reset(key).call(lambda: self._pandas())
        >>> key = 'light-from-pandas'
        >>> ti.reset(key).call(lambda: DataFrameLight(df))
        >>> key = 'light-from-dict'
        >>> ti.reset(key).call(lambda: DataFrameLight(self._data))
        >>> print('==============')
        >>> print('====== BENCHMARK: .LOC[] =======')
        >>> from netharn.util.util_dataframe import *
        >>> ti = ub.Timerit(num=20, bestof=4, verbose=True, unit='ms')
        >>> df_light = DataFrameLight._demodata(num=NUM)
        >>> df_heavy = df_light._pandas()
        >>> series_data = df_heavy.to_dict(orient='series')
        >>> list_data = df_heavy.to_dict(orient='list')
        >>> np_data = {k: v.values for k, v in df_heavy.to_dict(orient='series').items()}
        >>> for timer in ti.reset('DF-heavy.iloc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_heavy.iloc[i]
        >>> for timer in ti.reset('DF-heavy.loc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_heavy.iloc[i]
        >>> for timer in ti.reset('dict[SERIES].loc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: series_data[key].loc[i] for key in series_data.keys()}
        >>> for timer in ti.reset('dict[SERIES].iloc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: series_data[key].iloc[i] for key in series_data.keys()}
        >>> for timer in ti.reset('dict[SERIES][]'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: series_data[key][i] for key in series_data.keys()}
        >>> for timer in ti.reset('dict[NDARRAY][]'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: np_data[key][i] for key in np_data.keys()}
        >>> for timer in ti.reset('dict[list][]'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             {key: list_data[key][i] for key in np_data.keys()}
        >>> for timer in ti.reset('DF-Light.iloc/loc'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_light.iloc[i]
        >>> for timer in ti.reset('DF-Light._getrow'):
        >>>     with timer:
        >>>         for i in range(NUM):
        >>>             df_light._getrow(i)
    """
    def __init__(self, data=None, columns=None):
        if columns is not None:
            if data is None:
                data = ub.odict(zip(columns, [[]] * len(columns)))
            else:
                data = ub.odict(zip(columns, data.T))

        self._raw = data
        self._data = None
        self._localizer = LocLight(self)
        self.__normalize__()

    @property
    def iloc(self):
        return self._localizer

    @property
    def loc(self):
        return self._localizer

    def to_string(self, *args, **kwargs):
        return self._pandas().to_string(*args, **kwargs)

    def _pandas(self):
        """
        CommandLine:
            xdoctest netharn.util.util_dataframe DataFrameLight._demodata

            xdoctest -m netharn.util.util_dataframe DataFrameLight:._pandas:0 --bench

        Example:
            >>> from netharn.util.util_dataframe import *
            >>> df_light = DataFrameLight._demodata(num=7)
            >>> df_heavy = df_light._pandas()
            >>> got = DataFrameLight(df_heavy)
            >>> assert got._data == df_light._data
        """
        return pd.DataFrame(self._data)

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
        """
        demodata = {
            'foo': [0] * num,
            'bar': [1] * num,
            'baz': [2.73] * num,
        }
        self = cls(demodata)
        return self

    def __nice__(self):
        return 'keys: {}, len={}'.format(list(self.keys()), len(self))

    def __len__(self):
        if self._data:
            key = next(iter(self.keys()))
            return len(self._data[key])
        else:
            return 0

    def __normalize__(self):
        if self._raw is None:
            self._data = {}
        elif isinstance(self._raw, dict):
            self._data = self._raw
            if __debug__:
                lens = []
                for d in self._data.values():
                    if not isinstance(d, (list, np.ndarray)):
                        raise TypeError(type(d))
                    lens.append(len(d))
                assert ub.allsame(lens)
        elif isinstance(self._raw, DataFrameLight):
            self._data = copy.copy(self._raw._data)
        elif isinstance(self._raw, pd.DataFrame):
            self._data = self._raw.to_dict(orient='list')
        else:
            raise TypeError('Unknown _raw type')

    @property
    def columns(self):
        return list(self.keys())

    def sort_values(self, key, inplace=False):
        sortx = np.argsort(self._getcol(key))
        return self.take(sortx, inplace=inplace)

    def keys(self):
        if self._data:
            for key in self._data.keys():
                yield key

    def _getrow(self, index):
        return {key: self._data[key][index] for key in self._data.keys()}

    def _getcol(self, key):
        return self._data[key]

    def __getitem__(self, key):
        return self._getcol(key)

    def __setitem__(self, key, value):
        self._data[key] = value

    def compress(self, flags, inplace=False):
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = list(ub.compress(self._data[key], flags))
        return subset

    def take(self, indices, inplace=False):
        subset = self if inplace else self.__class__()
        if isinstance(indices, slice):
            for key in self._data.keys():
                subset._data[key] = self._data[key][indices]
        else:
            for key in self._data.keys():
                subset._data[key] = list(ub.take(self._data[key], indices))
        return subset

    def extend(self, other):
        for key in self._data.keys():
            # TODO: handle numpy values
            vals1 = self._data[key]
            vals2 = other._data[key]
            try:
                vals1.extend(vals2)
            except AttributeError:
                if isinstance(vals1, np.ndarray):
                    self._data[key] = np.hstack([vals1, vals2])

    def copy(self):
        other = copy.copy(self)
        other._data = other._data.copy()
        other._localizer = LocLight(other)
        return other

    def union(self, *others):
        if isinstance(self, DataFrameLight):
            first = self
            rest = others
        else:
            if len(others) == 0:
                return DataFrameLight()
            first = others[0]
            rest = others[1:]

        both = first.copy()
        if not both.keys:
            for other in rest:
                if other.keys:
                    both.keys = copy.copy(other.keys)
                    break

        for other in rest:
            both.extend(other)
        return both

    @classmethod
    def concat(cls, others):
        return cls.union(*others)

    @classmethod
    def from_dict(cls, records):
        record_iter = iter(records)
        columns = {}
        try:
            r = next(record_iter)
            for key, value in r.items():
                columns[key] = [value]
        except StopIteration:
            pass
        else:
            for r in record_iter:
                for key, value in r.items():
                    columns[key].append(value)
        self = cls(columns)
        return self

    def reset_index(self, drop=False):
        """ noop for compatability, the light version doesnt store an index """
        return self

    def groupby(self, *args, **kw):
        """ hacked slow pandas implementation of groupby """
        return self._pandas().gropuby(*args, **kw)

    def rename(self, columns, inplace=False):
        if not inplace:
            self = self.copy()
        for old, new in columns.items():
            if old in self._data:
                self._data[new] = self._data.pop(old)
        return self


class DataFrameArray(DataFrameLight):
    """
    Take and compress are much faster, but extend and union are slower
    """

    def __normalize__(self):
        if self._raw is None:
            self._data = {}
        elif isinstance(self._raw, dict):
            self._data = self._raw
            if __debug__:
                lens = []
                for d in self._data.values():
                    if not isinstance(d, (list, np.ndarray)):
                        raise TypeError(type(d))
                    lens.append(len(d))
                assert ub.allsame(lens), (
                    'lens are not all same {} for columns {}'.format(
                        lens,
                        list(self._data.keys()))
                )
        elif isinstance(self._raw, DataFrameLight):
            self._data = copy.copy(self._raw._data)
        elif isinstance(self._raw, pd.DataFrame):
            self._data = {k: v.values for k, v in self._raw.to_dict(orient='series').items()}
        else:
            raise TypeError('Unknown _raw type')
        # self._data = ub.map_vals(np.asarray, self._data)  # does this break anything?

    def extend(self, other):
        for key in self._data.keys():
            vals1 = self._data[key]
            vals2 = other._data[key]
            self._data[key] = np.hstack([vals1, vals2])

    def compress(self, flags, inplace=False):
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = self._data[key][flags]
        return subset

    def take(self, indices, inplace=False):
        subset = self if inplace else self.__class__()
        for key in self._data.keys():
            subset._data[key] = self._data[key][indices]
        return subset

    # def min(self, axis=None):
    #     return self._extreme(func=np.minimum, axis=axis)

    # def max(self, axis=None):
    #     """
    #     Example:
    #         >>> from netharn.util.util_dataframe import *
    #         >>> self = DataFrameArray._demodata(num=7)
    #         >>> func = np.maximum
    #     """
    #     return self._extreme(func=np.maximum, axis=axis)

    # def _extreme(self, func, axis=None):
    #     import netharn as nh
    #     if axis is None:
    #         raise NotImplementedError
    #     if axis == 0:
    #         raise NotImplementedError
    #     elif axis == 1:
    #         newdata = nh.util.iter_reduce_ufunc(func, (self[key] for key in self.keys()))
    #         newobj = self.__class__(newdata, self._keys)
    #     else:
    #         raise NotImplementedError

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_dataframe all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
