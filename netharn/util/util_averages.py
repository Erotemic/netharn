# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import pandas as pd
import ubelt as ub
import numpy as np


def stats_dict(list_, axis=None, nan=False, sum=False, extreme=True,
               n_extreme=False, median=False, shape=True, size=False):
    """
    Args:
        list_ (listlike): values to get statistics of
        axis (int): if `list_` is ndarray then this specifies the axis
        nan (bool): report number of nan items
        sum (bool): report sum of values
        extreme (bool): report min and max values
        n_extreme (bool): report extreme value frequencies
        median (bool): report median
        size (bool): report array size
        shape (bool): report array shape

    Returns:
        collections.OrderedDict: stats: dictionary of common numpy statistics
            (min, max, mean, std, nMin, nMax, shape)

    SeeAlso:
        scipy.stats.describe

    CommandLine:
        python -m netharn.util.util_averages stats_dict

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> list_ = rng.rand(10, 2).astype(np.float32)
        >>> stats = stats_dict(list_, axis=axis, nan=False)
        >>> result = str(ub.repr2(stats, nl=1, precision=4, with_dtype=True))
        >>> print(result)
        {
            'mean': np.array([ 0.5206,  0.6425], dtype=np.float32),
            'std': np.array([ 0.2854,  0.2517], dtype=np.float32),
            'min': np.array([ 0.0202,  0.0871], dtype=np.float32),
            'max': np.array([ 0.9637,  0.9256], dtype=np.float32),
            'shape': (10, 2),
        }

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> axis = 0
        >>> rng = np.random.RandomState(0)
        >>> list_ = rng.randint(0, 42, size=100).astype(np.float32)
        >>> list_[4] = np.nan
        >>> stats = stats_dict(list_, axis=axis, nan=True)
        >>> result = str(ub.repr2(stats, nl=0, precision=1, strkeys=True))
        >>> print(result)
        {mean: 20.0, std: 13.2, min: 0.0, max: 41.0, shape: (100,), num_nan: 1}
    """
    stats = collections.OrderedDict([])

    # Ensure input is in numpy format
    if isinstance(list_, np.ndarray):
        nparr = list_
    elif isinstance(list_, list):
        nparr = np.array(list_)
    else:
        nparr = np.array(list(list_))
    # Check to make sure stats are feasible
    if len(nparr) == 0:
        stats['empty_list'] = True
        if size:
            stats['size'] = 0
    else:
        if nan:
            min_val = np.nanmin(nparr, axis=axis)
            max_val = np.nanmax(nparr, axis=axis)
            mean_ = np.nanmean(nparr, axis=axis)
            std_  = np.nanstd(nparr, axis=axis)
        else:
            min_val = nparr.min(axis=axis)
            max_val = nparr.max(axis=axis)
            mean_ = nparr.mean(axis=axis)
            std_  = nparr.std(axis=axis)
        # number of entries with min/max val
        nMin = np.sum(nparr == min_val, axis=axis)
        nMax = np.sum(nparr == max_val, axis=axis)

        # Notes:
        # the first central moment is 0
        # the first raw moment is the mean
        # the second central moment is the variance
        # the third central moment is the skweness * var ** 3
        # the fourth central moment is the kurtosis * var ** 4
        # the third standardized moment is the skweness
        # the fourth standardized moment is the kurtosis

        if True:
            stats['mean'] = np.float32(mean_)
            stats['std'] = np.float32(std_)
        if extreme:
            stats['min'] = np.float32(min_val)
            stats['max'] = np.float32(max_val)
        if n_extreme:
            stats['nMin'] = np.int32(nMin)
            stats['nMax'] = np.int32(nMax)
        if size:
            stats['size'] = nparr.size
        if shape:
            stats['shape'] = nparr.shape
        if median:
            stats['med'] = np.nanmedian(nparr)
        if nan:
            stats['num_nan'] = np.isnan(nparr).sum()
        if sum:
            sumfunc = np.nansum if nan else np.sum
            stats['sum'] = sumfunc(nparr, axis=axis)
    return stats


class MovingAve(ub.NiceRepr):
    def average(self):
        raise NotImplementedError()

    def update(self, other):
        raise NotImplementedError()

    def __nice__(self):
        return str(ub.repr2(self.average(), nl=0))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class CumMovingAve(MovingAve):
    """
    Cumulative moving average of dictionary values

    References:
        https://en.wikipedia.org/wiki/Moving_average

    Example:
        >>> from netharn.util.util_averages import *
        >>> self = CumMovingAve()
        >>> print(str(self.update({'a': 10})))
        <CumMovingAve({'a': 10.0})>
        >>> print(str(self.update({'a': 0})))
        <CumMovingAve({'a': 5.0})>
        >>> print(str(self.update({'a': 2})))
        <CumMovingAve({'a': 4.0})>

    Example:
        >>> from netharn.util.util_averages import *
        >>> self = CumMovingAve(nan_method='ignore')
        >>> print(str(self.update({'a': 10})))
        >>> print(str(self.update({'a': 0})))
        >>> print(str(self.update({'a': np.nan})))
    """
    def __init__(self, nan_method='zero'):
        self.totals = ub.odict()
        self.nan_method = nan_method
        self.weights = ub.odict()
        if self.nan_method not in {'ignore', 'zero'}:
            raise KeyError(self.nan_method)

    def average(self):
        return {k: v / float(self.weights[k]) for k, v in self.totals.items()}

    def update(self, other):
        for k, v in other.items():
            if pd.isnull(v):
                if self.nan_method == 'ignore':
                    continue
                elif self.nan_method == 'zero':
                    v = 0.0
                    w = 1.0
                else:
                    raise AssertionError
            else:
                w = 1.0
            if k not in self.totals:
                self.totals[k] = 0.0
                self.weights[k] = 0.0
            self.totals[k] += v  # should be v * w
            self.weights[k] += w
        return self


class WindowedMovingAve(MovingAve):
    """
    Windowed moving average of dictionary values

    Args:
        window (int): number of previous observations to consider

    Example:
        >>> self = WindowedMovingAve(window=3)
        >>> print(str(self.update({'a': 10})))
        <WindowedMovingAve({'a': 10.0})>
        >>> print(str(self.update({'a': 0})))
        <WindowedMovingAve({'a': 5.0})>
        >>> print(str(self.update({'a': 2})))
        <WindowedMovingAve({'a': 4.0})>
    """
    def __init__(self, window=500):
        self.window = window
        self.totals = ub.odict()
        self.history = {}

    def average(self):
        return {k: v / float(len(self.history[k])) for k, v in self.totals.items()}

    def update(self, other):
        for k, v in other.items():
            if pd.isnull(v):
                v = 0
            if k not in self.totals:
                self.history[k] = collections.deque()
                self.totals[k] = 0
            self.totals[k] += v
            self.history[k].append(v)
            if len(self.history[k]) > self.window:
                # Push out the oldest value
                self.totals[k] -= self.history[k].popleft()
        return self


class ExpMovingAve(MovingAve):
    """
    Exponentially weighted moving average of dictionary values

    Args:
        span (float): roughly corresponds to window size.
            equivalent to (2 / alpha) - 1
        alpha (float): roughly corresponds to window size.
            equivalent to 2 / (span + 1)

    References:
        http://greenteapress.com/thinkstats2/html/thinkstats2013.html

    Example:
        >>> self = ExpMovingAve(span=3)
        >>> print(str(self.update({'a': 10})))
        <ExpMovingAve({'a': 10})>
        >>> print(str(self.update({'a': 0})))
        <ExpMovingAve({'a': 5.0})>
        >>> print(str(self.update({'a': 2})))
        <ExpMovingAve({'a': 3.5})>
    """
    def __init__(self, span=None, alpha=None):
        values = ub.odict()
        self.values = values
        if span is None and alpha is None:
            alpha = 0
        if not bool(span is None) ^ bool(alpha is None):
            raise ValueError('specify either alpha xor span')

        if alpha is not None:
            self.alpha = alpha
        elif span is not None:
            self.alpha = 2 / (span + 1)
        else:
            raise AssertionError('impossible state')

    def average(self):
        return self.values

    def update(self, other):
        alpha = self.alpha
        for k, v in other.items():
            if pd.isnull(v):
                v = 0
            if k not in self.values:
                self.values[k] = v
            else:
                self.values[k] = (alpha * v) + (1 - alpha) * self.values[k]
        return self


class RunningStats(object):
    """
    Dynamically records per-element array statistics and can summarized them
    per-element, across channels, or globally.

    SeeAlso:
        InternalRunningStats

    Example:
        >>> run = RunningStats()
        >>> ch1 = np.array([[0, 1], [3, 4]])
        >>> ch2 = np.zeros((2, 2))
        >>> img = np.dstack([ch1, ch2])
        >>> run.update(np.dstack([ch1, ch2]))
        >>> run.update(np.dstack([ch1 + 1, ch2]))
        >>> run.update(np.dstack([ch1 + 2, ch2]))
        >>> # Scalar averages
        >>> print(ub.repr2(run.simple(), nobr=1, si=True))
        >>> # Per channel averages
        >>> print(ub.repr2(ub.map_vals(lambda x: np.array(x).tolist(), run.simple()), nobr=1, si=True, nl=1))
        >>> # Per-pixel averages
        >>> print(ub.repr2(ub.map_vals(lambda x: np.array(x).tolist(), run.detail()), nobr=1, si=True, nl=1))
        """

    def __init__(run):
        run.raw_max = -np.inf
        run.raw_min = np.inf
        run.raw_total = 0
        run.raw_squares = 0
        run.n = 0

    def update(run, img):
        run.n += 1
        # Update stats across images
        run.raw_max = np.maximum(run.raw_max, img)
        run.raw_min = np.minimum(run.raw_min, img)
        run.raw_total += img
        run.raw_squares += img ** 2

    def _sumsq_std(run, total, squares, n):
        """
        Sum of squares method to compute standard deviation
        """
        numer = (n * squares - total ** 2)
        denom = (n * (n - 1))
        std = np.sqrt(numer / denom)
        return std

    def simple(run, axis=None):
        assert run.n > 0, 'no stats exist'
        maxi    = run.raw_max.max(axis=axis, keepdims=True)
        mini    = run.raw_min.min(axis=axis, keepdims=True)
        total   = run.raw_total.sum(axis=axis, keepdims=True)
        squares = run.raw_squares.sum(axis=axis, keepdims=True)
        if not hasattr(run.raw_total, 'shape'):
            n = run.n
        elif axis is None:
            n = run.n * np.prod(run.raw_total.shape)
        else:
            n = run.n * np.prod(np.take(run.raw_total.shape, axis))
        info = ub.odict([
            ('n', n),
            ('max', maxi),
            ('min', mini),
            ('total', total),
            ('squares', squares),
            ('mean', total / n),
            ('std', run._sumsq_std(total, squares, n)),
        ])
        return info

    def detail(run):
        total = run.raw_total
        squares = run.raw_squares
        maxi = run.raw_max
        mini = run.raw_min
        n = run.n
        info = ub.odict([
            ('n', n),
            ('max', maxi),
            ('min', mini),
            ('total', total),
            ('squares', squares),
            ('mean', total / n),
            ('std', run._sumsq_std(total, squares, n)),
        ])
        return info


def absdev(x, ave=np.mean, central=np.median, axis=None):
    """
    Average absolute deviation from a point of central tendency

    The `ave` absolute deviation from the `central`.

    Args:
        x (np.ndarray): input data
        axis (tuple): summarize over
        central (np.ufunc): function to get measure the center
            defaults to np.median
        ave (np.ufunc): function to average deviation over.
            defaults to np.mean

    Returns:
        np.ndarray : average_deviations

    References:
        https://en.wikipedia.org/wiki/Average_absolute_deviation

    Example:
        >>> # xdoctest: +IGNORE_WHITESPACE
        >>> x = np.array([[[0, 1], [3, 4]],
        >>>               [[0, 0], [0, 0]]])
        >>> axis = (0, 1)
        >>> absdev(x, np.mean, np.median, axis=(0, 1))
        array([0.75, 1.25])
        >>> absdev(x, np.median, np.median, axis=(0, 1))
        array([0. , 0.5])
        >>> absdev(x, np.mean, np.median)
        1.0
        >>> absdev(x, np.median, np.median)
        0.0
        >>> absdev(x, np.median, np.median, axis=0)
        array([[0. , 0.5], [1.5, 2. ]])
    """
    point = central(x, axis=axis, keepdims=True)
    deviations = np.abs(x - point)
    average_deviations = ave(deviations, axis=axis)
    return average_deviations


class InternalRunningStats():
    """
    Maintains an averages of average internal statistics across a dataset.

    The difference between `RunningStats` and this is that the former can keep
    track of the average value of pixel (x, y) or channel (c) across the
    dataset, whereas this class tracks the average pixel value within an image
    across the dataset. So, this is an average of averages.

    Example:
        >>> ch1 = np.array([[0, 1], [3, 4]])
        >>> ch2 = np.zeros((2, 2))
        >>> img = np.dstack([ch1, ch2])
        >>> irun = InternalRunningStats(axis=(0, 1))
        >>> irun.update(np.dstack([ch1, ch2]))
        >>> irun.update(np.dstack([ch1 + 1, ch2]))
        >>> irun.update(np.dstack([ch1 + 2, ch2]))
        >>> # Scalar averages
        >>> print(ub.repr2(irun.info(), nobr=1, si=True))
    """

    def __init__(irun, axis=None):
        from functools import partial
        irun.axis = axis
        # Define a running stats object for each as well as the function to
        # compute the internal statistic
        irun.runs = ub.odict([
            ('mean', (
                RunningStats(), np.mean)),
            ('std', (
                RunningStats(), np.std)),
            ('median', (
                RunningStats(), np.median)),
            # ('mean_absdev_from_mean', (
            #     RunningStats(),
            #     partial(absdev, ave=np.mean, central=np.mean))),
            ('mean_absdev_from_median', (
                RunningStats(),
                partial(absdev, ave=np.mean, central=np.median))),
            ('median_absdev_from_median', (
                RunningStats(),
                partial(absdev, ave=np.median, central=np.median))),
        ])

    def update(irun, img):
        axis = irun.axis
        for run, func in irun.runs.values():
            stat = func(img, axis=axis)
            run.update(stat)

    def info(irun):
        return {
            key: run.detail() for key, (run, _) in irun.runs.items()
        }


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.util.util_averages all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
