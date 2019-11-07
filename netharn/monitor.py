"""
TODO: Implement algorithm from dlib
http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html

"""
from netharn import util
import itertools as it
import numpy as np
import ubelt as ub

__all__ = ['Monitor']


def demodata_monitor():
    rng = np.random.RandomState(0)
    n = 300
    losses = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)[::-1]
    mious = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)
    monitor = Monitor(minimize=['loss'], maximize=['miou'], smoothing=.6)
    for epoch, (loss, miou) in enumerate(zip(losses, mious)):
        monitor.update(epoch, {'loss': loss, 'miou': miou})
    return monitor


class Monitor(object):
    """
    Example:
        >>> # simulate loss going down and then overfitting
        >>> from netharn.monitor import *
        >>> rng = np.random.RandomState(0)
        >>> n = 300
        >>> losses = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)[::-1]
        >>> mious = (sorted(rng.randint(10, n, size=n)) + rng.randint(0, 20, size=n) - 10)
        >>> monitor = Monitor(minimize=['loss'], maximize=['miou'], smoothing=.6)
        >>> for epoch, (loss, miou) in enumerate(zip(losses, mious)):
        >>>     monitor.update(epoch, {'loss': loss, 'miou': miou})
        >>> # xdoc: +REQUIRES(--show)
        >>> monitor.show()
    """

    def __init__(monitor, minimize=['loss'], maximize=[], smoothing=.6,
                 patience=None, max_epoch=1000):
        monitor.ewma = util.ExpMovingAve(alpha=1 - smoothing)
        monitor.raw_metrics = []
        monitor.smooth_metrics = []
        monitor.epochs = []
        monitor.is_good = []
        # monitor.other_data = []

        # Keep track of which metrics we want to maximize / minimize
        monitor.minimize = minimize
        monitor.maximize = maximize
        # print('monitor.minimize = {!r}'.format(monitor.minimize))
        # print('monitor.maximize = {!r}'.format(monitor.maximize))

        monitor.best_raw_metrics = None
        monitor.best_smooth_metrics = None
        monitor.best_epoch = None

        # early stopping

        monitor.patience = patience
        monitor.n_bad_epochs = 0

        monitor.max_epoch = max_epoch

    def show(monitor):
        import matplotlib.pyplot as plt
        from netharn.util import mplutil
        import pandas as pd
        # mplutil.qtensure()
        smooth_ydatas = pd.DataFrame.from_dict(monitor.smooth_metrics).to_dict('list')
        raw_ydatas = pd.DataFrame.from_dict(monitor.raw_metrics).to_dict('list')
        keys = monitor.minimize + monitor.maximize
        pnum_ = mplutil.PlotNums(nSubplots=len(keys))
        for i, key in enumerate(keys):
            mplutil.multi_plot(
                monitor.epochs, {'raw ' + key: raw_ydatas[key],
                                 'smooth ' + key: smooth_ydatas[key]},
                xlabel='epoch', ylabel=key, pnum=pnum_[i], fnum=1,
                # markers={'raw ' + key: '-', 'smooth ' + key: '--'},
                # colors={'raw ' + key: 'b', 'smooth ' + key: 'b'},
            )

            # star all the good epochs
            flags = np.array(monitor.is_good)
            if np.any(flags):
                plt.plot(list(ub.compress(monitor.epochs, flags)),
                         list(ub.compress(smooth_ydatas[key], flags)), 'b*')

    def __getstate__(monitor):
        state = monitor.__dict__.copy()
        ewma = state.pop('ewma')
        state['ewma_state'] = ewma.__dict__
        return state

    def __setstate__(monitor, state):
        ewma_state = state.pop('ewma_state', None)
        if ewma_state is not None:
            monitor.ewma = util.ExpMovingAve()
            monitor.ewma.__dict__.update(ewma_state)
        monitor.__dict__.update(**state)

    def state_dict(monitor):
        return monitor.__getstate__()

    def load_state_dict(monitor, state):
        return monitor.__setstate__(state)

    def update(monitor, epoch, raw_metrics):
        monitor.epochs.append(epoch)
        monitor.raw_metrics.append(raw_metrics)
        monitor.ewma.update(raw_metrics)
        # monitor.other_data.append(other)

        smooth_metrics = monitor.ewma.average()
        monitor.smooth_metrics.append(smooth_metrics.copy())

        improved_keys = monitor._improved(smooth_metrics, monitor.best_smooth_metrics)
        if improved_keys:
            if monitor.best_smooth_metrics is None:
                monitor.best_smooth_metrics = smooth_metrics.copy()
                monitor.best_raw_metrics = raw_metrics.copy()
            else:
                for key in improved_keys:
                    monitor.best_smooth_metrics[key] = smooth_metrics[key]
                    monitor.best_raw_metrics[key] = raw_metrics[key]
            monitor.best_epoch = epoch
            monitor.n_bad_epochs = 0
        else:
            monitor.n_bad_epochs += 1

        improved = len(improved_keys) > 0
        monitor.is_good.append(improved)
        return improved

    def _improved(monitor, metrics, best_metrics):
        """
        If any of the metrics we care about is improving then we are happy

        Example:
            >>> from netharn.monitor import *
            >>> monitor = Monitor(['loss'], ['acc'])
            >>> metrics = {'loss': 5, 'acc': .99}
            >>> best_metrics = {'loss': 4, 'acc': .98}
        """
        keys = monitor.maximize + monitor.minimize

        def _as_minimization(metrics):
            # convert to a minimization problem
            sign = np.array(([-1] * len(monitor.maximize)) +
                            ([1] * len(monitor.minimize)))
            chosen = np.array(list(ub.take(metrics, keys)))
            return chosen, sign

        current, sign1 = _as_minimization(metrics)

        if not best_metrics:
            return keys

        best, sign2 = _as_minimization(best_metrics)

        # TODO: also need to see if anything got significantly worse

        # only use threshold rel mode
        monitor.rel_threshold = 1e-6
        rel_epsilon = 1.0 - monitor.rel_threshold
        improved_flags = (sign1 * current) < (rel_epsilon * sign2 * best)
        # * rel_epsilon

        improved_keys = list(ub.compress(keys, improved_flags))
        return improved_keys

    def is_done(monitor):
        if monitor.patience is None:
            return False
        return monitor.n_bad_epochs >= monitor.patience

    def message(monitor):
        if not monitor.epochs:
            return ub.color_text('vloss is unevaluated', 'blue')

        prev_loss = monitor.smooth_metrics[-1]['loss']
        best_loss = monitor.best_smooth_metrics['loss']

        message = 'vloss: {:.4f} (n_bad={:02d}, best={:.4f})'.format(
            prev_loss, monitor.n_bad_epochs, best_loss,
        )
        if monitor.patience is None:
            patience = monitor.max_epoch
        else:
            patience = monitor.patience
        if monitor.n_bad_epochs <= int(patience * .25):
            message = ub.color_text(message, 'green')
        elif monitor.n_bad_epochs >= int(patience * .75):
            message = ub.color_text(message, 'red')
        else:
            message = ub.color_text(message, 'yellow')
        return message

    def best_epochs(monitor, num=None, smooth=True):
        """
        Returns the best `num` epochs for every metric.

        Example:
            >>> monitor = demodata_monitor()
            >>> metric_ranks = monitor.best_epochs(5)
            >>> print(ub.repr2(metric_ranks, with_dtype=False, nl=1))
            {
                'loss': np.array([297, 299, 298, 296, 295]),
                'miou': np.array([299, 298, 297, 296, 295]),
            }
        """
        metric_ranks = {}
        for key in it.chain(monitor.minimize, monitor.maximize):
            metric_ranks[key] = monitor._rank(key, smooth=True)[:num]
        return metric_ranks

    def _rank(monitor, key, smooth=True):
        """
        Example:
            >>> monitor = demodata_monitor()
            >>> ranked_epochs = monitor._rank('loss', smooth=False)
            >>> ranked_epochs = monitor._rank('miou', smooth=True)
        """
        if smooth:
            metrics = monitor.smooth_metrics
        else:
            metrics = monitor.raw_metrics

        values = [m[key] for m in metrics]
        sortx = np.argsort(values)
        if key in monitor.maximize:
            sortx = np.argsort(values)[::-1]
        elif key in monitor.minimize:
            sortx = np.argsort(values)
        else:
            raise KeyError(type)
        ranked_epochs = np.array(monitor.epochs)[sortx]
        return ranked_epochs

    def rank_epochs(monitor):
        """
        FIXME:
            broken - implement better rank aggregation with custom weights

        Example:
            >>> monitor = demodata_monitor()
            >>> monitor.rank_epochs()
        """
        rankings = {}
        for key, value in monitor.best_epochs(smooth=False).items():
            rankings[key + '_raw'] = value

        for key, value in monitor.best_epochs(smooth=True).items():
            rankings[key + '_smooth'] = value

        # borda-like weighted rank aggregation.
        # probably could do something better.
        epoch_to_weight = ub.ddict(lambda: 0)
        for key, ranking in rankings.items():
            # weights = np.linspace(0, 1, num=len(ranking))[::-1]
            weights = np.logspace(0, 2, num=len(ranking))[::-1] / 100
            for epoch, w in zip(ranking, weights):
                epoch_to_weight[epoch] += w

        agg_ranking = ub.argsort(epoch_to_weight)[::-1]
        return agg_ranking
