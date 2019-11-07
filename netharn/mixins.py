"""
Under development! Function names and logic may change at any time. Nothing in
this file should be considered as stable! Use at your own risk.

These are methods that you can mixin to your FitHarn implementation to extend
its functionality to typical, but non-default cases.

The purpose of this file is to contain functions that might not general-purpose
enough to add to FitHarn itself, but they are also common enough, where it
makes no sense to write them from scratch for each new project.
"""


# import xdev
# @xdev.profile
def _dump_monitor_tensorboard(harn, mode='epoch'):
    """
    Dumps PNGs to disk visualizing tensorboard scalars.
    Also dumps pickles to disk containing the same information.

    Args:
        mode : can be either epoch or iter

    CommandLine:
        xdoctest -m netharn.mixins _dump_monitor_tensorboard --profile

    Example:
        >>> from netharn.export.deployer import _demodata_toy_harn
        >>> from netharn.mixins import _dump_monitor_tensorboard
        >>> harn = _demodata_toy_harn()
        >>> harn.run()
        >>> try:
        >>>     _dump_monitor_tensorboard(harn)
        >>> except ImportError:
        >>>     pass
    """
    import ubelt as ub
    import netharn as nh
    from os.path import join
    import json
    from six.moves import cPickle as pickle

    serial = False
    harn.debug('Plotting tensorboard data. serial={}, mode={}'.format(serial, mode))

    tb_data = nh.util.read_tensorboard_scalars(harn.train_dpath, cache=0,
                                               verbose=0)

    out_dpath = ub.ensuredir((harn.train_dpath, 'monitor', 'tensorboard'))

    tb_data_pickle_fpath = join(out_dpath, 'tb_data.pkl')
    with open(tb_data_pickle_fpath, 'wb') as file:
        pickle.dump(tb_data, file)

    tb_data_json_fpath = join(out_dpath, 'tb_data.json')
    with open(tb_data_json_fpath, 'w') as file:
        try:
            json.dump(tb_data, file, indent=' ')
        except Exception as ex:
            json.dump({
                'error': 'Unable to write to json.',
                'info': 'See pickle file: {}'.format(tb_data_json_fpath)},
                file, indent=' ')

    nice = harn.hyper.nice

    func = _dump_measures
    args = (tb_data, nice, out_dpath, mode)

    if not serial:
        import multiprocessing
        proc = multiprocessing.Process(target=func, args=args)
        proc.daemon = True
        proc.start()
    else:
        func(*args)


def _dump_measures(tb_data, nice, out_dpath, mode, smoothing=0.6,
                   ignore_outliers=True):
    """
    This is its own function in case we need to modify formatting

    Ignore:
        # Reread a dumped pickle file
        import pickle
        import ubelt as ub
        out_dpath = ub.expandpath('~/work/project/fit/nice/nicename/monitor/tensorboard/')
        fpath = join(out_dpath, 'tb_data.pkl')
        tb_data = pickle.load(open(fpath, 'rb'))
        nice = 'nicename'
        _dump_measures(tb_data, nice, out_dpath)

    Ignore:
        import seaborn as sbn
        sbn.set()
        # Reread a dumped pickle file
        from netharn.mixins import _dump_measures
        import pickle
        import ubelt as ub
        out_dpath = ub.expandpath('~/work/lava/fit/nice/holdout_m1_abrams_v7.4.2/monitor/tensorboard/')
        fpath = join(out_dpath, 'tb_data.pkl')
        tb_data = pickle.load(open(fpath, 'rb'))
        nice = 'holdout_m1_abrams_v7.4.2'
        _dump_measures(tb_data, nice, out_dpath, smoothing=0.6)
    """
    import ubelt as ub
    import netharn as nh
    from os.path import join
    import numpy as np
    nh.util.autompl()

    fig = nh.util.figure(fnum=1)

    plot_keys = [key for key in tb_data if
                 ('train_' + mode in key or
                  'vali_' + mode in key or
                  'test_' + mode in key or
                  mode + '_' in key)]
    y01_measures = ['_acc', '_ap', '_mAP', '_auc', '_mcc', '_brier', '_mauc']
    y0_measures = ['error', 'loss']

    keys = set(tb_data.keys()).intersection(set(plot_keys))

    # print('mode = {!r}'.format(mode))
    # print('tb_data.keys() = {!r}'.format(tb_data.keys()))
    # print('plot_keys = {!r}'.format(plot_keys))
    # print('keys = {!r}'.format(keys))

    def smooth_curve(ydata, beta):
        """
        Curve smoothing algorithm used by tensorboard
        """
        import pandas as pd
        alpha = 1.0 - beta
        if alpha <= 0:
            return ydata
        ydata_smooth = pd.Series(ydata).ewm(alpha=alpha).mean().values
        return ydata_smooth

    def inlier_ylim(ydatas):
        """
        outlier removal used by tensorboard
        """
        low, high = None, None
        for ydata in ydatas:
            low_, high_ = np.percentile(ydata, [5, 95])
            low = low_ if low is None else min(low_, low)
            high = high_ if high is None else max(high_, high)
        return (low, high)

    GROUP_LOSSES = True
    if GROUP_LOSSES:
        # Group all losses in one plot for comparison
        def tag_grouper(k):
            # parts = ['train_epoch', 'vali_epoch', 'test_epoch']
            # parts = [p.replace('epoch', 'mode') for p in parts]
            parts = [p + mode for p in ['train_', 'vali_', 'test_']]
            for p in parts:
                if p in k:
                    return p.split('_')[0]
            return 'unknown'
        loss_keys = [k for k in keys if 'loss' in k]
        tagged_losses = ub.group_items(loss_keys, tag_grouper)
        tagged_losses.pop('unknown', None)
        kw = {}
        kw['ymin'] = 0.0
        # print('tagged_losses = {!r}'.format(tagged_losses))
        for tag, losses in tagged_losses.items():

            xydata = ub.odict()
            for key in sorted(losses):
                ydata = tb_data[key]['ydata']
                ydata = smooth_curve(ydata, smoothing)
                xydata[key] = (tb_data[key]['xdata'], ydata)

            if ignore_outliers:
                low, kw['ymax'] = inlier_ylim([t[1] for t in xydata.values()])

            fig.clf()
            ax = fig.gca()
            title = nice + '\n' + tag + '_' + mode + ' losses'
            nh.util.multi_plot(xydata=xydata, ylabel='loss', xlabel=mode,
                               # yscale='symlog',
                               title=title, fnum=1, ax=ax,
                               **kw)

            # png is slightly smaller than jpg for this kind of plot
            fpath = join(out_dpath, tag + '_' + mode + '_multiloss.png')
            # print('fpath = {!r}'.format(fpath))
            ax.figure.savefig(fpath)

        # don't dump losses individually if we dump them in a group
        GROUP_AND_INDIVIDUAL = False
        if not GROUP_AND_INDIVIDUAL:
            keys.difference_update(set(loss_keys))
            # print('keys = {!r}'.format(keys))

    INDIVIDUAL_PLOTS = True
    if INDIVIDUAL_PLOTS:
        # print('keys = {!r}'.format(keys))
        for key in keys:
            d = tb_data[key]

            ydata = d['ydata']
            ydata = smooth_curve(ydata, smoothing)

            kw = {}
            if any(m.lower() in key.lower() for m in y01_measures):
                kw['ymin'] = 0.0
                kw['ymax'] = 1.0
            elif any(m.lower() in key.lower() for m in y0_measures):
                kw['ymin'] = 0.0
                if ignore_outliers:
                    low, kw['ymax'] = inlier_ylim([ydata])

            # NOTE: this is actually pretty slow
            fig.clf()
            ax = fig.gca()
            title = nice + '\n' + key
            nh.util.multi_plot(d['xdata'], ydata, ylabel=key,
                               xlabel=mode, title=title, fnum=1, ax=ax, **kw)

            # png is slightly smaller than jpg for this kind of plot
            fpath = join(out_dpath, key + '.png')
            # print('save fpath = {!r}'.format(fpath))
            ax.figure.savefig(fpath)
