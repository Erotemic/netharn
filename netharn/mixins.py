"""
Under development! Function names and logic may change at any time. Nothing in
this file should be considered as stable! Use at your own risk.

These are methods that you can mixin to your FitHarn implementation to extend
its functionality to typical, but non-default cases.

The purpose of this file is to contain functions that might not general-purpose
enough to add to FitHarn itself, but they are also common enough, where it
makes no sense to write them from scratch for each new project.
"""


def _dump_monitor_tensorboard(harn):
    """
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

    harn.debug('Plotting tensorboard data')

    tb_data = nh.util.read_tensorboard_scalars(harn.train_dpath, cache=0,
                                               verbose=0)

    plot_keys = [key for key in tb_data if
                 ('train_epoch' in key or
                  'vali_epoch' in key or
                  'test_epoch' in key or
                  # 'epoch_lr' in key)
                  'epoch_' in key)]
    y01_measures = ['_acc', '_ap', '_mAP', '_auc', '_mcc', '_brier', '_mauc']
    y0_measures = ['error', 'loss']

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

    nh.util.autompl()
    keys = set(tb_data.keys()).intersection(set(plot_keys))
    for key in keys:
        d = tb_data[key]
        kw = {}
        if any(m.lower() in key.lower() for m in y01_measures):
            kw['ymin'] = 0.0
            kw['ymax'] = 1.0
        elif any(m.lower() in key.lower() for m in y0_measures):
            kw['ymin'] = 0.0
        ax = nh.util.multi_plot(d['xdata'], d['ydata'], ylabel=key,
                                xlabel='epoch', title=key, fnum=1, doclf=True,
                                **kw)

        # png is slightly smaller than jpg for this kind of plot
        fpath = join(out_dpath, key + '.png')
        ax.figure.savefig(fpath)
