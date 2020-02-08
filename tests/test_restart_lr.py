import ubelt as ub
import netharn as nh


class Failpoint(Exception):
    pass


class MyHarn(nh.FitHarn):
    def _run_epoch(harn, loader, tag, learn=False):
        if harn.epoch == harn.failpoint:
            raise Failpoint
        # Overload run_epoch to do nothing
        epoch_metrics = {'loss': 3}
        return epoch_metrics


def test_restart_lr():
    size = 3
    datasets = {
        'train': nh.data.ToyData2d(size=size, border=1, n=256, rng=0),
        'vali': nh.data.ToyData2d(size=size, border=1, n=128, rng=1),
    }

    lr = 1.0

    hyper = {
        # --- data first
        'datasets'    : datasets,
        'nice'        : 'restart_lr',
        'workdir'     : ub.ensure_app_cache_dir('netharn/test/restart_lr'),
        'loaders'     : {'batch_size': 64},
        'xpu'         : nh.XPU.coerce('cpu'),
        # --- algorithm second
        'model'       : (nh.models.ToyNet2d, {}),
        'optimizer'   : (nh.optimizers.SGD, {'lr': 99}),
        'criterion'   : (nh.criterions.FocalLoss, {}),
        'initializer' : (nh.initializers.NoOp, {}),
        'scheduler': (nh.schedulers.ListedLR, {
            'points': {
                0:  lr * 0.10,
                1:  lr * 1.00,
                9:  lr * 1.10,
                10: lr * 0.10,
                13: lr * 0.01,
            },
            'interpolate': True
        }),
        'dynamics'   : {'batch_step': 4},
        'monitor'    : (nh.Monitor, {'max_epoch': 13}),
    }
    harn = MyHarn(hyper=hyper)
    harn.preferences['prog_backend'] = 'progiter'
    harn.preferences['use_tensorboard'] = False
    # Delete previous data
    harn.initialize(reset='delete')

    # Cause the harness to fail
    try:
        harn.failpoint = 5
        harn.run()
    except Failpoint as ex:
        pass
    failpoint_lrs = set(harn._current_lrs())

    # Restarting the harness should begin at the same point
    harn = MyHarn(hyper=hyper)
    harn.preferences['prog_backend'] = 'progiter'
    harn.preferences['use_tensorboard'] = False
    harn.initialize()

    restart_lrs = set(harn._current_lrs())
    print('failpoint_lrs = {!r}'.format(failpoint_lrs))
    print('restart_lrs = {!r}'.format(restart_lrs))

    harn.failpoint = None
    harn.run()

    assert restart_lrs == failpoint_lrs

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/tests/test_restart_lr.py
    """
    import warnings
    warnings.filterwarnings('error')

    test_restart_lr()
