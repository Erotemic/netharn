import ubelt as ub
import netharn as nh


class MyHarn(nh.FitHarn):
    def _run_epoch(harn, loader, tag, learn=False):
        if harn.epoch == harn.failpoint:
            raise Exception('Stop Training')
        # Overload run_epoch to do nothing
        epoch_metrics = {'loss': 3}
        return epoch_metrics


def test_restart_lr():
    size = 3
    max_epoch = 100
    datasets = {
        'train': nh.data.ToyData2d(size=size, border=1, n=256, rng=0),
        'vali': nh.data.ToyData2d(size=size, border=1, n=128, rng=1),
    }

    lr = 0.0001

    hyper = {
        # --- data first
        'datasets'    : datasets,
        'nice'        : 'restart_lr',
        'workdir'     : ub.ensure_app_cache_dir('netharn/test/restart_lr'),
        'loaders'     : {'batch_size': 64},
        'xpu'         : nh.XPU.cast('cpu'),
        # --- algorithm second
        'model'       : (nh.models.ToyNet2d, {}),
        'optimizer'   : (nh.optimizers.SGD, {
            'lr': lr / 10,
            'momentum': 0.9,
        }),
        'criterion'   : (nh.criterions.FocalLoss, {}),
        'initializer' : (nh.initializers.KaimingNormal, {
            'param': 0,
        }),
        'scheduler': (nh.schedulers.ListedLR, {
            'points': {
                0:  lr / 10,
                1:  lr,
                59: lr * 1.1,
                60: lr / 10,
                90: lr / 100,
            },
            'interpolate': True
        }),
        'dynamics'   : {'batch_step': 4},
        'monitor'     : (nh.Monitor, {
            'max_epoch': max_epoch,
        }),
    }
    harn = MyHarn(hyper=hyper)
    harn.config['use_tqdm'] = 0
    # Delete previous data
    harn.initialize(reset='delete')

    # Cause the harness to fail
    try:
        harn.failpoint = 30
        harn.run()
    except Exception as ex:
        pass
    failpoint_lrs = harn._current_lrs()

    # Restarting the harness should begin at the same point
    harn = MyHarn(hyper=hyper)
    harn.config['use_tqdm'] = 0
    harn.initialize()

    restart_lrs = harn._current_lrs()
    print('failpoint_lrs = {!r}'.format(failpoint_lrs))
    print('restart_lrs = {!r}'.format(restart_lrs))

    # harn.failpoint = 60
    # harn.run()
