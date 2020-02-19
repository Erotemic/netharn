import ubelt as ub
import netharn as nh
import torch


class Failpoint(Exception):
    pass


class MyHarn(nh.FitHarn):

    def after_initialize(harn):
        harn.xdata = []
        harn.ydata = ub.ddict(list)

    def prepare_batch(harn, *a, **kw):
        if harn.epoch == harn.failpoint:
            raise Failpoint
        return super(MyHarn, harn).prepare_batch(*a, **kw)

    def run_batch(harn, batch):
        loss = torch.rand((1,), requires_grad=True)
        return None, loss

    # def on_epoch(harn):
    #     harn.xdata.append(harn.epoch)
    #     harn.ydata['lr'].append(list(harn._current_lrs())[0])

    def on_batch(harn, batch, outputs, loss):
        if harn.current_tag != 'train':
            return

        frac = harn.bxs[harn.current_tag] / len(harn.loaders[harn.current_tag])
        epoch_ = harn.epoch + frac
        if harn.epoch == harn.failpoint and frac > .5:
            raise Failpoint
        harn.xdata.append(epoch_)
        harn.ydata['lr'].append(list(harn._current_lrs())[0])

        # harn._ensure_prog_newline()
        # print('current lrs: {}'.format(harn._current_lrs()))
        # print('harn.scheduler.__dict__ = {!r}'.format(harn.scheduler.__dict__))
        return {}


def test_yolo_lr():
    if 0:
        datasets = {
            'train': nh.data.ToyData2d(size=3, border=1, n=18, rng=0),
            # 'vali': nh.data.ToyData2d(size=3, border=1, n=16, rng=1),
        }
        burn_in = 2.5
        lr = 0.1
        bstep = 2
        bsize = 2
        decay = 0.0005
        simulated_bsize = bstep * bsize
        max_epoch = 4
        points = {
            0: lr * 1.0,
            3: lr * 1.0,
            4: lr * 0.1,
        }
    else:
        datasets = {
            'train': nh.data.ToyData2d(size=3, border=1, n=16551 // 100, rng=0),
            'vali': nh.data.ToyData2d(size=3, border=1, n=4952 // 100, rng=1),
        }
        # number of epochs to burn_in for. approx 1000 batches?
        burn_in = 3.86683584
        lr = 0.001
        bstep = 2
        bsize = 32
        decay = 0.0005
        simulated_bsize = bstep * bsize
        max_epoch = 311
        points = {
            0:   lr * 1.0 / simulated_bsize,
            154: lr * 1.0 / simulated_bsize,  # 1.5625e-05
            155: lr * 0.1 / simulated_bsize,  # 1.5625e-06
            232: lr * 0.1 / simulated_bsize,
            233: lr * 0.01 / simulated_bsize,  # 1.5625e-07
        }

    hyper = {
        # --- data first
        'datasets'    : datasets,
        'nice'        : 'restart_lr',
        'workdir'     : ub.ensure_app_cache_dir('netharn/test/restart_lr'),
        'loaders'     : {'batch_size': bsize},
        'xpu'         : nh.XPU.coerce('cpu'),
        # --- algorithm second
        'model'       : (nh.models.ToyNet2d, {}),
        'optimizer'   : (nh.optimizers.SGD, {
            'lr': points[0],
            'weight_decay': decay * simulated_bsize
        }),
        'criterion'   : (nh.criterions.FocalLoss, {}),
        'initializer' : (nh.initializers.NoOp, {}),
        'scheduler': (nh.schedulers.YOLOScheduler, {
            'points': points,
            'burn_in': burn_in,
            'dset_size': len(datasets['train']),
            'batch_size': bsize,
            'interpolate': False,
        }),
        'dynamics'   : {'batch_step': bstep},
        'monitor'    : (nh.Monitor, {'max_epoch': max_epoch}),
    }
    harn = MyHarn(hyper=hyper)
    harn.preferences['prog_backend'] = 'progiter'
    harn.preferences['use_tensorboard'] = False
    # Delete previous data
    harn.initialize(reset='delete')

    # Cause the harness to fail
    try:
        harn.failpoint = 100
        harn.run()
    except Failpoint:
        pass
    print('\nFAILPOINT REACHED\n')
    failpoint_lrs = set(harn._current_lrs())

    old_harn = harn

    # Restarting the harness should begin at the same point
    harn = MyHarn(hyper=hyper)
    harn.preferences['prog_backend'] = 'progiter'
    harn.preferences['use_tensorboard'] = False
    harn.initialize()
    harn.xdata = old_harn.xdata
    harn.ydata = old_harn.ydata

    restart_lrs = set(harn._current_lrs())
    print('failpoint_lrs = {!r}'.format(failpoint_lrs))
    print('restart_lrs   = {!r}'.format(restart_lrs))

    harn.failpoint = None
    harn.run()

    if ub.argflag('--show'):
        import kwplot
        kwplot.autompl()
        kwplot.multi_plot(harn.xdata, harn.ydata)
        from matplotlib import pyplot as plt
        plt.show()

    assert restart_lrs == failpoint_lrs

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/tests/test_yolo_lr.py --show --profile
    """
    # import warnings
    # warnings.filterwarnings('error')
    test_yolo_lr()
