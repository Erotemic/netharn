# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import netharn as nh


def mwe():
    # Assuming optimizer has two groups.
    import torch.optim.lr_scheduler
    import netharn as nh
    model = nh.models.ToyNet2d()
    optimizer = torch.optim.SGD(model.parameters(), lr=10)

    class DummySchedule(torch.optim.lr_scheduler._LRScheduler):
        def get_lr(self):
            print('Set LR based on self.last_epoch = {!r}'.format(self.last_epoch))
            self._current_lr = self.last_epoch
            return [self.last_epoch]

    # Initialize the optimizer with epoch 0's LR
    # self = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: x)
    self = DummySchedule(optimizer)
    for epoch in range(3):
        print('------')
        print('Run epoch = {!r}'.format(epoch))
        # Pretend we run epoch 0
        print('Training with self._current_lr = {!r}'.format(self._current_lr))
        lr = self.get_lr()[0]
        print('Training with lr = {!r}'.format(lr))
        # Pretend epoch 0 has finished, so step the scheduler.
        # self.step(epoch=epoch)
        self.step()


class MyHarn(nh.FitHarn):
    def __init__(self, *args, **kw):
        super(MyHarn, self).__init__(*args, **kw)
        self.epoch_to_lr = ub.odict()

    def _run_epoch(harn, loader, tag, learn=False):
        # Overload run_epoch to do nothing
        lrs = set(harn._current_lrs())
        print('* RUN WITH lrs = {!r}'.format(lrs))
        harn.epoch_to_lr[harn.epoch] = lrs
        epoch_metrics = {'loss': 3}
        return epoch_metrics


def test_lr():
    size = 3
    datasets = {
        'train': nh.data.ToyData2d(size=size, border=1, n=256, rng=0),
        # 'vali': nh.data.ToyData2d(size=size, border=1, n=128, rng=1),
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
                10: lr * 0.10,
                13: lr * 0.01,
            },
            'interpolate': False,
        }),
        'dynamics'   : {'batch_step': 4},
        'monitor'    : (nh.Monitor, {'max_epoch': 15}),
    }
    harn = MyHarn(hyper=hyper)
    harn.preferences['use_tqdm'] = 0
    # Delete previous data
    harn.initialize(reset='delete')

    # Cause the harness to fail
    harn.run()

    print(ub.repr2(harn.epoch_to_lr, nl=1))

    # restart_lrs = set(harn._current_lrs())
    # print('restart_lrs = {!r}'.format(restart_lrs))

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/tests/test_lr.py
    """
    test_lr()
