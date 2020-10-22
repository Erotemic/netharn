"""
Tests the order in which things happen in "run"
"""
# import torch.nn.functional as F
import numpy as np
import ubelt as ub
import netharn as nh
import torch


class Failpoint(Exception):
    pass


class MyHarn(nh.FitHarn):

    def run_batch(harn, raw_batch):
        if harn.epoch == harn.failpoint and harn.batch_index >= 4:
            raise Failpoint

        x = torch.Tensor([[1, 2]])
        f = torch.nn.Linear(2, 1)
        y = f(x)
        loss = y.sum()
        output = y

        # harn._all_iters[harn.current_tag].append(harn.iter_index)
        # batch = harn.xpu.move(raw_batch)
        # output = harn.model(batch['im'])
        # log_probs = F.log_softmax(output, dim=1)
        # loss_parts = {
        #     'nll_loss': F.nll_loss(log_probs, batch['label']),
        # }
        return output, loss


def test_run_sequence():
    """
    main test function
    """
    datasets = {
        'train': nh.data.ToyData2d(size=3, border=1, n=7, rng=0),
        'vali': nh.data.ToyData2d(size=3, border=1, n=3, rng=0),
    }
    model = nh.models.ToyNet2d()

    hyper = {
        # --- data first
        'datasets'    : datasets,
        'nice'        : 'test_run_sequence',
        'workdir'     : ub.ensure_app_cache_dir('netharn/test/test_run_sequence'),
        'loaders'     : {'batch_size': 1},
        'xpu'         : nh.XPU.coerce('cpu'),
        # --- algorithm second
        'model'       : model,
        'optimizer'   : nh.api.Optimizer.coerce({'optim': 'sgd'}),
        'initializer' : nh.api.Initializer.coerce({'init': 'noop'}),
        'scheduler'  : nh.api.Scheduler.coerce({'scheduler': 'step-3-7'}),
        'dynamics'   : nh.api.Dynamics.coerce({'batch_step': 1, 'warmup_iters': 6}),
        'monitor'    : (nh.Monitor, {'max_epoch': 4}),
    }
    harn1 = MyHarn(hyper=hyper)
    harn1.preferences['verbose'] = 1
    harn1.preferences['use_tensorboard'] = False
    harn1.preferences['eager_dump_tensorboard'] = False

    harn1.intervals['log_iter_train'] = 1
    harn1.intervals['log_iter_vali'] = 1
    harn1.intervals['cleanup'] = 5
    # Delete previous data
    harn1.initialize(reset='delete')

    # Cause the harness to fail
    try:
        harn1.failpoint = 0
        harn1.run()
    except Failpoint:
        pass
    print('\nFAILPOINT REACHED\n')

    # Restarting the harness should begin at the same point
    harn2 = MyHarn(hyper=hyper)
    harn2.preferences.update(harn1.preferences)
    harn2.intervals.update(harn1.intervals)
    harn2.failpoint = None
    harn2.run()

    if 0:
        idxs1 = harn1._all_iters['train']
        idxs2 = harn2._all_iters['train']
        diff1 = np.diff(idxs1)
        diff2 = np.diff(idxs2)
        print('idxs1 = {!r}'.format(idxs1))
        print('idxs2 = {!r}'.format(idxs2))
        print('diff1 = {!r}'.format(diff1))
        print('diff2 = {!r}'.format(diff2))
        assert np.all(diff1 == 1)
        assert np.all(diff2 == 1)
        assert idxs1[0] == 0
        assert idxs1[-1] == (idxs2[0] - 1)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/tests/test_run_sequence.py
    """
    test_run_sequence()
