"""
Checks to ensure that the iteration index is maintained properly (before
version 0.5.3 it wasn't).
"""
import numpy as np
import torch.nn.functional as F


import ubelt as ub
import netharn as nh
import torch


class Failpoint(Exception):
    pass


class MyHarn(nh.FitHarn):

    def after_initialize(harn):
        harn._all_iters = ub.ddict(list)
        pass

    def prepare_batch(harn, raw_batch):
        return raw_batch

    def before_epochs(harn):
        # change the size of the dataset every epoch
        harn.datasets['train'].total = (harn.epoch % 10) + 1

    def run_batch(harn, raw_batch):

        if harn.epoch == harn.failpoint:
            raise Failpoint

        harn._all_iters[harn.current_tag].append(harn.iter_index)

        batch = harn.xpu.move(raw_batch)

        output = harn.model(batch['im'])

        log_probs = F.log_softmax(output, dim=1)
        loss_parts = {
            'nll_loss': F.nll_loss(log_probs, batch['label']),
        }
        return output, loss_parts


class VariableSizeDataset(torch.utils.data.Dataset):

    def __init__(self, total=100):
        self.total = total
        self.subdata = nh.data.ToyData2d(
            size=3, border=1, n=1000, rng=0)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        index = index % len(self.subdata)
        image, label = self.subdata[index]
        item = {
            'im': image,
            'label': label,
        }
        return item


def test_iter_idx():
    """
    main test function
    """
    datasets = {
        'train': VariableSizeDataset(total=1),
        'vali': VariableSizeDataset(total=10),
    }
    model = nh.models.ToyNet2d()

    hyper = {
        # --- data first
        'datasets'    : datasets,
        'nice'        : 'test_iter_idx',
        'workdir'     : ub.ensure_app_cache_dir('netharn/test/test_iter_idx'),
        'loaders'     : {'batch_size': 1},
        'xpu'         : nh.XPU.coerce('cpu'),
        # --- algorithm second
        'model'       : model,
        'optimizer'   : nh.api.Optimizer.coerce({'optim': 'sgd'}),
        'initializer' : nh.api.Initializer.coerce({'init': 'noop'}),
        'scheduler'  : nh.api.Scheduler.coerce({'scheduler': 'step-3-7'}),
        'dynamics'   : nh.api.Dynamics.coerce({'batch_step': 1}),
        'monitor'    : (nh.Monitor, {'max_epoch': 10}),
    }
    harn1 = MyHarn(hyper=hyper)
    harn1.preferences['use_tensorboard'] = True
    harn1.preferences['eager_dump_tensorboard'] = True

    harn1.intervals['log_iter_train'] = 1
    harn1.intervals['log_iter_vali'] = 1
    harn1.intervals['cleanup'] = 5
    # Delete previous data
    harn1.initialize(reset='delete')

    # Cause the harness to fail
    try:
        harn1.failpoint = 5
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
        python ~/code/netharn/tests/test_iter_idx.py
    """
    # import warnings
    # warnings.filterwarnings('error')
    test_iter_idx()
