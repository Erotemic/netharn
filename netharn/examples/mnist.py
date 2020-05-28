# -*- coding: utf-8 -*-
"""
fit_harness takes your hyperparams and
applys standardized "state-of-the-art" training procedures

But everything is overwritable.
Experimentation and freedom to protype quickly is extremely important.
We do our best not to get in the way, just performing a jumping off point.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub
import torch
import torch.nn
import torchvision  # NOQA
import torch.nn.functional as F
from torch import nn
from os.path import join
import netharn as nh
import numpy as np


class MnistNet(nn.Module):
    def __init__(self, classes, num_channels=1):
        super(MnistNet, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, len(self.classes))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MnistHarn(nh.FitHarn):
    """
    Customize relevant parts of the training loop
    """

    def prepare_batch(harn, raw_batch):
        """
        Ensure batch is in a standardized structure
        """
        # Simply move the data from the datasets on the the GPU(s)
        inputs, labels = raw_batch
        inputs = harn.xpu.move(inputs)
        labels = harn.xpu.move(labels)
        batch = {
            'input': inputs,
            'label': labels,
        }
        return batch

    def run_batch(harn, batch):
        """ Core learning / backprop """
        inputs = batch['input']
        labels = batch['label']

        outputs = harn.model(inputs)

        loss = harn.criterion(outputs, labels)
        return outputs, loss

    def on_batch(harn, batch, outputs, loss):
        """ Compute relevent metrics to monitor """
        class_probs = torch.nn.functional.softmax(outputs, dim=1)
        scores, pred = class_probs.max(dim=1)

        pred_labels = pred.cpu().numpy()
        true_labels = batch['label'].cpu().numpy()

        if harn.batch_index < 3:
            import kwimage
            decoded = harn._decode(outputs, batch['label'])
            stacked = harn._draw_batch(batch, decoded)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'epoch_{}_batch_{}.jpg'.format(harn.epoch, harn.batch_index))
            kwimage.imwrite(fpath, stacked)

        acc = (true_labels == pred_labels).mean()

        metrics_dict = {
            'acc': acc,
        }
        return metrics_dict

    def _decode(harn, outputs, true_cxs=None):
        class_probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_scores, pred_cxs = class_probs.data.max(dim=1)

        decoded = {
            'class_probs': class_probs,
            'pred_cxs': pred_cxs,
            'pred_scores': pred_scores,
        }
        if true_cxs is not None:
            import kwarray
            hot = kwarray.one_hot_embedding(true_cxs, class_probs.shape[1])
            true_probs = (hot * class_probs).sum(dim=1)
            decoded['true_scores'] = true_probs
        return decoded

    def _draw_batch(harn, batch, decoded, limit=32):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--download)
            >>> harn = setup_harn().initialize()
            >>> #
            >>> batch = harn._demo_batch(0, tag='test')
            >>> outputs, loss = harn.run_batch(batch)
            >>> bx = harn.bxs[harn.current_tag]
            >>> decoded = harn._decode(outputs, batch['label'])
            >>> fpath = harn._draw_batch(bx, batch, decoded, limit=42)
            >>> print('fpath = {!r}'.format(fpath))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(fpath, colorspace='rgb', doclf=True)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        import kwplot
        inputs = batch['input']
        inputs = inputs[0:limit]

        input_shape = inputs.shape
        dims = [160] * (len(input_shape) - 2)
        min_, max_ = inputs.min(), inputs.max()
        inputs = (inputs - min_) / (max_ - min_)
        inputs = torch.nn.functional.interpolate(inputs, size=dims)
        inputs = (inputs * 255).byte()
        inputs = inputs.data.cpu().numpy()

        dset = harn.datasets[harn.current_tag]

        true_cxs = batch['label'].data.cpu().numpy()
        pred_cxs = decoded['pred_cxs'].data.cpu().numpy()
        class_probs = decoded['class_probs'].data.cpu().numpy()

        todraw = []
        for im, pcx, tcx, probs in zip(inputs, pred_cxs, true_cxs, class_probs):
            im_ = im.transpose(1, 2, 0)
            im_ = kwimage.convert_colorspace(im_, 'gray', 'rgb')
            im_ = np.ascontiguousarray(im_)
            im_ = kwplot.draw_clf_on_image(im_, dset.classes, tcx, probs)
            todraw.append(im_)

        stacked = kwimage.stack_images_grid(todraw, overlap=-10, bg_value=(10, 40, 30), chunksize=8)
        return stacked


def setup_datasets(workdir=None):
    if workdir is None:
        workdir = ub.expandpath('~/data/mnist/')

    # Define your dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    learn_dset = torchvision.datasets.MNIST(workdir, transform=transform,
                                            train=True, download=True)

    test_dset = torchvision.datasets.MNIST(workdir, transform=transform,
                                           train=False, download=True)

    # split the learning dataset into training and validation
    # take a subset of data
    factor = .15
    n_vali = int(len(learn_dset) * factor)
    learn_idx = np.arange(len(learn_dset))

    rng = np.random.RandomState(0)
    rng.shuffle(learn_idx)

    reduction = int(ub.argval('--reduction', default=1))
    vali_idx  = torch.LongTensor(learn_idx[:n_vali][::reduction])
    train_idx = torch.LongTensor(learn_idx[n_vali:][::reduction])

    train_dset = torch.utils.data.Subset(learn_dset, train_idx)
    vali_dset = torch.utils.data.Subset(learn_dset, vali_idx)

    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
               'eight', 'nine']

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }
    for tag, dset in datasets.items():
        dset.classes = classes
        dset.num_classes = len(classes)

    # Give the training dataset an input_id
    datasets['train'].input_id = 'mnist_' + ub.hash_data(train_idx.numpy())[0:8]
    return datasets, workdir


def setup_harn(**kw):
    """
    CommandLine:
        python examples/mnist.py

        python ~/code/netharn/examples/mnist.py --gpu=2
        python ~/code/netharn/examples/mnist.py
    """
    xpu = nh.XPU.from_argv(min_memory=300)

    config = {
        'batch_size': kw.get('batch_size', 128),
        'workers': kw.get('workers', 6 if xpu.is_gpu() else 0)
    }

    if config['workers'] > 0:
        # Workaround deadlocks with DataLoader
        import cv2
        cv2.setNumThreads(0)

    datasets, workdir = setup_datasets()

    loaders = {
        tag: torch.utils.data.DataLoader(
            dset, batch_size=config['batch_size'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'))
        for tag, dset in datasets.items()
    }

    if False:
        initializer = (nh.initializers.Pretrained, {
            'fpath': 'path/to/pretained/weights.pt'
        })
    else:
        initializer = (nh.initializers.KaimingNormal, {})

    # Here is the FitHarn magic.
    # They nh.HyperParams object keeps track of and helps log all declarative
    # info related to training a model.
    hyper = nh.hyperparams.HyperParams(
        name='my-mnist-demo',
        xpu=xpu,
        workdir=workdir,
        datasets=datasets,
        loaders=loaders,
        model=(MnistNet, dict(num_channels=1, classes=datasets['train'].classes)),
        # optimizer=torch.optim.AdamW,
        optimizer=(torch.optim.SGD, {'lr': 0.01, 'weight_decay': 3e-6}),
        # scheduler='ReduceLROnPlateau',
        scheduler=(nh.schedulers.ListedScheduler, {
            'points': {
                'lr': {
                    0   : 0.01,
                    2   : 0.05,
                    10  : 0.10,
                    20  : 0.01,
                    40  : 0.0001,
                },
                'momentum': {
                    0   : 0.95,
                    10  : 0.85,
                    20  : 0.95,
                    40  : 0.99,
                },
                'weight_decay': {
                    0: 3e-6,
                }
            },
            'interpolation': 'linear',
        }),
        criterion=torch.nn.CrossEntropyLoss,
        initializer=initializer,
        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['acc'],
            'patience': 10,
            'max_epoch': 300,
            'smoothing': .4,
        }),
    )

    harn = MnistHarn(hyper=hyper)
    harn.preferences.update({
        'keyboard_debug': True,
    })

    # Set how often vali / test will be run
    harn.intervals.update({
        # 'vali': slice(5, None, 1),
        # Start testing after the 5th epoch and then test every 4 epochs
        'test': slice(5, None, 4),
    })
    return harn


def main():
    harn = setup_harn()
    reset = ub.argflag('--reset')

    # Initializing a FitHarn object can take a little time, but not too much.
    # This is where instances of the model, optimizer, scheduler, monitor, and
    # initializer are created. This is also where we check if there is a
    # pre-existing checkpoint that we can restart from.
    harn.initialize(reset=reset)

    if ub.argflag(('--vd', '--view-directory')):
        ub.startfile(harn.train_dpath)

    # This starts the main loop which will run until a the monitor's terminator
    # criterion is satisfied. If the initialize step loaded a checkpointed that
    # already met the termination criterion, then this will simply return.
    deploy_fpath = harn.run()

    # The returned deploy_fpath is the path to an exported netharn model.
    # This model is the on with the best weights according to the monitor.
    print('deploy_fpath = {!r}'.format(deploy_fpath))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.examples.mnist

        tensorboard --logdir ~/data/work/mnist/fit/nice
    """
    main()
