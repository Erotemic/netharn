# -*- coding: utf-8 -*-
"""
fit_harness takes your hyperparams and
applys standardized "state-of-the-art" training procedures

But everything is overwritable.
Experimentation and freedom to protype quickly is extremely important
We do our best not to get in the way, just performing a jumping off
point.

TODO:
    TrainingModes:
        [x] categorical
            see demos on:
                [x] MNIST
                [.] Cifar100
                [ ] ImageNet
                [ ] ...
        [ ] segmentation
            [ ] semantic
                [ ] CamVid
                [ ] CityScapes
                [ ] Diva
                [ ] UrbanMapper3D
                [ ] ...
            [ ] instance
                [ ] UrbanMapper3D
        [ ] tracking
            [ ] ...
        [ ] detection
            [ ] ...
            [ ] VOC2007
        [ ] identification
            [ ] 1-vs-all
            [ ] N-vs-all
            [ ] (1-vs-1) pairwise
            [ ] (N-vs-N)
            [ ] ...
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
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
    def __init__(self, n_channels=1, n_classes=10):
        super(MnistNet, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

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
        inputs = harn.xpu.variable(inputs)
        labels = harn.xpu.variable(labels)
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

        bx = harn.bxs[harn.current_tag]
        if bx < 3:
            decoded = harn._decode(outputs, batch['label'])
            stacked = harn._draw_batch(bx, batch, decoded)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'epoch_{}_batch_{}.jpg'.format(harn.epoch, bx))
            nh.util.imwrite(fpath, stacked)

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
            hot = nh.criterions.focal.one_hot_embedding(true_cxs, class_probs.shape[1])
            true_probs = (hot * class_probs).sum(dim=1)
            decoded['true_scores'] = true_probs
        return decoded

    def _draw_batch(harn, bx, batch, decoded, limit=32):
        """
        CommandLine:
            xdoctest -m ~/code/netharn/examples/cifar.py CIFAR_FitHarn._draw_batch --show --arch=wrn_22

        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/netharn/examples')
            >>> from mnist import *
            >>> harn = setup_mnist_harn().initialize()
            >>> #
            >>> batch = harn._demo_batch(0, tag='test')
            >>> outputs, loss = harn.run_batch(batch)
            >>> bx = harn.bxs[harn.current_tag]
            >>> decoded = harn._decode(outputs, batch['label'])
            >>> fpath = harn._draw_batch(bx, batch, decoded, limit=42)
            >>> print('fpath = {!r}'.format(fpath))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import netharn as nh
            >>> nh.util.autompl()
            >>> nh.util.imshow(fpath, colorspace='rgb', doclf=True)
            >>> nh.util.show_if_requested()
        """
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
        catgraph = dset.categories

        pred_cxs = decoded['pred_cxs'].data.cpu().numpy()
        pred_scores = decoded['pred_scores'].data.cpu().numpy()

        true_cxs = batch['label'].data.cpu().numpy()
        true_scores = decoded['true_scores'].data.cpu().numpy()

        todraw = []
        for im, pcx, tcx, pred_score, true_score in zip(inputs, pred_cxs, true_cxs, pred_scores, true_scores):
            im_ = im.transpose(1, 2, 0)
            im_ = nh.util.convert_colorspace(im_, 'gray', 'rgb')
            im_ = np.ascontiguousarray(im_)
            h, w = im_.shape[0:2][::-1]

            true_name = catgraph[tcx]
            pred_name = catgraph[pcx]
            org1 = np.array((2, h - 32))
            org2 = np.array((2, 25))
            pred_label = 'p:{pcx}@{pred_score:.2f}:\n{pred_name}'.format(**locals())
            true_label = 't:{tcx}@{true_score:.2f}:\n{true_name}'.format(**locals())
            if pcx == tcx:
                true_label = 't:{tcx}:{true_name}'.format(**locals())

            fontkw = {
                'fontScale': 1.0,
                'thickness': 2
            }
            color = 'dodgerblue' if pcx == tcx else 'orangered'

            im_ = nh.util.draw_text_on_image(im_, pred_label, org=org1 - 2,
                                             color='white', **fontkw)
            im_ = nh.util.draw_text_on_image(im_, true_label, org=org2 - 2,
                                             color='white', **fontkw)

            for i in [-2, -1, 1, 2]:
                for j in [-2, -1, 1, 2]:
                    im_ = nh.util.draw_text_on_image(im_, pred_label, org=org1 + i,
                                                     color='black', **fontkw)
                    im_ = nh.util.draw_text_on_image(im_, true_label, org=org2 + j,
                                                     color='black', **fontkw)

            im_ = nh.util.draw_text_on_image(im_, pred_label, org=org1,
                                             color=color, **fontkw)
            im_ = nh.util.draw_text_on_image(im_, true_label, org=org2,
                                             color='lawngreen', **fontkw)
            todraw.append(im_)

        stacked = nh.util.stack_images_grid(todraw, overlap=-10, bg_value=(10, 40, 30), chunksize=8)
        return stacked


def setup_mnist_harn():
    """
    CommandLine:
        python examples/mnist.py

        python ~/code/netharn/examples/mnist.py --gpu=2
        python ~/code/netharn/examples/mnist.py
    """
    root = os.path.expanduser('~/data/mnist/')

    # Define your dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    learn_dset = torchvision.datasets.MNIST(root, transform=transform,
                                            train=True, download=True)

    test_dset = torchvision.datasets.MNIST(root, transform=transform,
                                           train=True, download=True)

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

    categories = ['zero', 'one', 'two', 'three', 'four', 'five', 'six',
                  'seven', 'eight', 'nine']

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }
    for tag, dset in datasets.items():
        dset.categories = categories

    # Give the training dataset an input_id
    datasets['train'].input_id = 'mnist_' + ub.hash_data(train_idx.numpy())[0:8]

    batch_size = 128
    n_classes = 10
    xpu = nh.XPU.from_argv(min_memory=300)

    if False:
        initializer = (nh.initializers.Pretrained, {
            'fpath': 'path/to/pretained/weights.pt'
        })
    else:
        initializer = (nh.initializers.KaimingNormal, {})

    loaders = ub.odict()
    data_kw = {'batch_size': batch_size}
    if xpu.is_gpu():
        data_kw.update({'num_workers': 6, 'pin_memory': True})
    for tag in ['train', 'vali', 'test']:
        if tag not in datasets:
            continue
        dset = datasets[tag]
        shuffle = tag == 'train'
        data_kw_ = data_kw.copy()
        loader = torch.utils.data.DataLoader(dset, shuffle=shuffle, **data_kw_)
        loaders[tag] = loader

    # Workaround deadlocks with DataLoader
    import cv2
    cv2.setNumThreads(0)

    # Here is the FitHarn magic.
    # They nh.HyperParams object keeps track of and helps log all declarative
    # info related to training a model.
    hyper = nh.hyperparams.HyperParams(
        nice='mnist',
        xpu=xpu,
        workdir=ub.truepath('~//work/mnist/'),
        datasets=datasets,
        loaders=loaders,
        model=(MnistNet, dict(n_channels=1, n_classes=n_classes)),
        # optimizer=torch.optim.Adam,
        optimizer=(torch.optim.SGD, {'lr': 0.01}),
        scheduler='ReduceLROnPlateau',
        criterion=torch.nn.CrossEntropyLoss,
        initializer=initializer,
        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['acc'],
            'patience': 10,
            'max_epoch': 300,
            'smoothing': .4,
        }),
        other={
            # record any other information that will be used to compare
            # different training runs here
            'n_classes': n_classes,
        }
    )

    harn = MnistHarn(hyper=hyper)

    # Set how often vali / test will be run
    harn.intervals.update({
        # 'vali': slice(5, None, 1),

        # Start testing after the 5th epoch and then test every 4 epochs
        'test': slice(5, None, 4),
    })
    return harn


def train_mnist():
    harn = setup_mnist_harn()
    reset = ub.argflag('--reset')

    # Initializing a FitHarn object can take a little time, but not too much.
    # This is where instances of the model, optimizer, scheduler, monitor, and
    # initializer are created. This is also where we check if there is a
    # pre-existing checkpoint that we can restart from.
    harn.initialize(reset=reset)

    if ub.argval(('--vd', '--view-directory')):
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
        python examples/mnist.py

        tensorboard --logdir ~/data/work/mnist/fit/nice
    """
    train_mnist()
