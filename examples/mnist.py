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
import copy
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
            'inputs': inputs,
            'labels': labels,
        }
        return batch

    def run_batch(harn, batch):
        """ Core learning / backprop """
        inputs = batch['inputs']
        labels = batch['labels']

        outputs = harn.model(inputs)

        loss = harn.criterion(outputs, labels)
        return outputs, loss

    def _draw_batch(harn, bx, batch, pred_labels, true_labels, pred_scores, true_scores):
        import kwil
        input_shape = batch['inputs'].shape
        dims = [256] * (len(input_shape) - 2)
        inputs = batch['inputs']
        min_, max_ = inputs.min(), inputs.max()
        inputs = (inputs - min_) / (max_ - min_)
        inputs = torch.nn.functional.interpolate(inputs, size=dims)
        inputs = (inputs * 255).byte()
        inputs = inputs.data.cpu().numpy()

        # inputs = inputs[0:64]
        todraw = []
        for im, pcx, tcx, pred_score, true_score in zip(inputs, pred_labels, true_labels, pred_scores, true_scores):
            pred_label = 'cx={} @ {:.3f}'.format(pcx, pred_score)
            if pcx == tcx:
                true_label = 'tx={}'.format(tcx)
            else:
                true_label = 'tx={} @ {:.3f}'.format(tcx, true_score)
            im_ = kwil.convert_colorspace(im[0], 'gray', 'rgb')
            color = kwil.Color('dodgerblue') if pcx == tcx else kwil.Color('orangered')
            h, w = im_.shape[0:2][::-1]
            im_ = kwil.draw_text_on_image(im_, pred_label, org=(5, 32), fontScale=1.0, thickness=2, color=color.as255())
            im_ = kwil.draw_text_on_image(im_, true_label, org=(5, h - 4), fontScale=1.0, thickness=1, color=kwil.Color('lawngreen').as255())
            todraw.append(im_)

        dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
        fpath = join(dpath, 'epoch_{}_batch_{}.jpg'.format(harn.epoch, bx))
        stacked = kwil.stack_images_grid(todraw, overlap=-10, bg_value=(10, 40, 30), chunksize=16)
        kwil.imwrite(fpath, stacked)

    def on_batch(harn, batch, outputs, loss):
        """ Compute relevent metrics to monitor """
        class_probs = torch.nn.functional.softmax(outputs, dim=1)
        scores, pred = class_probs.max(dim=1)
        pred_labels = pred.cpu().numpy()

        true = batch['labels']
        hot = nh.criterions.focal.one_hot_embedding(true, class_probs.shape[1])
        true_probs = (hot * class_probs).sum(dim=1)

        true_labels = batch['labels'].cpu().numpy()
        pred_scores = scores.data.cpu().numpy()
        true_scores = true_probs.data.cpu().numpy()

        bx = harn.bxs[harn.current_tag]
        if bx == 0:
            harn._draw_batch(bx, batch, pred_labels, true_labels, pred_scores,
                             true_scores)

        acc = (true_labels == pred_labels).mean()

        metrics_dict = {
            'acc': acc,
        }
        return metrics_dict

    def after_epochs(harn):
        # Note: netharn.mixins is unstable functionality
        from netharn.mixins import _dump_monitor_tensorboard
        _dump_monitor_tensorboard(harn)


def train_mnist():
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

    train_dset = learn_dset
    vali_dset = copy.copy(learn_dset)

    # split the learning dataset into training and validation
    # take a subset of data
    # factor = .15
    factor = .15
    n_vali = int(len(learn_dset) * factor)
    learn_idx = np.arange(len(learn_dset))

    rng = np.random.RandomState(0)
    rng.shuffle(learn_idx)

    reduction = 1
    valid_idx = torch.LongTensor(learn_idx[:n_vali][::reduction])
    train_idx = torch.LongTensor(learn_idx[n_vali:][::reduction])

    def _torch_take(tensor, indices, axis):
        TensorType = learn_dset.train_data.type()
        TensorType = getattr(torch, TensorType.split('.')[1])
        return TensorType(tensor.numpy().take(indices, axis=axis))

    vali_dset.train_data   = _torch_take(learn_dset.train_data, valid_idx,
                                         axis=0)
    vali_dset.train_labels = _torch_take(learn_dset.train_labels, valid_idx,
                                         axis=0).long()

    train_dset.train_data   = _torch_take(learn_dset.train_data, train_idx,
                                          axis=0)
    train_dset.train_labels = _torch_take(learn_dset.train_labels, train_idx,
                                          axis=0).long()

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }

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

    """
    # Here is the FitHarn magic.
    # This keeps track of your stuff
    """
    hyper = nh.hyperparams.HyperParams(
        nice='mnist',
        xpu=xpu,
        workdir=ub.truepath('~//work/mnist/'),
        datasets=datasets,
        loaders=loaders,
        model=(MnistNet, dict(n_channels=1, n_classes=n_classes)),
        # optimizer=torch.optim.Adam,
        optimizer=(torch.optim.SGD, {'lr': 0.01}),
        # FIXME: the ReduceLROnPleateau is broken with restarts
        scheduler='ReduceLROnPlateau',
        criterion=torch.nn.CrossEntropyLoss,
        initializer=initializer,
        monitor=(nh.Monitor, {
            # 'minimize': ['loss'],
            # 'maximize': ['acc'],
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

    reset = ub.argflag('--reset')
    harn.initialize(reset=reset)
    harn.run()

    # if False:
    #     import plottool as pt
    #     pt.qtensure()
    #     ims, gts = next(iter(harn.loaders['train']))
    #     pic = im_loaders.rgb_tensor_to_imgs(ims, norm=False)[0]
    #     pt.clf()
    #     pt.imshow(pic, norm=True, cmap='viridis', data_colorbar=True)

    #     with pt.RenderingContext() as render:
    #         tensor_data = datasets['train'][0][0][None, :]
    #         pic = im_loaders.rgb_tensor_to_imgs(tensor_data, norm=False)[0]
    #         pt.figure(fnum=1, doclf=True)
    #         pt.imshow(pic, norm=True, cmap='viridis', data_colorbar=True,
    #                   fnum=1)
    #     render.image

if __name__ == '__main__':
    r"""
    CommandLine:
        python examples/mnist.py

        tensorboard --logdir ~/data/work/mnist/fit/nice
    """
    train_mnist()
