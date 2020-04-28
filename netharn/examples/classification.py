# -*- coding: utf-8 -*-
"""
This is a simple generalized harness for training a classifier on a coco dataset.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import numpy as np
import sys
import torch
import ubelt as ub

import netharn as nh
import kwarray
import scriptconfig as scfg
from netharn.data.channel_spec import ChannelSpec


class ClfConfig(scfg.Config):
    """
    This is the default configuration for running the classification example
    """
    default = {
        'name': scfg.Value('clf_example', help='A human readable tag that is "name" for humans'),
        'workdir': scfg.Path('~/work/netharn', help='Dump all results in your workdir'),

        'workers': scfg.Value(2, help='number of parallel dataloading jobs'),
        'xpu': scfg.Value('auto', help='See netharn.XPU for details. can be auto/cpu/xpu/cuda0/0,1,2,3)'),

        'datasets': scfg.Value('special:shapes256', help='Either a special key or a coco file'),
        'train_dataset': scfg.Value(None),
        'vali_dataset': scfg.Value(None),
        'test_dataset': scfg.Value(None),

        'sampler_backend': scfg.Value(None, help='ndsampler backend'),

        'channels': scfg.Value('rgb', help='special channel code. See ChannelSpec'),

        'arch': scfg.Value('resnet50', help='Network architecture code'),
        'optim': scfg.Value('adam', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'input_dims': scfg.Value((224, 224), help='Window size to input to the network'),
        'normalize_inputs': scfg.Value(True, help=(
            'if True, precompute training mean and std for data whitening')),

        'balance': scfg.Value(None, help='balance strategy. Can be category or None'),

        'augmenter': scfg.Value('simple', help='type of training dataset augmentation'),

        'batch_size': scfg.Value(3, help='number of items per batch'),
        'num_batches': scfg.Value('auto', help='Number of batches per epoch'),

        'max_epoch': scfg.Value(140, help='Maximum number of epochs'),
        'patience': scfg.Value(140, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-4, help='Base learning rate'),
        'decay':  scfg.Value(1e-5, help='Base weight decay'),
        'schedule': scfg.Value(
            'step90-120', help=(
                'Special coercable netharn code. Eg: onecycle50, step50, gamma')),

        'init': scfg.Value('kaiming_normal', help='How to initialized weights. (can be a path to a pretrained model)'),
        'pretrained': scfg.Path(help=('alternative way to specify a path to a pretrained model')),
    }

    def normalize(self):
        if self['pretrained'] in ['null', 'None']:
            self['pretrained'] = None

        if self['pretrained'] is not None:
            self['init'] = 'pretrained'


class ClfModel(nh.layers.Module):
    """

    Example:
        >>> from netharn.examples.classification import *  # NOQA
        >>> self = ClfModel(channels='gray', classes=3)
        >>> inputs = torch.rand(4, 1, 256, 256)
        >>> outputs = self(inputs)
        >>> self.coder.decode_batch(outputs)
    """

    def __init__(self, arch='resnet50', classes=1000, channels='rgb',
                 input_stats=None):
        super(ClfModel, self).__init__()

        import ndsampler
        if input_stats is None:
            input_stats = {}
        input_norm = nh.layers.InputNorm(**input_stats)

        self.classes = ndsampler.CategoryTree.coerce(classes)

        self.channels = ChannelSpec.coerce(channels)
        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1
        in_channels = len(ub.peek(chann_norm.values()))
        num_classes = len(self.classes)

        if arch == 'resnet50':
            from torchvision import models
            model = models.resnet50()
            new_conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                        stride=3, padding=3, bias=False)
            new_fc = torch.nn.Linear(2048, num_classes, bias=True)
            new_conv1.weight.data[:, 0:in_channels, :, :] = model.conv1.weight.data[0:, 0:in_channels, :, :]
            new_fc.weight.data[0:num_classes, :] = model.fc.weight.data[0:num_classes, :]
            new_fc.bias.data[0:num_classes] = model.fc.bias.data[0:num_classes]
            model.fc = new_fc
            model.conv1 = new_conv1
        else:
            raise KeyError(arch)

        self.input_norm = input_norm
        self.model = model

        self.coder = ClfCoder(self.classes)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            assert len(inputs) == 1
            im = ub.peek(inputs.values())
        else:
            im = inputs

        im = self.input_norm(im)
        class_energy = self.model(im)
        outputs = {
            'class_energy': class_energy,
        }
        return outputs


class ClfCoder(object):
    """
    classifier coder, todo: expand
    """
    def __init__(self, classes):
        self.classes = classes

    def decode_batch(self, outputs):
        class_energy = outputs['class_energy']
        class_probs = self.classes.hierarchical_softmax(class_energy, dim=1)
        pred_cxs, pred_conf = self.classes.decision(
            class_probs, dim=1, thresh=0.1,
            criterion='entropy',
        )
        decoded = {
            'class_probs': class_probs,
            'pred_cxs': pred_cxs,
            'pred_conf': pred_conf,
        }
        return decoded


class ClfDataset(torch.utils.data.Dataset):
    """
    Efficient loader for classification training on coco samplers.


    Example:
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler.demo()
        >>> self = ClfDataset(sampler)
        >>> index = 0
        >>> self[index]['inputs']['rgb'].shape
        >>> loader = self.make_loader(batch_size=8, shuffle=True, num_workers=0, num_batches=10)
        >>> for batch in ub.ProgIter(iter(loader), total=len(loader)):
        >>>     break
        >>> print('batch = {}'.format(ub.repr2(batch, nl=1)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(batch['inputs']['rgb'][0])
    """
    def __init__(self, sampler, input_dims=(256, 256), augmenter=None):
        self.sampler = sampler
        self.augmenter = None
        self.conditional_augmentors = None
        self.input_dims = input_dims
        self.classes = self.sampler.catgraph

        self.augmenter = self._coerce_augmenter(augmenter)

    def __len__(self):
        return self.sampler.n_positives

    @ub.memoize_property
    def input_id(self):
        def imgaug_json_id(aug):
            import imgaug
            if isinstance(aug, tuple):
                return [imgaug_json_id(item) for item in aug]
            elif isinstance(aug, imgaug.parameters.StochasticParameter):
                return str(aug)
            else:
                try:
                    info = ub.odict()
                    info['__class__'] = aug.__class__.__name__
                    params = aug.get_parameters()
                    if params:
                        info['params'] = [imgaug_json_id(p) for p in params]
                    if isinstance(aug, list):
                        children = aug[:]
                        children = [imgaug_json_id(c) for c in children]
                        info['children'] = children
                    return info
                except Exception:
                    # imgaug is weird and buggy
                    return str(aug)
        depends = [
            self.sampler._depends(),
            self.augmenter and imgaug_json_id(self.augmenter),
        ]
        _input_id = ub.hash_data(depends, hasher='sha512', base='abc')[0:40]
        return _input_id

    def __getitem__(self, index):
        import kwimage

        # Load sample image and category
        sample = self.sampler.load_positive(index, with_annots=False)
        image = kwimage.atleast_3channels(sample['im'])[:, :, 0:3]
        target = sample['tr']

        image = kwimage.ensure_uint255(image)
        if self.augmenter is not None:
            det = self.augmenter.to_deterministic()
            image = det.augment_image(image)

        # Resize to input dimensinos
        if self.input_dims is not None:
            dsize = tuple(self.input_dims[::-1])
            image = kwimage.imresize(image, dsize=dsize, letterbox=True)

        class_id_to_idx = self.sampler.classes.id_to_idx
        cid = target['category_id']
        cidx = class_id_to_idx[cid]

        im_chw = image.transpose(2, 0, 1) / 255.0
        inputs = {
            'rgb': torch.FloatTensor(im_chw),
        }
        labels = {
            'class_idxs': cidx,
        }
        batch = {
            'inputs': inputs,
            'labels': labels,
        }
        return batch

    def _coerce_augmenter(self, augmenter):
        import netharn as nh
        import imgaug.augmenters as iaa
        if augmenter is True:
            augmenter = 'simple'
        if not augmenter:
            augmenter = None
        elif augmenter == 'simple':
            augmenter = iaa.Sequential([
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        elif augmenter == 'medium':
            augmenter = iaa.Sequential([
                iaa.Sometimes(0.2, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        else:
            raise KeyError('Unknown augmentation {!r}'.format(self.augment))
        return augmenter

    def make_loader(self, batch_size=16, num_batches='auto', num_workers=0,
                    shuffle=False, pin_memory=False, drop_last=False,
                    balance=None):

        if len(self) == 0:
            raise Exception('must have some data')

        def worker_init_fn(worker_id):
            for i in range(worker_id + 1):
                seed = np.random.randint(0, int(2 ** 32) - 1)
            seed = seed + worker_id
            kwarray.seed_global(seed)
            if self.augmenter:
                rng = kwarray.ensure_rng(None)
                self.augmenter.seed_(rng)

        loaderkw = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'worker_init_fn': worker_init_fn,
        }
        if balance is None:
            loaderkw['shuffle'] = shuffle
            loaderkw['batch_size'] = batch_size
            loaderkw['drop_last'] = drop_last
        elif balance == 'classes':
            from netharn.data.batch_samplers import BalancedBatchSampler
            index_to_cid = [
                cid for cid in self.sampler.regions.targets['category_id']
            ]
            batch_sampler = BalancedBatchSampler(
                index_to_cid, batch_size=batch_size,
                shuffle=shuffle, num_batches=num_batches)
            loaderkw['batch_sampler'] = batch_sampler
        else:
            raise KeyError(balance)

        loader = torch.utils.data.DataLoader(self, **loaderkw)
        return loader


class ClfHarn(nh.FitHarn):
    """
    The Harness provides the important details to the training loop via the
    `run_batch` method. The rest of the loop boilerplate is taken care of by
    `nh.FitHarn` internals. In addition to `run_batch`, we also define several
    callbacks to perform customized monitoring of training progress.
    """

    def after_initialize(harn, **kw):
        harn._accum_confusion_vectors = {
            'y_true': [],
            'y_pred': [],
            'probs': [],
        }

    def prepare_batch(harn, raw_batch):
        return raw_batch

    def run_batch(harn, batch):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> harn = setup_harn(datasets='special:shapes256', batch_size=4).initialize()
            >>> batch = harn._demo_batch(0, tag='train')
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, loss)
        """
        classes = harn.raw_model.classes
        inputs = harn.xpu.move(batch['inputs'])
        labels = harn.xpu.move(batch['labels'])

        outputs = harn.model(inputs)

        class_energy = outputs['class_energy']
        class_logprobs = classes.hierarchical_log_softmax(
            class_energy, dim=1)

        class_idxs = labels['class_idxs']
        loss = nh.criterions.focal.nll_focal_loss(
            class_logprobs, class_idxs, focus=2.0, reduction='mean')

        loss_parts = {}
        loss_parts['clf'] = loss

        decoded = harn.raw_model.coder.decode_batch(outputs)

        outputs['class_probs'] = decoded['class_probs']
        outputs['pred_cxs'] = decoded['pred_cxs']
        outputs['true_cxs'] = class_idxs
        return outputs, loss_parts

    def on_batch(harn, batch, outputs, loss):
        """
        Custom code executed at the end of each batch.
        """
        bx = harn.bxs[harn.current_tag]
        if bx < 3:
            stacked = harn._draw_batch(batch, outputs)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'batch_{}_epoch_{}.jpg'.format(bx, harn.epoch))
            import kwimage
            kwimage.imwrite(fpath, stacked)

        y_pred = kwarray.ArrayAPI.numpy(outputs['pred_cxs'])
        y_true = outputs['true_cxs'].data.cpu().numpy()
        probs = outputs['class_probs'].data.cpu().numpy()
        harn._accum_confusion_vectors['y_true'].append(y_true)
        harn._accum_confusion_vectors['y_pred'].append(y_pred)
        harn._accum_confusion_vectors['probs'].append(probs)

    def _draw_batch(harn, batch, outputs, limit=32):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--download)
            >>> harn = setup_harn(batch_size=3).initialize()
            >>> batch = harn._demo_batch(0, tag='train')
            >>> outputs, loss = harn.run_batch(batch)
            >>> stacked = harn._draw_batch(batch, outputs, limit=12)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked, colorspace='rgb', doclf=True)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        inputs = batch['inputs']['rgb'][0:limit].data.cpu().numpy()
        true_cxs = batch['labels']['class_idxs'].data.cpu().numpy()
        class_probs = outputs['class_probs'].data.cpu().numpy()
        pred_cxs = kwarray.ArrayAPI.numpy(outputs['pred_cxs'])

        dset = harn.datasets[harn.current_tag]
        classes = dset.classes

        todraw = []
        for im, pcx, tcx, probs in zip(inputs, pred_cxs, true_cxs, class_probs):
            im_ = im.transpose(1, 2, 0)

            # Renormalize and resize image for drawing
            min_, max_ = im_.min(), im_.max()
            im_ = ((im_ - min_) / (max_ - min_) * 255).astype(np.uint8)
            im_ = np.ascontiguousarray(im_)
            im_ = kwimage.imresize(im_, dsize=(200, 200),
                                   interpolation='nearest')

            # Draw classification information on the image
            im_ = kwimage.draw_clf_on_image(im_, classes=classes, tcx=tcx,
                                            pcx=pcx, probs=probs)
            todraw.append(im_)

        stacked = kwimage.stack_images_grid(todraw, overlap=-10,
                                            bg_value=(10, 40, 30),
                                            chunksize=8)
        return stacked

    def on_epoch(harn):
        """
        Custom code executed at the end of each epoch.

        This function can optionally return a dictionary containing any scalar
        quality metrics that you wish to log and monitor. (Note these will be
        plotted to tensorboard if that is installed).

        Notes:
            It is ok to do some medium lifting in this function because it is
            run relatively few times.

        Example:
            >>> # xdoctest: +REQUIRES(--download)
            >>> harn = setup_harn().initialize()
            >>> harn._demo_epoch('vali')
        """
        from netharn.metrics import clf_report
        dset = harn.datasets[harn.current_tag]

        probs = np.vstack(harn._accum_confusion_vectors['probs'])
        y_true = np.hstack(harn._accum_confusion_vectors['y_true'])
        y_pred = np.hstack(harn._accum_confusion_vectors['y_pred'])

        # _pred = probs.argmax(axis=1)
        # assert np.all(_pred == y_pred)

        # from netharn.metrics import confusion_vectors
        # cfsn_vecs = confusion_vectors.ConfusionVectors.from_arrays(
        #     true=y_true, pred=y_pred, probs=probs, classes=dset.classes)
        # report = cfsn_vecs.classification_report()
        # combined_report = report['metrics'].loc['combined'].to_dict()

        # ovr_cfsn = cfsn_vecs.binarize_ovr()
        # Compute multiclass metrics (new way!)
        target_names = dset.classes
        ovr_report = clf_report.ovr_classification_report(
            y_true, probs, target_names=target_names, metrics=[
                'auc', 'ap', 'mcc', 'brier'
            ])

        # percent error really isn't a great metric, but its easy and standard.
        errors = (y_true != y_pred)
        acc = 1.0 - errors.mean()
        percent_error = (1.0 - acc) * 100

        metrics_dict = ub.odict()
        metrics_dict['ave_brier'] = ovr_report['ave']['brier']
        metrics_dict['ave_mcc'] = ovr_report['ave']['mcc']
        metrics_dict['ave_auc'] = ovr_report['ave']['auc']
        metrics_dict['ave_ap'] = ovr_report['ave']['ap']
        metrics_dict['percent_error'] = percent_error
        metrics_dict['acc'] = acc

        harn.info(ub.color_text('ACC FOR {!r}: {!r}'.format(harn.current_tag, acc), 'yellow'))

        # Clear confusion vectors accumulator for the next epoch
        harn._accum_confusion_vectors = {
            'y_true': [],
            'y_pred': [],
            'probs': [],
        }
        return metrics_dict


def setup_harn(cmdline=True, **kw):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> kw = {'datasets': 'special:shapes256'}
        >>> cmdline = False
        >>> harn = setup_harn(cmdline, **kw)
        >>> harn.initialize()
    """
    import ndsampler
    config = ClfConfig(default=kw)
    config.load(cmdline=cmdline)
    print('config = {}'.format(ub.repr2(config.asdict())))

    nh.configure_hacks(config)
    coco_datasets = nh.api.Datasets.coerce(config)

    print('coco_datasets = {}'.format(ub.repr2(coco_datasets, nl=1)))
    for tag, dset in coco_datasets.items():
        dset._build_hashid(hash_pixels=False)

    workdir = ub.ensuredir(ub.expandpath(config['workdir']))
    samplers = {
        tag: ndsampler.CocoSampler(dset, workdir=workdir, backend=config['sampler_backend'])
        for tag, dset in coco_datasets.items()
    }

    for tag, sampler in ub.ProgIter(list(samplers.items()), desc='prepare frames'):
        sampler.frames.prepare(workers=config['workers'])

    torch_datasets = {
        'train': ClfDataset(
            samplers['train'],
            input_dims=config['input_dims'],
            augmenter=config['augmenter'],
        ),
        'vali': ClfDataset(
            samplers['vali'],
            input_dims=config['input_dims'],
            augmenter=False),
    }

    if config['normalize_inputs']:
        # Get stats on the dataset (todo: turn off augmentation for this)
        _dset = torch_datasets['train']
        stats_idxs = kwarray.shuffle(np.arange(len(_dset)), rng=0)[0:min(1000, len(_dset))]
        stats_subset = torch.utils.data.Subset(_dset, stats_idxs)

        cacher = ub.Cacher('dset_mean', cfgstr=_dset.input_id + 'v3')
        input_stats = cacher.tryload()

        channels = ChannelSpec.coerce(config['channels'])

        if input_stats is None:
            # Use parallel workers to load data faster
            from netharn.data.data_containers import container_collate
            from functools import partial
            collate_fn = partial(container_collate, num_devices=1)

            loader = torch.utils.data.DataLoader(
                stats_subset,
                collate_fn=collate_fn,
                num_workers=config['workers'],
                shuffle=True,
                batch_size=config['batch_size'])

            # Track moving average of each fused channel stream
            channel_stats = {key: nh.util.RunningStats()
                             for key in channels.keys()}
            assert len(channel_stats) == 1, (
                'only support one fused stream for now')
            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                for key, val in batch['inputs'].items():
                    try:
                        for part in val.numpy():
                            channel_stats[key].update(part)
                    except ValueError:  # final batch broadcast error
                        pass

            perchan_input_stats = {}
            for key, running in channel_stats.items():
                running = ub.peek(channel_stats.values())
                perchan_stats = running.simple(axis=(1, 2))
                perchan_input_stats[key] = {
                    'std': perchan_stats['mean'].round(3),
                    'mean': perchan_stats['std'].round(3),
                }

            input_stats = ub.peek(perchan_input_stats.values())
            cacher.save(input_stats)
    else:
        input_stats = {}

    torch_loaders = {
        tag: dset.make_loader(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'),
            balance=(config['balance'] if tag == 'train' else None),
            pin_memory=True)
        for tag, dset in torch_datasets.items()
    }

    initializer_ = None
    classes = torch_datasets['train'].classes

    modelkw = {
        'arch': config['arch'],
        'input_stats': input_stats,
        'classes': classes.__json__(),
        'channels': channels,
    }
    model = ClfModel(**modelkw)
    model._initkw = modelkw

    if initializer_ is None:
        initializer_ = nh.Initializer.coerce(config)

    hyper = nh.HyperParams(
        name=config['name'],

        workdir=config['workdir'],
        xpu=nh.XPU.coerce(config['xpu']),

        datasets=torch_datasets,
        loaders=torch_loaders,

        model=model,
        criterion=None,

        optimizer=nh.Optimizer.coerce(config),
        dynamics=nh.Dynamics.coerce(config),
        scheduler=nh.Scheduler.coerce(config),

        initializer=initializer_,

        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': config['patience'],
            'max_epoch': config['max_epoch'],
            'smoothing': 0.0,
        }),
        other={
            'name': config['name'],
            'batch_size': config['batch_size'],
            'balance': config['balance'],
        },
        extra={
            'argv': sys.argv,
            'config': ub.repr2(config.asdict()),
        }
    )
    harn = ClfHarn(hyper=hyper)
    harn.preferences.update({
        'num_keep': 3,
        'keep_freq': 10,
        'tensorboard_groups': ['loss'],
        'eager_dump_tensorboard': True,
    })
    harn.intervals.update({})
    harn.script_config = config
    return harn


def main():
    harn = setup_harn()
    harn.initialize()

    if ub.argflag('--lrtest'):
        # Undocumented hidden feature,
        # Perform an LR-test, then resetup the harness. Optionally draw the
        # results using matplotlib.
        from netharn.prefit.lr_tests import lr_range_test
        result = lr_range_test(
            harn, init_value=1e-4, final_value=0.5, beta=0.3,
            explode_factor=10, num_iters=200)
        if ub.argflag('--show'):
            import kwplot
            plt = kwplot.autoplt()
            result.draw()
            plt.show()
        # Recreate a new version of the harness with the recommended LR.
        config = harn.script_config.asdict()
        config['lr'] = (result.recommended_lr * 10)
        harn = setup_harn(**config)
        harn.initialize()
    # This starts the main loop which will run until the monitor's terminator
    # criterion is satisfied. If the initialize step loaded a checkpointed that
    # already met the termination criterion, then this will simply return.
    deploy_fpath = harn.run()

    # The returned deploy_fpath is the path to an exported netharn model.
    # This model is the on with the best weights according to the monitor.
    print('deploy_fpath = {!r}'.format(deploy_fpath))
    return harn


if __name__ == '__main__':
    """
    python -m netharn.examples.classification --datasets=shapes5000 --name=shapes_clf5000 --batch_size=32
    """
    main()
