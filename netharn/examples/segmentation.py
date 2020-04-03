# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import ubelt as ub
import torch  # NOQA
from torch.utils import data as torch_data  # NOQA
import numpy as np
import netharn as nh
import kwimage
import scriptconfig as scfg
from torch.nn import functional as F

import imgaug.augmenters as iaa
import imgaug


class SegmentationConfig(scfg.Config):
    """
    Default configuration for setting up a training session
    """
    default = {
        'nice': scfg.Value('untitled', help='A human readable tag that is "nice" for humans'),
        'workdir': scfg.Path('~/work/sseg', help='Dump all results in your workdir'),

        'workers': scfg.Value(0, help='number of parallel dataloading jobs'),
        'xpu': scfg.Value('argv', help='See netharn.XPU for details. can be cpu/gpu/cuda0/0,1,2,3)'),

        'augmenter': scfg.Value('simple', help='type of training dataset augmentation'),
        'class_weights': scfg.Value('log-median-idf', help='how to weight inbalanced classes'),
        # 'class_weights': scfg.Value(None, help='how to weight inbalanced classes'),

        'datasets': scfg.Value('special:shapes256', help='Either a special key or a coco file'),
        'train_dataset': scfg.Value(None),
        'vali_dataset': scfg.Value(None),
        'test_dataset': scfg.Value(None),

        'arch': scfg.Value('psp', help='Network architecture code'),
        'optim': scfg.Value('adam', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'backend': scfg.Value('npy', help='fast lookup backnd. may be npy or cog'),
        'input_dims': scfg.Value((224, 224), help='Window size to input to the network'),
        'input_overlap': scfg.Value(0.25, help='amount of overlap when creating a sliding window dataset'),
        'normalize_inputs': scfg.Value(True, help='if True, precompute training mean and std for data whitening'),

        'batch_size': scfg.Value(4, help='number of items per batch'),
        'bstep': scfg.Value(4, help='number of batches before a gradient descent step'),

        'max_epoch': scfg.Value(140, help='Maximum number of epochs'),
        'patience': scfg.Value(140, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-4, help='Base learning rate'),
        'decay':  scfg.Value(1e-5, help='Base weight decay'),

        'focus': scfg.Value(0.0, help='focus for focal loss'),

        'schedule': scfg.Value('step90', help=('Special coercable netharn code. Eg: onecycle50, step50, gamma')),

        'init': scfg.Value('kaiming_normal', help='How to initialized weights. (can be a path to a pretrained model)'),
        'pretrained': scfg.Path(help=('alternative way to specify a path to a pretrained model')),
    }

    def normalize(self):
        if self['pretrained'] in ['null', 'None']:
            self['pretrained'] = None

        if self['pretrained'] is not None:
            self['init'] = 'pretrained'


class SegmentationDataset(torch.utils.data.Dataset):
    """
    Efficient loader for training on a sementic segmentation dataset

    Example:
        >>> # DISABLE_DOCTEST
        >>> #input_dims = (224, 224)
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> import ndsampler
        >>> sampler = ndsampler.CocoSampler.demo('shapes')
        >>> input_dims = (512, 512)
        >>> self = dset = SegmentationDataset(sampler, input_dims)
        >>> output = self[10]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> cidxs = output['class_idxs']
        >>> colored_labels = self._colorized_labels(cidxs)
        >>> kwplot.figure(doclf=True)
        >>> kwplot.imshow(output['im'])
        >>> kwplot.imshow(colored_labels, alpha=.4)

    Example:
        >>> # xdoctest: +REQUIRES(--interact)
        >>> import xdev
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> indices = list(range(len(self)))
        >>> for index in xdev.InteractiveIter(indices):
        >>>     output = self[index]
        >>>     cidxs = output['class_idxs']
        >>>     colored_labels = self._colorized_labels(cidxs)
        >>>     kwplot.figure(doclf=True)
        >>>     kwplot.imshow(output['im'])
        >>>     kwplot.imshow(colored_labels, alpha=.4)
        >>>     xdev.InteractiveIter.draw()
    """
    def __init__(self, sampler, input_dims=(224, 224), input_overlap=0.5,
                 augmenter=False):
        self.input_dims = None
        self.input_id = None
        self.cid_to_cidx = None
        self.classes = None
        self.sampler = None
        self.subindex = None
        self.gid_to_slider = None
        self._gids = None
        self._sliders = None

        self.sampler = sampler

        self.input_id = self.sampler.dset.hashid

        self.cid_to_cidx = sampler.catgraph.id_to_idx
        self.classes = sampler.catgraph

        # Create a slider for every image
        self._build_sliders(input_dims=input_dims, input_overlap=input_overlap)
        self.augmenter = self._rectify_augmenter(augmenter)

    def _rectify_augmenter(self, augmenter):
        import netharn as nh
        if augmenter is True:
            augmenter = 'simple'

        if not augmenter:
            augmenter = None
        elif augmenter == 'simple':
            augmenter = iaa.Sequential([
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        elif augmenter == 'complex':
            augmenter = iaa.Sequential([
                iaa.Sometimes(0.2, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        else:
            raise KeyError('Unknown augmentation {!r}'.format(self.augment))
        return augmenter

    def _build_sliders(self, input_dims=(224, 224), input_overlap=0.5):
        """
        Use the ndsampler.Sampler and sliders to build a flat index that can
        reach every subregion of every image in the training set.
        """
        import netharn as nh
        gid_to_slider = {}
        for img in self.sampler.dset.imgs.values():
            full_dims = [img['height'], img['width']]
            slider = nh.util.SlidingWindow(full_dims, input_dims,
                                           overlap=input_overlap,
                                           allow_overshoot=True)
            gid_to_slider[img['id']] = slider

        self.input_dims = input_dims
        self.gid_to_slider = gid_to_slider
        self._gids = list(gid_to_slider.keys())
        self._sliders = list(gid_to_slider.values())
        self.subindex = nh.util.FlatIndexer.fromlist(self._sliders)
        return gid_to_slider

    def __len__(self):
        return len(self.subindex)

    def __getitem__(self, index):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> self = SegmentationDataset.demo(augment=True)
            >>> output = self[10]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> colored_labels = self._colorized_labels(output['class_idxs'])
            >>> kwplot.figure(doclf=True)
            >>> kwplot.imshow(output['im'])
            >>> kwplot.imshow(colored_labels, alpha=.4)
        """
        outer, inner = self.subindex.unravel(index)
        gid = self._gids[outer]
        slider = self._sliders[outer]
        slices = slider[inner]

        tr = {'gid': gid, 'slices': slices}
        sample = self.sampler.load_sample(tr, with_annots=['segmentation'])

        imdata = sample['im']
        heatmap = self._sample_to_sseg_heatmap(sample)

        heatmap = heatmap.numpy()
        heatmap.data['class_idx'] = heatmap.data['class_idx'].astype(np.int32)
        cidx_segmap = heatmap.data['class_idx']

        if self.augmenter:
            augdet = self.augmenter.to_deterministic()
            imdata = augdet.augment_image(imdata)
            if hasattr(imgaug, 'SegmentationMapsOnImage'):
                # Oh imgaug, stop breaking.
                cidx_segmap_oi = imgaug.SegmentationMapsOnImage(cidx_segmap, cidx_segmap.shape)
                cidx_segmap_oi = augdet.augment_segmentation_maps(cidx_segmap_oi)
                assert cidx_segmap_oi.arr.shape[2] == 1
                cidx_segmap = cidx_segmap_oi.arr[..., 0]
                cidx_segmap = np.ascontiguousarray(cidx_segmap)
            else:
                cidx_segmap_oi = imgaug.SegmentationMapOnImage(cidx_segmap, cidx_segmap.shape, nb_classes=len(self.classes))
                cidx_segmap_oi = augdet.augment_segmentation_maps([cidx_segmap_oi])[0]
                cidx_segmap = cidx_segmap_oi.arr.argmax(axis=2)

        im_chw = torch.FloatTensor(
            imdata.transpose(2, 0, 1).astype(np.float32) / 255.)

        cidxs = torch.LongTensor(cidx_segmap)
        weight = (1 - (cidxs == 0).float())

        output = {
            'im': im_chw,
            'class_idxs': cidxs,
            'weight': weight,
        }
        return output

    def _sample_to_sseg_heatmap(self, sample):
        imdata = sample['im']
        annots = sample['annots']
        aids = annots['aids']
        cids = annots['cids']
        boxes = annots['rel_boxes']

        # Clip boxes to the image boundary
        input_dims = imdata.shape[0:2]
        boxes = boxes.clip(0, 0, input_dims[1], input_dims[0])

        class_idxs = np.array([self.cid_to_cidx[cid] for cid in cids])
        segmentations = annots['rel_ssegs']

        raw_dets = kwimage.Detections(
            aids=aids,
            boxes=boxes,
            class_idxs=class_idxs,
            segmentations=segmentations,
            classes=self.classes,
            datakeys=['aids'],
        )

        keep = []
        for i, s in enumerate(raw_dets.data['segmentations']):
            # TODO: clip polygons
            m = s.to_mask(input_dims)
            if m.area > 0:
                keep.append(i)
        dets = raw_dets.take(keep)

        heatmap = dets.rasterize(
            bg_size=(1, 1), input_dims=input_dims,
            soften=0,
            exclude=['diameter', 'class_probs', 'offset']
        )

        try:
            # TODO: THIS MAY NOT BE THE CORRECT TRANSFORM
            input_shape = (1, 3,) + input_dims
            output_dims = getattr(self, '_output_dims', None)
            if output_dims is None:
                output_shape = self.raw_model.output_shape_for(input_shape)
                output_dims = self.output_dims = output_shape[2:]
            sf = np.array(output_dims) / np.array(input_dims)
            heatmap = heatmap.scale(sf, output_dims=output_dims, interpolation='nearest')
        except Exception:
            pass

        return heatmap

    def _colorized_labels(self, cidxs):
        self.cx_to_color = np.array([
            self.sampler.dset.name_to_cat[self.classes[cx]]['color']
            for cx in range(len(self.cid_to_cidx))
        ])
        colorized = self.cx_to_color[cidxs]
        return colorized

    @classmethod
    def demo(cls, **kwargs):
        # from grab_camvid import grab_coco_camvid
        # dset = grab_coco_camvid()
        import ndsampler
        sampler = ndsampler.CocoSampler.demo('shapes', workdir=None, backend='npy')
        self = cls(sampler, **kwargs)
        return self


class SegmentationHarn(nh.FitHarn):
    """
    Custom harness to address a basic semantic segmentation problem
    """

    def after_initialize(harn, **kw):
        harn.draw_timer = ub.Timer().tic()

        # hack:
        for k, v in harn.datasets.items():
            v.raw_model = harn.xpu.raw(harn.model)

    def prepare_batch(harn, raw_batch):
        """
        Move a batch onto the XPU
        """
        batch = harn.xpu.move(raw_batch)
        return batch

    def run_batch(harn, batch):
        """
        How to compute a forward pass through the network and compute loss

        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> kw = {'workers': 0, 'xpu': 'cpu', 'batch_size': 2}
            >>> harn = setup_harn(cmdline=False, **kw).initialize()
            >>> batch = harn._demo_batch(tag='train')
            >>> outputs, loss_parts = harn.run_batch(batch)
        """
        im = batch['im']
        class_idxs = batch['class_idxs']

        # You'll almost always have to do something custom to get the batch
        # into the network. Define that here.
        outputs = harn.model(im)
        if not isinstance(outputs, dict):
            outputs = {
                'class_energy': outputs,
            }
        class_energy = outputs['class_energy']

        class_probs = F.softmax(class_energy, dim=1)

        # Heirarchical softmax seems to have a memory leak
        # class_probs = harn.classes.hierarchical_softmax(class_energy, dim=1)

        outputs['class_probs'] = class_probs
        outputs['class_idxs'] = class_idxs

        # You'll almost always have to do something custom to compute loss.
        # Define that here.
        harn.criterion.reduction = 'none'
        pixel_loss = harn.criterion(class_energy, class_idxs)

        pixel_weight = batch['weight']
        clf_loss = (pixel_weight * pixel_loss).sum() / pixel_weight.sum()

        # Return info for netharn to track
        loss_parts = {}
        loss_parts['clf'] = clf_loss

        return outputs, loss_parts

    def on_batch(harn, batch, outputs, loss):
        batch_metrics = {}
        with torch.no_grad():
            # Track accuracy of each batch
            clf_true = outputs['class_idxs'].data.cpu().numpy().astype(np.uint8)
            clf_prob = outputs['class_probs'].data.cpu().numpy().astype(np.float32)
            clf_pred = clf_prob.argmax(axis=1)
            batch_metrics['clf_accuracy'] = (clf_pred == clf_true)[clf_true != 0].mean()

            # Draw first X batches or once every Y minutes
            do_draw = (harn.batch_index <= 8)
            do_draw |= (harn.draw_timer.toc() > 60 * 3)
            if do_draw:
                harn.draw_timer.tic()
                toshow = harn._draw_batch_preds(batch, outputs)
                dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag, 'batch'))
                fpath = join(dpath, 'batch{}_epoch{}.jpg'.format(harn.batch_index, harn.epoch))
                kwimage.imwrite(fpath, toshow)
        return batch_metrics

    def _draw_batch_preds(harn, batch, outputs, lim=16):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--slow)
            >>> kw = {'workers': 0, 'xpu': 'cpu', 'batch_size': 8}
            >>> harn = setup_harn(cmdline=False, **kw).initialize()
            >>> batch = harn._demo_batch(tag='train')
            >>> outputs, loss_parts = harn.run_batch(batch)
            >>> toshow = harn._draw_batch_preds(batch, outputs)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(toshow)
        """
        im = batch['im'].data.cpu().numpy()
        class_true = batch['class_idxs'].data.cpu().numpy()
        class_pred = outputs['class_probs'].data.cpu().numpy().argmax(axis=1)

        batch_imgs = []

        for bx in range(min(len(class_true), lim)):

            orig_img = im[bx].transpose(1, 2, 0)

            import cv2
            out_size = class_pred[bx].shape[::-1]

            orig_img = cv2.resize(orig_img, tuple(map(int, out_size)))
            orig_img = kwimage.ensure_alpha_channel(orig_img)

            pred_heatmap = kwimage.Heatmap(
                class_idx=class_pred[bx],
                classes=harn.classes
            )
            true_heatmap = kwimage.Heatmap(
                class_idx=class_true[bx],
                classes=harn.classes
            )

            # TODO: scale up to original image size

            pred_img = pred_heatmap.draw_on(orig_img, channel='idx', with_alpha=.5)
            true_img = true_heatmap.draw_on(orig_img, channel='idx', with_alpha=.5)

            true_img = kwimage.ensure_uint255(true_img)
            pred_img = kwimage.ensure_uint255(pred_img)

            true_img = kwimage.draw_text_on_image(
                true_img, 'true', org=(0, 0), valign='top', color='blue')

            pred_img = kwimage.draw_text_on_image(
                pred_img, 'pred', org=(0, 0), valign='top', color='blue')

            item_img = kwimage.stack_images([pred_img, true_img], axis=1)
            batch_imgs.append(item_img)

        toshow = kwimage.stack_images_grid(batch_imgs, chunksize=2, overlap=-32)
        return toshow

    def on_complete(harn):
        """
        Ignore:
            >>> kw = {'workers': 0, 'xpu': 'auto', 'batch_size': 8}
            >>> harn = setup_harn(cmdline=False, **kw).initialize()
            >>> harn.datasets['test']
        """
        deployed = harn.raw_model
        out_dpath = ub.ensuredir((harn.train_dpath, 'monitor/test/'))
        # test_dset = harn.datasets['test']
        # sampler = test_dset.sampler

        eval_config = {
            'deployed': deployed,
            'xpu': harn.xpu,
            'out_dpath': out_dpath,
            'do_draw': True,
        }
        print('todo: evalute eval_config = {!r}'.format(eval_config))


class SegmentationModel(nh.layers.Sequential):
    """
    Dummy wrapper around the real model but with input norm
    """
    def __init__(self, arch, classes, in_channels=3, input_stats=None):
        super(SegmentationModel, self).__init__()
        import ndsampler
        if input_stats is None:
            input_stats = {}
        input_norm = nh.layers.InputNorm(**input_stats)

        classes = ndsampler.CategoryTree.coerce(classes)
        self.classes = classes
        if arch == 'unet':
            # Note: UNet can get through 256x256 images at a rate of ~17Hz with
            # batch_size=8. This is pretty slow and can likely be improved by fixing
            # some of the weird padding / mirror stuff I have to do in unet to get
            # output_dims = input_dims.
            from netharn.models.unet import UNet
            model_ = (UNet, {
                'classes': classes,
                'in_channels': in_channels,
            })
        elif arch == 'segnet':
            from netharn.models.segnet import Segnet
            model_ = (Segnet, {
                'in_channels': in_channels,
            })
        elif arch == 'psp':
            from netharn.models.psp import PSPNet_Resnet50_8s
            model_ = (PSPNet_Resnet50_8s, {
                'classes': classes,
                'in_channels': in_channels,
            })
        elif arch == 'deeplab_v3':
            from netharn.models.deeplab_v3 import DeepLabV3
            model_ = (DeepLabV3, {
                'classes': classes,
                'in_channels': in_channels,
            })
        else:
            raise KeyError(arch)

        self.add_module('input_norm', input_norm)
        self.add_module('model', model_[0](**model_[1]))


def _cached_class_frequency(dset, workers=0):
    import ubelt as ub
    import copy
    # Copy the dataset so we can muck with it
    dset_copy = copy.copy(dset)

    dset_copy._build_sliders(input_overlap=0)
    dset_copy.augmenter = None

    cfgstr = '_'.join([dset_copy.sampler.dset.hashid, 'v1'])
    cacher = ub.Cacher('class_freq', cfgstr=cfgstr)
    total_freq = cacher.tryload()
    if total_freq is None:

        total_freq = np.zeros(len(dset_copy.classes), dtype=np.int64)
        if True:
            loader = torch_data.DataLoader(dset_copy, batch_size=16,
                                           num_workers=workers, shuffle=False,
                                           pin_memory=True)

            prog = ub.ProgIter(loader, desc='computing (par) class freq')
            for batch in prog:
                class_idxs = batch['class_idxs'].data.numpy()
                item_freq = np.histogram(class_idxs, bins=len(dset_copy.classes))[0]
                total_freq += item_freq
        else:
            prog = ub.ProgIter(range(len(dset_copy)), desc='computing (ser) class freq')
            for index in prog:
                item = dset_copy[index]
                class_idxs = item['class_idxs'].data.numpy()
                item_freq = np.histogram(class_idxs, bins=len(dset_copy.classes))[0]
                total_freq += item_freq
        cacher.save(total_freq)
    return total_freq


def _precompute_class_weights(dset, workers=0, mode='median-idf'):
    """
    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> harn = setup_harn(0, workers=0, xpu='cpu').initialize()
        >>> dset = harn.datasets['train']
    """

    assert mode in ['median-idf', 'log-median-idf']

    total_freq = _cached_class_frequency(dset, workers=workers)

    def logb(arr, base):
        if base == 'e':
            return np.log(arr)
        elif base == 2:
            return np.log2(arr)
        elif base == 10:
            return np.log10(arr)
        else:
            out = np.log(arr)
            out /= np.log(base)
            return out

    _min, _max = np.percentile(total_freq, [5, 95])
    is_valid = (_min <= total_freq) & (total_freq <= _max)
    if np.any(is_valid):
        middle_value = np.median(total_freq[is_valid])
    else:
        middle_value = np.median(total_freq)

    # variant of median-inverse-frequency
    nonzero_freq = total_freq[total_freq != 0]
    if len(nonzero_freq):
        total_freq[total_freq == 0] = nonzero_freq.min() / 2

    if mode == 'median-idf':
        weights = (middle_value / total_freq)
        weights[~np.isfinite(weights)] = 1.0
    elif mode == 'log-median-idf':
        weights = (middle_value / total_freq)
        weights[~np.isfinite(weights)] = 1.0
        base = 2
        base = np.exp(1)
        weights = logb(weights + (base - 1), base)
        weights = np.maximum(weights, .1)
        weights = np.minimum(weights, 10)
    else:
        raise KeyError('mode = {!r}'.format(mode))

    weights = np.round(weights, 2)
    cname_to_weight = ub.dzip(dset.classes, weights)
    print('weights: ' + ub.repr2(cname_to_weight))
    return weights


def setup_harn(cmdline=True, **kw):
    """
    CommandLine:
        xdoctest -m netharn.examples.segmentation setup_harn

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> kw = {'workers': 0, 'xpu': 'cpu', 'batch_size': 2}
        >>> cmdline = False
        >>> # Just sets up the harness, does not do any heavy lifting
        >>> harn = setup_harn(cmdline=cmdline, **kw)
        >>> #
        >>> harn.initialize()
        >>> #
        >>> batch = harn._demo_batch(tag='train')
        >>> epoch_metrics = harn._demo_epoch(tag='vali', max_iter=2)
    """
    import sys
    import ndsampler
    import kwarray
    # kwarray.seed_global(2108744082)

    config = SegmentationConfig(default=kw)
    config.load(cmdline=cmdline)
    nh.configure_hacks(config)  # fix opencv bugs

    coco_datasets = nh.api.Datasets.coerce(config)
    print('coco_datasets = {}'.format(ub.repr2(coco_datasets)))
    for tag, dset in coco_datasets.items():
        dset._build_hashid(hash_pixels=False)

    workdir = ub.ensuredir(ub.expandpath(config['workdir']))
    samplers = {
        tag: ndsampler.CocoSampler(dset, workdir=workdir, backend=config['backend'])
        for tag, dset in coco_datasets.items()
    }

    for tag, sampler in ub.ProgIter(list(samplers.items()), desc='prepare frames'):
        try:
            sampler.frames.prepare(workers=config['workers'])
        except AttributeError:
            pass

    torch_datasets = {
        tag: SegmentationDataset(
            sampler,
            config['input_dims'],
            input_overlap=((tag == 'train') and config['input_overlap']),
            augmenter=((tag == 'train') and config['augmenter']),
        )
        for tag, sampler in samplers.items()
    }
    torch_loaders = {
        tag: torch_data.DataLoader(dset,
                                   batch_size=config['batch_size'],
                                   num_workers=config['workers'],
                                   shuffle=(tag == 'train'),
                                   drop_last=True, pin_memory=True)
        for tag, dset in torch_datasets.items()
    }

    if config['class_weights']:
        mode = config['class_weights']
        dset = torch_datasets['train']
        class_weights = _precompute_class_weights(dset, mode=mode,
                                                  workers=config['workers'])
        class_weights = torch.FloatTensor(class_weights)
        class_weights[dset.classes.index('background')] = 0
    else:
        class_weights = None

    if config['normalize_inputs']:
        stats_dset = torch_datasets['train']
        stats_idxs = kwarray.shuffle(np.arange(len(stats_dset)), rng=0)[0:min(1000, len(stats_dset))]
        stats_subset = torch.utils.data.Subset(stats_dset, stats_idxs)
        cacher = ub.Cacher('dset_mean', cfgstr=stats_dset.input_id + 'v3')
        input_stats = cacher.tryload()
        if input_stats is None:
            loader = torch.utils.data.DataLoader(
                stats_subset, num_workers=config['workers'],
                shuffle=True, batch_size=config['batch_size'])
            running = nh.util.RunningStats()
            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                try:
                    running.update(batch['im'].numpy())
                except ValueError:  # final batch broadcast error
                    pass
            input_stats = {
                'std': running.simple(axis=None)['mean'].round(3),
                'mean': running.simple(axis=None)['std'].round(3),
            }
            cacher.save(input_stats)
    else:
        input_stats = {}

    print('input_stats = {!r}'.format(input_stats))

    # TODO: infer numbr of channels
    model_ = (SegmentationModel, {
        'arch': config['arch'],
        'input_stats': input_stats,
        'classes': torch_datasets['train'].classes.__json__(),
        'in_channels': 3,
    })

    initializer_ = nh.Initializer.coerce(config)
    # if config['init'] == 'cls':
    #     initializer_ = model_[0]._initializer_cls()

    # Create hyperparameters
    hyper = nh.HyperParams(
        nice=config['nice'],
        workdir=config['workdir'],
        xpu=nh.XPU.coerce(config['xpu']),

        datasets=torch_datasets,
        loaders=torch_loaders,

        model=model_,
        initializer=initializer_,

        scheduler=nh.Scheduler.coerce(config),
        optimizer=nh.Optimizer.coerce(config),
        dynamics=nh.Dynamics.coerce(config),

        criterion=(nh.criterions.FocalLoss, {
            'focus': config['focus'],
            'weight': class_weights,
            # 'reduction': 'none',
        }),

        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': config['patience'],
            'max_epoch': config['max_epoch'],
            'smoothing': .6,
        }),

        other={
            'batch_size': config['batch_size'],
        },
        extra={
            'argv': sys.argv,
            'config': ub.repr2(config.asdict()),
        }
    )

    # Create harness
    harn = SegmentationHarn(hyper=hyper)
    harn.classes = torch_datasets['train'].classes
    harn.preferences.update({
        'num_keep': 2,
        'keyboard_debug': True,
        # 'export_modules': ['netharn'],
    })
    harn.intervals.update({
        'vali': 1,
        'test': 10,
    })
    harn.script_config = config
    return harn


def main():
    harn = setup_harn()
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:

        conda install gdal

        python -m netharn.examples.segmentation \
                --nice=shapes_demo --datasets=shapes32 \
                --workers=0 --xpu=cpu

        # You can use MS-COCO files to learn to segment your own data To
        # demonstrate grab the CamVid dataset (the following script also
        # transforms camvid into the MS-COCO format)

        python -m netharn.data.grab_camvid  # Download MS-COCO files

        python -m netharn.examples.segmentation --workers=4 --xpu=0 --nice=camvid_deeplab \
            --train_dataset=$HOME/.cache/netharn/camvid/camvid-master/camvid-train.mscoco.json \
            --vali_dataset=$HOME/.cache/netharn/camvid/camvid-master/camvid-train.mscoco.json \
            --schedule=step-90-120 --arch=deeplab --batch_size=8 --lr=1e-5 --input_dims=224,224 --optim=sgd --bstep=8

        python -m netharn.examples.segmentation --workers=4 --xpu=auto --nice=camvid_psp_wip \
            --train_dataset=$HOME/.cache/netharn/camvid/camvid-master/camvid-train.mscoco.json \
            --vali_dataset=$HOME/.cache/netharn/camvid/camvid-master/camvid-train.mscoco.json \
            --schedule=step-90-120 --arch=psp --batch_size=6 --lr=1e-3 --input_dims=512,512 --optim=sgd --bstep=1

        # Note you would need to change the path to a pretrained network
        python -m netharn.examples.segmentation --workers=4 --xpu=auto --nice=camvid_psp_wip_fine \
            --train_dataset=$HOME/.cache/netharn/camvid/camvid-master/camvid-train.mscoco.json \
            --vali_dataset=$HOME/.cache/netharn/camvid/camvid-master/camvid-train.mscoco.json \
            --pretrained=$HOME/work/sseg/fit/runs/camvid_psp_wip/fowjplca/deploy_SegmentationModel_fowjplca_134_CZARGB.zip \
            --schedule=step-90-120 --arch=psp --batch_size=6 --lr=1e-2 --input_dims=512,512 --optim=sgd --bstep=8
    """
    main()
