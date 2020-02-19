# -*- coding: utf-8 -*-
"""
An train an example semenatic segmenation model on the CamVid dataset.
For a more general segmentation example that works with any (ndsampler-style)
MS-COCO dataset see segmentation.py.

CommandLine:
    python ~/code/netharn/examples/sseg_camvid.py --workers=4 --xpu=0 --batch_size=2 --nice=expt1
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
import ubelt as ub
import torch  # NOQA
from torch.utils import data as torch_data  # NOQA
import numpy as np
import netharn as nh
import kwimage
import six
import scriptconfig as scfg
from torch.nn import functional as F

import imgaug.augmenters as iaa
import imgaug


class SegmentationConfig(scfg.Config):
    """
    Default configuration for setting up a training session
    """
    default = {
        'nice': scfg.Path('untitled', help='A human readable tag that is "nice" for humans'),
        'workdir': scfg.Path('~/work/camvid', help='Dump all results in your workdir'),

        'workers': scfg.Value(0, help='number of parallel dataloading jobs'),
        'xpu': scfg.Value('argv', help='See netharn.XPU for details. can be cpu/gpu/cuda0/0,1,2,3)'),

        'augment': scfg.Value('simple', help='type of training dataset augmentation'),
        'class_weights': scfg.Value('log-median-idf', help='how to weight inbalanced classes'),
        # 'class_weights': scfg.Value(None, help='how to weight inbalanced classes'),

        'datasets': scfg.Value('special:camvid', help='Eventually you may be able to sepcify a coco file'),
        'train_dataset': scfg.Value(None),
        'vali_dataset': scfg.Value(None),

        'arch': scfg.Value('psp', help='Network architecture code'),
        'optim': scfg.Value('adamw', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'input_dims': scfg.Value((128, 128), help='Window size to input to the network'),
        'input_overlap': scfg.Value(0.25, help='amount of overlap when creating a sliding window dataset'),

        'batch_size': scfg.Value(4, help='number of items per batch'),
        'bstep': scfg.Value(1, help='number of batches before a gradient descent step'),

        'max_epoch': scfg.Value(140, help='Maximum number of epochs'),
        'patience': scfg.Value(140, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-3, help='Base learning rate'),
        'decay':  scfg.Value(1e-5, help='Base weight decay'),

        'focus': scfg.Value(2.0, help='focus for focal loss'),

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
        >>> # xdoctest: +REQUIRES(--download)
        >>> sampler = grab_camvid_sampler()
        >>> #input_dims = (224, 224)
        >>> input_dims = (512, 512)
        >>> self = dset = SegmentationDataset(sampler, input_dims)
        >>> item = self[10]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> cidxs = item['class_idxs']
        >>> colored_labels = self._colorized_labels(cidxs)
        >>> kwplot.figure(doclf=True)
        >>> kwplot.imshow(item['im'])
        >>> kwplot.imshow(colored_labels, alpha=.4)

    Example:
        >>> # xdoctest: +REQUIRES(--interact)
        >>> import xdev
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> indices = list(range(len(self)))
        >>> for index in xdev.InteractiveIter(indices):
        >>>     item = self[index]
        >>>     cidxs = item['class_idxs']
        >>>     colored_labels = self._colorized_labels(cidxs)
        >>>     kwplot.figure(doclf=True)
        >>>     kwplot.imshow(item['im'])
        >>>     kwplot.imshow(colored_labels, alpha=.4)
        >>>     xdev.InteractiveIter.draw()
    """
    def __init__(self, sampler, input_dims=(224, 224), input_overlap=0.5,
                 augment=False):
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

        self.augmenter = self._rectify_augmenter(augment)

        # Create a slider for every image
        self._build_sliders(input_dims=input_dims, input_overlap=input_overlap)

    def _rectify_augmenter(self, augment):
        import netharn as nh
        if augment is True:
            augment = 'simple'

        if not augment:
            augmenter = None
        elif augment == 'simple':
            augmenter = iaa.Sequential([
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        elif augment == 'complex':
            augmenter = iaa.Sequential([
                iaa.Sometimes(0.2, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5)
            ])
        else:
            raise KeyError('Unknown augmentation {!r}'.format(augment))
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
            >>> # xdoctest: +REQUIRES(--download)
            >>> self = SegmentationDataset.demo(augment='complex')
            >>> item = self[10]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()
            >>> colored_labels = self._colorized_labels(item['class_idxs'])
            >>> kwplot.figure(doclf=True)
            >>> kwplot.imshow(item['im'])
            >>> kwplot.imshow(colored_labels, alpha=.4)
        """
        outer, inner = self.subindex.unravel(index)
        gid = self._gids[outer]
        slider = self._sliders[outer]
        slices = slider[inner]

        tr = {'gid': gid, 'slices': slices}
        sample = self.sampler.load_sample(tr)

        imdata = sample['im']
        heatmap = self._sample_to_sseg_heatmap(imdata, sample)

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

        item = {
            'im': im_chw,
            'class_idxs': cidxs,
            'weight': weight,
        }
        return item

    def _sample_to_sseg_heatmap(self, imdata, sample):
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

        heatmap = dets.rasterize(bg_size=(1, 1), input_dims=input_dims,
                                 soften=0, exclude=['diameter', 'class_probs', 'offset'])
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
        from netharn.data.grab_camvid import grab_coco_camvid
        import ndsampler
        dset = grab_coco_camvid()
        sampler = ndsampler.CocoSampler(dset, workdir=None, backend='npy')
        self = cls(sampler, **kwargs)
        return self


class SegmentationHarn(nh.FitHarn):
    """
    Custom harness to address a basic semantic segmentation problem
    """

    def after_initialize(harn, **kw):
        harn.draw_timer = ub.Timer().tic()

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
            >>> # xdoctest: +REQUIRES(--download)
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

        REDUCE_LOSS = False
        if REDUCE_LOSS:
            # Use only a few of the hardest pixels to compute loss
            # (not sure where I saw this trick, or if it works)
            weighted_loss = pixel_weight * pixel_loss

            parts = []
            for item_loss in weighted_loss:
                item_loss = item_loss.view(-1)
                idxs = item_loss.argsort()
                part = item_loss[idxs[-1024:]].mean()
                parts.append(part)
            clf_loss = sum(parts) / len(parts)
        else:
            # Do the normal thin
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

            # Draw first 8 batches or once every 5 minutes
            do_draw = (harn.batch_index <= 8)
            do_draw |= (harn.draw_timer.toc() > 60 * 5)
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
            >>> # xdoctest: +REQUIRES(--download)
            >>> kw = {'workers': 0, 'xpu': 'cpu', 'batch_size': 8}
            >>> harn = setup_harn(cmdline=False, **kw).initialize()
            >>> batch = harn._demo_batch(tag='train')
            >>> outputs, loss_parts = harn.run_batch(batch)
            >>> toshow = harn._draw_batch_preds(batch, outputs)
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
            orig_img = kwimage.ensure_alpha_channel(orig_img)

            pred_heatmap = kwimage.Heatmap(
                class_idx=class_pred[bx],
                classes=harn.classes
            )
            true_heatmap = kwimage.Heatmap(
                class_idx=class_true[bx],
                classes=harn.classes
            )

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
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.expandpath('~/code/netharn/examples'))
            >>> from sseg_camvid import *  # NOQA
            >>> kw = {'workers': 0, 'xpu': 'auto', 'batch_size': 8}
            >>> harn = setup_harn(cmdline=False, **kw).initialize()
            >>> harn.datasets['test']
        """
        deployed = harn.raw_model
        out_dpath = ub.ensuredir((harn.train_dpath, 'monitor/test/'))
        test_dset = harn.datasets['test']
        sampler = test_dset.sampler

        eval_config = {
            'deployed': deployed,
            'xpu': harn.xpu,
            'out_dpath': out_dpath,
            'do_draw': True,
        }
        evaluate_network(sampler, eval_config)


def evaluate_network(sampler, eval_config):
    """
    TODO:
        - [ ] set this up as its own script
        - [ ] find a way to generalize the Evaluator concept using the Predictor.

    Notes:
        scores to beat: http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html
        basic pixel_acc=82.8% class_acc=62.3% mean_iou=46.3%
        best pixel_acc=88.6% class_acc=81.3% mean_iou=69.1%
        Segnet (3.5K dataset) 86.8%, 81.3%, 69.1%,

        ours pixel_acc=86.2%, class_acc=45.4% mean_iou=31.2%

    Ignore:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/netharn/examples'))
        >>> from sseg_camvid import *  # NOQA
        >>> kw = {'workers': 0, 'xpu': 'auto', 'batch_size': 8}
        >>> harn = setup_harn(cmdline=False, **kw).initialize()
        >>> harn.datasets['test']
        >>> out_dpath = ub.ensuredir((harn.train_dpath, 'monitor/test/'))
        >>> test_dset = harn.datasets['test']
        >>> sampler = test_dset.sampler
        >>> deployed = ub.expandpath('/home/joncrall/work/camvid/fit/nice/camvid_augment_weight_v2/deploy_UNet_otavqgrp_089_FOAUOG.zip')
        >>> out_dpath = ub.expandpath('/home/joncrall/work/camvid/fit/nice/monitor/test')
        >>> do_draw = True
        >>> evaluate_network(sampler, deployed, out_dpath, do_draw)
    """
    from netharn.data.grab_camvid import rgb_to_cid
    coco_dset = sampler.dset
    classes = sampler.classes

    # TODO: find a way to generalize the Evaluator concept using the
    # Predictor.
    pred_cfgs = {
        'workers': 0,
        'deployed': eval_config['deployed'],
        'xpu': eval_config['xpu'],
        # 'window_dims': (720, 960),
        'window_dims': (512, 512),
        'batch_size': 2,
    }
    segmenter = SegmentationPredictor(**pred_cfgs)
    segmenter._preload()

    evaluator = SegmentationEvaluator(classes)

    cid_to_cx = classes.id_to_idx
    cx_to_cid = np.zeros(len(classes), dtype=np.int32)
    for cid, cx in cid_to_cx.items():
        cx_to_cid[cx] = cid

    def camvid_load_truth(img):
        # TODO: better way to load per-pixel truth with sampler
        mask_fpath = join(coco_dset.img_root, img['segmentation'])
        rgb_mask = kwimage.imread(mask_fpath, space='rgb')
        r, g, b  = rgb_mask.T.astype(np.int64)
        cid_mask = np.ascontiguousarray(rgb_to_cid(r, g, b).T)

        ignore_idx = 0
        cidx_mask = np.full_like(cid_mask, fill_value=ignore_idx)
        for cx, cid in enumerate(cx_to_cid):
            locs = (cid_mask == cid)
            cidx_mask[locs] = cx
        return cidx_mask

    prog = ub.ProgIter(sampler.image_ids, 'evaluating', clearline=False)
    for gid in prog:
        img, annots = sampler.load_image_with_annots(gid)
        prog.ensure_newline()
        print('Estimate: ' + ub.repr2(evaluator.estimate, nl=0, precision=3))

        full_rgb = img['imdata']
        pred_heatmap = segmenter.predict(full_rgb)

        # Prepare truth and predictions
        true_cidx = camvid_load_truth(img)
        true_heatmap = kwimage.Heatmap(class_idx=true_cidx, classes=classes)

        # Ensure predictions are comparable to the truth
        pred_cid = pred_heatmap.data['class_idx']
        pred_heatmap.data['class_cid'] = cx_to_cid[pred_cid]

        # Add truth and predictions to the evaluator
        img_results = evaluator.add(gid, true_heatmap, pred_heatmap)

        prog.set_extra('mean_iou_g={:.2f}% mean_iou_t={:.2f}%'.format(
            img_results['mean_iou'], evaluator.estimate['mean_iou'])
        )

        if eval_config['do_draw']:
            out_dpath = eval_config['out_dpath']
            canvas = pred_heatmap.draw_on(full_rgb, channel='idx', with_alpha=0.5)
            gpath = join(out_dpath, 'gid_{:04d}.jpg'.format(gid))
            kwimage.imwrite(gpath, canvas)

    print('Final: ' + ub.repr2(evaluator.estimate, nl=0, precision=3))


class SegmentationEvaluator(object):
    def __init__(evaluator, classes):
        evaluator.classes = classes

        # Remember metrics for each image individually
        evaluator.gid_to_metrics = {}

        # accum is a dictionary that will hold different metrics we accumulate
        evaluator.accum = ub.ddict(lambda : 0)

        # Estimate contains our current averaged metrics
        evaluator.estimate = {}

        # We don't care how we predict for the void class
        evaluator.void_idx = classes.index('background')

    def add(evaluator, gid, true_heatmap, pred_heatmap):
        true_idx = true_heatmap.data['class_idx']
        pred_idx = pred_heatmap.data['class_idx']

        is_valid = (true_idx != evaluator.void_idx)
        valid_true_idx = true_idx[is_valid]
        valid_pred_idx = pred_idx[is_valid]

        cfsn = nh.metrics.confusion_matrix(valid_true_idx, valid_pred_idx,
                                           labels=evaluator.classes)

        is_correct = (valid_pred_idx == valid_true_idx)

        metrics = {
            'cfsn': cfsn,
            'num_tp': is_correct.sum(),
            'n_total': is_correct.size,
        }

        evaluator.accum['cfsn'] = metrics['cfsn'] + evaluator.accum['cfsn']
        evaluator.accum['n_total'] += metrics['n_total']
        evaluator.accum['num_tp'] += metrics['num_tp']

        evaluator.estimate = evaluator._crunch(evaluator.accum)
        evaluator.gid_to_metrics[gid] = metrics

        img_results = evaluator._crunch(metrics)
        return img_results

    def _crunch(evaluator, metrics):
        def mean_iou_from_confusions(cfsn):
            """ Calculate IoU for each class (jaccard score) """
            tp = np.diag(cfsn)
            # real is rows, pred is columns
            fp = cfsn.sum(axis=0) - tp
            fn = cfsn.sum(axis=1) - tp
            denom = (tp + fp + fn)
            # Only look at classes with some truth or prediction support
            subidxs = np.where(denom > 0)[0]
            ious = tp[subidxs] / denom[subidxs]
            mean_iou = ious.mean()
            return mean_iou

        def pixel_accuracy_from_confusion(cfsn):
            # real is rows, pred is columns
            n_ii = np.diag(cfsn)
            # sum over pred = columns = axis1
            t_i = cfsn.sum(axis=1)
            global_acc = n_ii.sum() / t_i.sum()
            return global_acc

        def class_accuracy_from_confusion(cfsn):
            # real is rows, pred is columns
            n_ii = np.diag(cfsn)
            # sum over pred = columns = axis1
            t_i = cfsn.sum(axis=1)

            subidxs = np.where(t_i > 0)[0]
            class_acc = (n_ii[subidxs] / t_i[subidxs]).mean()
            return class_acc

        result = {
            'acc': metrics['num_tp'] / metrics['n_total'],
            'pixel_acc': pixel_accuracy_from_confusion(metrics['cfsn']),
            'class_acc': class_accuracy_from_confusion(metrics['cfsn']),
            'mean_iou': mean_iou_from_confusions(metrics['cfsn']),
        }
        return result


class SegmentationPredictor(object):
    """
    Wraps a pretrained model and allows prediction on full images

    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> deployed = ub.expandpath('/home/joncrall/work/camvid/fit/runs/camvid_augment_weight_v2/otavqgrp/deploy_UNet_otavqgrp_089_FOAUOG.zip')
        >>> full_rgb = kwimage.grab_test_image('astro')
        >>> self = SegmentationPredictor(deployed)
        >>> self._preload()
        >>> heatmap = self.predict(full_rgb)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> canvas = heatmap.draw_on(full_rgb, channel='idx', with_alpha=0.5)
        >>> kwplot.imshow(canvas)
    """

    def __init__(self, deployed, xpu='argv', **kw):
        self.config = {
            'window_dims': (256, 256),
            'batch_size': 1,
            'workers': 0,
            'xpu': xpu,
            'deployed': deployed,
        }
        unknown = set(kw) - set(self.config)
        assert not unknown, 'unknown={}'.format(unknown)

        self.config.update(kw)
        self.xpu = None
        self.model = None

    def _preload(self):
        if self.model is None:
            self.model, self.xpu = nh.export.DeployedModel.ensure_mounted_model(
                self.config['deployed'], self.config['xpu'])
        self.model.train(False)

    def predict(self, path_or_image):
        self._preload()
        if isinstance(path_or_image, six.string_types):
            print('Reading {!r}'.format(path_or_image))
            full_rgb = kwimage.imread(path_or_image, space='rgb')
        else:
            full_rgb = path_or_image

        # Note, this doesn't handle the case where the image is smaller than
        # the window
        input_dims = full_rgb.shape[0:2]

        try:
            classes = self.model.module.classes
        except AttributeError:
            classes = self.model.classes

        stitchers = {
            'class_energy': nh.util.Stitcher((len(classes),) + tuple(input_dims))
        }
        slider = nh.util.SlidingWindow(input_dims, self.config['window_dims'],
                                       overlap=0, keepbound=True,
                                       allow_overshoot=True)

        slider_dset = PredSlidingWindowDataset(slider, full_rgb)
        slider_loader = torch.utils.data.DataLoader(slider_dset,
                                                    batch_size=self.config['batch_size'],
                                                    num_workers=self.config['workers'],
                                                    shuffle=False,
                                                    pin_memory=True)

        prog = ub.ProgIter(slider_loader, desc='sliding window')

        with torch.no_grad():
            for raw_batch in prog:
                im = self.xpu.move(raw_batch['im'])
                outputs = self.model(im)

                class_energy = outputs['class_energy'].data.cpu().numpy()

                batch_sl_st_dims = raw_batch['sl_st_dims'].data.cpu().long().numpy().tolist()
                batch_sl_dims = [tuple(slice(s, t) for s, t in item)
                                 for item in batch_sl_st_dims]

                for sl_dims, energy in zip(batch_sl_dims, class_energy):
                    slices = (slice(None),) + sl_dims
                    stitchers['class_energy'].add(slices, energy)

            full_class_energy = stitchers['class_energy'].finalize()
            full_class_probs = torch.FloatTensor(full_class_energy).softmax(dim=0)
            full_class_probs = full_class_probs.numpy()
            full_class_idx = full_class_probs.argmax(axis=0)

        pred_heatmap = kwimage.Heatmap(
            class_probs=full_class_probs,
            class_idx=full_class_idx,
            classes=classes,
            datakeys=['class_idx'],
        )

        return pred_heatmap


class PredSlidingWindowDataset(torch_data.Dataset, ub.NiceRepr):
    """
    A basic torch dataset that accesses a larger image via sliding windows

    This is used for prediction and evaluation

    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> import kwimage
        >>> image = kwimage.grab_test_image('astro')
        >>> slider = nh.util.SlidingWindow(image.shape[0:2], (64, 64))
        >>> self = PredSlidingWindowDataset(slider, image)
        >>> self[5]
    """

    def __init__(self, slider, image):
        self.slider = slider
        self.image = image

    def __nice__(self):
        return self.slider.__nice__()

    def __len__(self):
        return len(self.slider)

    def __getitem__(self, index):
        slider = self.slider
        slices = slider[index]
        # Use floats in case we want to support subpixel accuracy later on
        # slice (sl) start-stop (st) dimensions (dims)
        sl_st_dims = torch.FloatTensor([(sl.start, sl.stop) for sl in slices])
        im_hwc = self.image[slices]
        # Transform chips into network-input space
        im_chw = torch.FloatTensor(im_hwc.transpose(2, 0, 1))
        im_chw /= 255.0
        # Packing return values as dictionaries is a good idea
        batch_item = {
            'im': im_chw,
            'sl_st_dims': sl_st_dims,
        }
        return batch_item


def setup_coco_datasets():
    """
    TODO:
        - [ ] Read arbitrary coco datasets here
        - [ ] Do proper train / validation split
        - [ ] Allow custom train / validation split
    """
    from netharn.data.grab_camvid import grab_coco_camvid, grab_camvid_train_test_val_splits
    coco_dset = grab_coco_camvid()

    # Use the same train/test/vali splits used in segnet
    gid_subsets = grab_camvid_train_test_val_splits(coco_dset, mode='segnet')
    print(ub.map_vals(len, gid_subsets))
    # gid_subsets.pop('test')

    # all_gids = list(coco_dset.imgs.keys())
    # gid_subsets = {
    #     'train': all_gids[0:-100],
    #     'vali': all_gids[-100:],
    # }
    coco_datasets = {
        tag: coco_dset.subset(gids)
        for tag, gids in gid_subsets.items()
    }
    print('coco_datasets = {}'.format(ub.repr2(coco_datasets)))
    for tag, dset in coco_datasets.items():
        dset._build_hashid(hash_pixels=False)
    return coco_datasets


def _cached_class_frequency(dset):
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
                                           num_workers=7, shuffle=False,
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


def _precompute_class_weights(dset, mode='median-idf'):
    """
    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/netharn/examples'))
        >>> from sseg_camvid import *  # NOQA
        >>> harn = setup_harn(0, workers=0, xpu='cpu').initialize()
        >>> dset = harn.datasets['train']
    """

    assert mode in ['median-idf', 'log-median-idf']

    total_freq = _cached_class_frequency(dset)

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

    if False:
        # Inspect the weights
        import kwplot
        kwplot.autoplt()

        cname_to_weight = ub.dzip(dset.classes, weights)
        cname_to_weight = ub.dict_subset(cname_to_weight, ub.argsort(cname_to_weight))
        kwplot.multi_plot(
            ydata=list(cname_to_weight.values()),
            kind='bar',
            xticklabels=list(cname_to_weight.keys()),
            xtick_rotation=90,
            fnum=2, doclf=True)

    return weights


def setup_harn(cmdline=True, **kw):
    """
    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/netharn/examples'))
        >>> from sseg_camvid import *  # NOQA
        >>> kw = {'workers': 0, 'xpu': 'cpu', 'batch_size': 2}
        >>> cmdline = False
        >>> # Just sets up the harness, does not do any heavy lifting
        >>> harn = setup_harn(cmdline=cmdline, **kw)
        >>> #
        >>> harn.initialize()
        >>> #
        >>> batch = harn._demo_batch(tag='train')
        >>> epoch_metrics = harn._demo_epoch(tag='vali', max_iter=4)
    """
    import sys
    import ndsampler

    config = SegmentationConfig(default=kw)
    config.load(cmdline=cmdline)
    nh.configure_hacks(config)  # fix opencv bugs

    assert config['datasets'] == 'special:camvid'

    coco_datasets = setup_coco_datasets()

    workdir = ub.ensuredir(ub.expandpath(config['workdir']))
    samplers = {
        # tag: ndsampler.CocoSampler(dset, workdir=workdir, backend='cog')
        tag: ndsampler.CocoSampler(dset, workdir=workdir, backend='npy')
        for tag, dset in coco_datasets.items()
    }
    torch_datasets = {
        tag: SegmentationDataset(
            sampler,
            config['input_dims'],
            input_overlap=((tag == 'train') and config['input_overlap']),
            augment=((tag == 'train') and config['augment']),
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
        class_weights = _precompute_class_weights(dset, mode=mode)
        class_weights = torch.FloatTensor(class_weights)
        class_weights[dset.classes.index('background')] = 0
    else:
        class_weights = None

    initializer_ = nh.Initializer.coerce(config)

    if config['arch'] == 'unet':
        # Note: UNet can get through 256x256 images at a rate of ~17Hz with
        # batch_size=8. This is pretty slow and can likely be improved by fixing
        # some of the weird padding / mirror stuff I have to do in unet to get
        # output_dims = input_dims.
        from netharn.models.unet import UNet
        model_ = (UNet, {
            'classes': torch_datasets['train'].classes,
            'in_channels': 3,
        })
    elif config['arch'] == 'segnet':
        from netharn.models.segnet import Segnet
        model_ = (Segnet, {
            'classes': torch_datasets['train'].classes,
            'in_channels': 3,
        })
    elif config['arch'] == 'psp':
        from netharn.models.psp import PSPNet_Resnet50_8s
        model_ = (PSPNet_Resnet50_8s, {
            'classes': torch_datasets['train'].classes,
            'in_channels': 3,
        })
    elif config['arch'] == 'deeplab':
        from netharn.models.deeplab import DeepLab_ASPP
        model_ = (DeepLab_ASPP, {
            'classes': torch_datasets['train'].classes,
            'in_channels': 3,
        })

    else:
        raise KeyError(config['arch'])

    if config['init'] == 'cls':
        initializer_ = model_[0]._initializer_cls()

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
        'num_keep': 5,
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

        python -m netharn.examples.sseg_camvid \
                --nice=camvid_segnet --arch=segnet --init=cls \
                --workers=4 --xpu=auto  \
                --batch_size=16 --lr=1e-3 \
                --input_dims=64,64

        python -m netharn.examples.sseg_camvid \
                --nice=camvid_psp --arch=psp --init=cls \
                --workers=4 --xpu=auto  \
                --batch_size=16 --lr=1e-3 \
                --input_dims=64,64

        python -m netharn.examples.sseg_camvid \
                --nice=camvid_deeplab --arch=deeplab --init=cls \
                --workers=4 --xpu=auto  \
                --batch_size=16 --lr=1e-3 \
                --input_dims=64,64
        """
    main()
