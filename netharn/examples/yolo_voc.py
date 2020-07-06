# -*- coding: utf-8 -*-
"""
References:
    https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import torch
import ubelt as ub
import numpy as np
import torch.utils.data as torch_data
import imgaug.augmenters as iaa
import netharn as nh
from netharn.models.yolo2 import multiscale_batch_sampler
from netharn.models.yolo2 import light_yolo
from netharn.data.transforms import HSVShift


class YoloVOCDataset(nh.data.voc.VOCDataset):
    """
    Extends VOC localization dataset (which simply loads the images in VOC2008
    with minimal processing) for multiscale training.

    CommandLine:
        python -m netharn.examples.yolo_voc YoloVOCDataset

    Example:
        >>> # DISABLE_DOCTSET
        >>> assert len(YoloVOCDataset(split='train', years=[2007])) == 2501
        >>> assert len(YoloVOCDataset(split='test', years=[2007])) == 4952
        >>> assert len(YoloVOCDataset(split='val', years=[2007])) == 2510
        >>> assert len(YoloVOCDataset(split='trainval', years=[2007])) == 5011

        >>> assert len(YoloVOCDataset(split='train', years=[2007, 2012])) == 8218
        >>> assert len(YoloVOCDataset(split='test', years=[2007, 2012])) == 4952
        >>> assert len(YoloVOCDataset(split='val', years=[2007, 2012])) == 8333

    Example:
        >>> # DISABLE_DOCTSET
        >>> self = YoloVOCDataset()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)
    """

    def __init__(self, devkit_dpath=None, split='train', years=[2007, 2012],
                 base_wh=[416, 416], scales=[-3, 6], factor=32):

        super(YoloVOCDataset, self).__init__(devkit_dpath, split=split,
                                             years=years)

        self.split = split

        self.factor = factor  # downsample factor of yolo grid

        self.base_wh = np.array(base_wh, dtype=np.int)

        assert np.all(self.base_wh % self.factor == 0)

        self.multi_scale_inp_size = np.array([
            self.base_wh + (self.factor * i) for i in range(*scales)])
        self.multi_scale_out_size = self.multi_scale_inp_size // self.factor

        self.augmenter = None

        if 'train' in split:
            augmentors = [
                # Order used in lightnet is hsv, rc, rf, lb
                # lb is applied externally to augmenters
                iaa.Sometimes(.9, HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Crop(percent=(0, .2), keep_size=False),
                iaa.Fliplr(p=.5),
            ]
            self.augmenter = iaa.Sequential(augmentors)

        # Used to resize images to the appropriate inp_size without changing
        # the aspect ratio.
        self.letterbox = nh.data.transforms.Resize(None, mode='letterbox')

    def __getitem__(self, index):
        """
        CommandLine:
            python -m netharn.examples.yolo_voc YoloVOCDataset.__getitem__ --show

        Example:
            >>> # DISABLE_DOCTSET
            >>> import kwimage
            >>> self = YoloVOCDataset(split='train')
            >>> index = 7
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> norm_boxes = label['targets'].numpy().reshape(-1, 5)[:, 1:5]
            >>> inp_size = hwc01.shape[-2::-1]
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(doclf=True, fnum=1)
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.imshow(hwc01, colorspace='rgb')
            >>> inp_boxes = kwimage.Boxes(norm_boxes, 'cxywh').scale(inp_size)
            >>> inp_boxes.draw()
            >>> kwplot.show_if_requested()

        Example:
            >>> # DISABLE_DOCTSET
            >>> import kwimage
            >>> self = YoloVOCDataset(split='test')
            >>> index = 0
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> norm_boxes = label[0].numpy().reshape(-1, 5)[:, 1:5]
            >>> inp_size = hwc01.shape[-2::-1]
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.figure(doclf=True, fnum=1)
            >>> kwplot.imshow(hwc01, colorspace='rgb')
            >>> inp_boxes = kwimage.Boxes(norm_boxes, 'cxywh').scale(inp_size)
            >>> inp_boxes.draw()
            >>> kwplot.show_if_requested()
        """
        import kwimage
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, size_index = index
            if size_index is None:
                inp_size = self.base_wh
            else:
                inp_size = self.multi_scale_inp_size[size_index]
        else:
            inp_size = self.base_wh
        inp_size = np.array(inp_size)

        image, tlbr, gt_classes, gt_weights = self._load_item(index)
        orig_size = np.array(image.shape[0:2][::-1])
        bbs = kwimage.Boxes(tlbr, 'tlbr').to_imgaug(shape=image.shape)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            image = seq_det.augment_image(image)
            bbs = seq_det.augment_bounding_boxes([bbs])[0]

            # Clip any bounding boxes that went out of bounds
            h, w = image.shape[0:2]
            tlbr = kwimage.Boxes.from_imgaug(bbs)

            old_area = tlbr.area
            tlbr = tlbr.clip(0, 0, w - 1, h - 1, inplace=True)
            new_area = tlbr.area

            # Remove any boxes that have gone significantly out of bounds.
            remove_thresh = 0.1
            flags = (new_area / old_area).ravel() > remove_thresh

            tlbr = tlbr.compress(flags, inplace=True)
            gt_classes = gt_classes[flags]
            gt_weights = gt_weights[flags]

            bbs = tlbr.to_imgaug(shape=image.shape)

        # Apply letterbox resize transform to train and test
        self.letterbox.target_size = inp_size
        image = self.letterbox.augment_image(image)
        bbs = self.letterbox.augment_bounding_boxes([bbs])[0]
        tlbr_inp = kwimage.Boxes.from_imgaug(bbs)

        # Remove any boxes that are no longer visible or out of bounds
        flags = (tlbr_inp.area > 0).ravel()
        tlbr_inp = tlbr_inp.compress(flags, inplace=True)
        gt_classes = gt_classes[flags]
        gt_weights = gt_weights[flags]

        chw01 = torch.FloatTensor(image.transpose(2, 0, 1) / 255.0)

        # Lightnet YOLO accepts truth tensors in the format:
        # [class_id, center_x, center_y, w, h]
        # where coordinates are noramlized between 0 and 1
        cxywh_norm = tlbr_inp.toformat('cxywh').scale(1 / inp_size)
        _target_parts = [gt_classes[:, None], cxywh_norm.data]
        target = np.concatenate(_target_parts, axis=-1)
        target = torch.FloatTensor(target)

        # Return index information in the label as well
        orig_size = torch.LongTensor(orig_size)
        index = torch.LongTensor([index])
        # how much do we care about each annotation in this image?
        gt_weights = torch.FloatTensor(gt_weights)
        # how much do we care about the background in this image?
        bg_weight = torch.FloatTensor([1.0])
        label = {
            'targets': target,
            'gt_weights': gt_weights,
            'orig_sizes': orig_size,
            'indices': index,
            'bg_weights': bg_weight
        }
        return chw01, label

    def _load_item(self, index):
        # load the raw data from VOC
        image = self._load_image(index)
        annot = self._load_annotation(index)
        # VOC loads annotations in tlbr
        tlbr = annot['boxes'].astype(np.float)
        gt_classes = annot['gt_classes']
        # Weight samples so we dont care about difficult cases
        gt_weights = 1.0 - annot['gt_ishard'].astype(np.float)
        return image, tlbr, gt_classes, gt_weights

    # @ub.memoize_method  # remove this if RAM is a problem
    def _load_image(self, index):
        return super(YoloVOCDataset, self)._load_image(index)

    # @ub.memoize_method
    def _load_annotation(self, index):
        return super(YoloVOCDataset, self)._load_annotation(index)

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=False, resize_rate=10, drop_last=False):
        """
        CommandLine:
            python -m netharn.examples.yolo_voc YoloVOCDataset.make_loader

        Example:
            >>> # DISABLE_DOCTSET
            >>> torch.random.manual_seed(0)
            >>> self = YoloVOCDataset(split='train')
            >>> self.augmenter = None
            >>> loader = self.make_loader(batch_size=1, shuffle=True)
            >>> # training batches should have multiple shapes
            >>> shapes = set()
            >>> for batch in ub.ProgIter(iter(loader), total=len(loader)):
            >>>     inputs, labels = batch
            >>>     # test to see multiscale works
            >>>     shapes.add(inputs.shape[-1])
            >>>     if len(shapes) > 1:
            >>>         break
            >>> assert len(shapes) > 1
        """
        import torch.utils.data.sampler as torch_sampler
        assert len(self) > 0, 'must have some data'
        if shuffle:
            sampler = torch_sampler.RandomSampler(self)
            resample_freq = resize_rate
        else:
            sampler = torch_sampler.SequentialSampler(self)
            resample_freq = None

        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            sampler, batch_size=batch_size, resample_freq=resample_freq,
            drop_last=drop_last,
        )
        # torch.utils.data.sampler.WeightedRandomSampler
        loader = torch_data.DataLoader(self, batch_sampler=batch_sampler,
                                       collate_fn=nh.data.collate.padded_collate,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)
        if loader.batch_size != batch_size:
            try:
                # Hack: ensure dataloader has batch size attr
                loader._DataLoader__initialized = False
                loader.batch_size = batch_size
                loader._DataLoader__initialized = True
            except Exception:
                pass
        return loader


class YoloHarn(nh.FitHarn):
    def __init__(harn, **kw):
        super(YoloHarn, harn).__init__(**kw)
        # Dictionary of detection metrics
        harn.dmets = {}  # Dict[str, nh.metrics.DetectionMetrics]
        harn.chosen_indices = {}

    def after_initialize(harn):
        # Prepare structures we will use to measure and quantify quality
        for tag, voc_dset in harn.datasets.items():
            dmet = nh.metrics.DetectionMetrics()
            dmet._pred_aidbase = getattr(dmet, '_pred_aidbase', 1)
            dmet._true_aidbase = getattr(dmet, '_true_aidbase', 1)
            harn.dmets[tag] = dmet

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        batch_inputs, batch_labels = raw_batch

        inputs = harn.xpu.move(batch_inputs)
        labels = {k: harn.xpu.move(d) for k, d in batch_labels.items()}

        batch = (inputs, labels)
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader

        CommandLine:
            python -m netharn.examples.yolo_voc YoloHarn.run_batch

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_yolo_harness(bsize=2)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
        """
        # Compute how many images have been seen before
        bsize = harn.loaders['train'].batch_sampler.batch_size
        nitems = (len(harn.datasets['train']) // bsize) * bsize
        bx = harn.bxs['train']
        n_seen = (bx * bsize) + (nitems * harn.epoch)

        inputs, labels = batch
        # if IS_PROFILING:
        #     torch.cuda.synchronize()

        outputs = harn.model(inputs)

        target2 = {
            'target': labels['targets'],
            'gt_weights': labels['gt_weights'],
        }
        # if IS_PROFILING:
        #     torch.cuda.synchronize()

        loss = harn.criterion(outputs, target2, seen=n_seen)

        # if IS_PROFILING:
        #     torch.cuda.synchronize()

        return outputs, loss

    def on_batch(harn, batch, outputs, loss):
        """
        custom callback

        CommandLine:
            python -m netharn.examples.yolo_voc YoloHarn.on_batch --gpu=0 --show

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_yolo_harness(bsize=8)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, loss)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> batch_dets = harn.model.module.postprocess(outputs)
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> stacked = harn.draw_batch(batch, outputs, batch_dets, thresh=0.01)
            >>> kwplot.imshow(stacked)
            >>> kwplot.show_if_requested()
        """
        dmet = harn.dmets[harn.current_tag]
        inputs, labels = batch
        inp_size = np.array(inputs.shape[-2:][::-1])

        # if IS_PROFILING:
        #     torch.cuda.synchronize()

        try:
            batch_dets = harn.model.module.postprocess(outputs)

            bx = harn.bxs[harn.current_tag]
            if bx < 4:
                import kwimage
                stacked = harn.draw_batch(batch, outputs, batch_dets, thresh=0.1)
                # img = kwplot.render_figure_to_image(fig)
                dump_dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag, 'batch'))
                dump_fname = 'pred_bx{:04d}_epoch{:08d}.png'.format(bx, harn.epoch)
                fpath = os.path.join(dump_dpath, dump_fname)
                harn.debug('dump viz fpath = {}'.format(fpath))
                kwimage.imwrite(fpath, stacked)
        except Exception as ex:
            harn.error('\n\n\n')
            harn.error('ERROR: FAILED TO POSTPROCESS OUTPUTS')
            harn.error('DETAILS: {!r}'.format(ex))
            raise

        # if IS_PROFILING:
        #     torch.cuda.synchronize()

        for gx, pred_dets in harn._postout_to_pred_dets(inp_size, labels, batch_dets, _aidbase=dmet._pred_aidbase):
            dmet._pred_aidbase += (len(pred_dets) + 1)
            dmet.add_predictions(pred_dets, gid=gx)

        for gx, true_dets in harn._labels_to_true_dets(inp_size, labels, _aidbase=dmet._true_aidbase):
            dmet._true_aidbase += (len(true_dets) + 1)
            dmet.add_truth(true_dets, gid=gx)

        # if IS_PROFILING:
        #     torch.cuda.synchronize()

        metrics_dict = ub.odict()
        metrics_dict['L_bbox'] = float(harn.criterion.loss_coord)
        metrics_dict['L_iou'] = float(harn.criterion.loss_conf)
        metrics_dict['L_cls'] = float(harn.criterion.loss_cls)
        for k, v in metrics_dict.items():
            if not np.isfinite(v):
                raise ValueError('{}={} is not finite'.format(k, v))
        return metrics_dict

    def _postout_to_pred_dets(harn, inp_size, labels, batch_dets, _aidbase=1,
                              undo_lb=True):
        """ Convert batch predictions to coco-style annotations for scoring """
        indices = labels['indices']
        orig_sizes = labels['orig_sizes']
        letterbox = harn.datasets[harn.current_tag].letterbox
        MAX_DETS = None
        bsize = len(indices)

        for ix in range(bsize):
            pred_dets = batch_dets[ix]
            # Unpack postprocessed predictions
            pred_dets = pred_dets.numpy()
            pred_dets.boxes.scale(inp_size, inplace=True)

            if undo_lb:
                orig_size = orig_sizes[ix].data.cpu().numpy()
                pred_dets.data['boxes'] = letterbox._boxes_letterbox_invert(
                    pred_dets.boxes, orig_size, inp_size)
            else:
                pred_dets.data['boxes'] = pred_dets.boxes

            # sort predictions by descending score

            # Take at most MAX_DETS detections to evaulate
            _pred_sortx = pred_dets.argsort(reverse=True)[:MAX_DETS]
            pred_dets = pred_dets.take(_pred_sortx)

            pred_dets.data['aids'] = np.arange(_aidbase, _aidbase + len(pred_dets))
            _aidbase += len(pred_dets)
            gx = int(indices[ix].data)

            # if IS_PROFILING:
            #     torch.cuda.synchronize()

            yield gx, pred_dets

    def _labels_to_true_dets(harn, inp_size, labels, _aidbase=1, undo_lb=True):
        """ Convert batch groundtruth to coco-style annotations for scoring """
        import kwimage
        indices = labels['indices']
        orig_sizes = labels['orig_sizes']
        targets = labels['targets']
        gt_weights = labels['gt_weights']

        letterbox = harn.datasets[harn.current_tag].letterbox
        # On the training set, we need to add truth due to augmentation
        bsize = len(indices)
        for ix in range(bsize):
            target = targets[ix].view(-1, 5)

            true_det = kwimage.Detections(
                boxes=kwimage.Boxes(target[:, 1:5].float(), 'cxywh'),
                class_idxs=target[:, 0].long(),
                weights=gt_weights[ix],
            )
            true_det = true_det.numpy()
            flags = true_det.class_idxs != -1
            true_det = true_det.compress(flags)

            if undo_lb:
                orig_size = orig_sizes[ix].cpu().numpy()
                true_det.data['boxes'] = letterbox._boxes_letterbox_invert(
                    true_det.boxes, orig_size, inp_size)

            true_det.data['aids'] = np.arange(_aidbase, _aidbase + len(true_det))
            gx = int(indices[ix].data.cpu().numpy())

            # if IS_PROFILING:
            #     torch.cuda.synchronize()

            yield gx, true_det

    def on_epoch(harn):
        """
        custom callback

        CommandLine:
            python -m netharn.examples.yolo_voc YoloHarn.on_epoch

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_yolo_harness(bsize=4)
            >>> harn.initialize()
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> tag = harn.current_tag = 'test'
            >>> # run a few batches
            >>> for i in ub.ProgIter(range(5)):
            ...     batch = harn._demo_batch(i, tag)
            ...     outputs, loss = harn.run_batch(batch)
            ...     harn.on_batch(batch, outputs, loss)
            >>> # then finish the epoch
            >>> harn.on_epoch()
        """
        metrics_dict = ub.odict()

        # Measure quality
        dmet = harn.dmets[harn.current_tag]
        # try:
        #     coco_scores = dmet.score_coco()
        #     metrics_dict['coco-mAP'] = coco_scores['mAP']
        # except ImportError:
        #     pass
        # except Exception as ex:
        #     print('ex = {!r}'.format(ex))

        # try:
        #     nh_scores = dmet.score_netharn()
        #     metrics_dict['nh-mAP'] = nh_scores['mAP']
        #     metrics_dict['nh-AP'] = nh_scores['peritem']['ap']
        # except Exception as ex:
        #     print('ex = {!r}'.format(ex))

        try:
            voc_scores = dmet.score_voc()
            metrics_dict['voc-mAP'] = voc_scores['mAP']
        except Exception as ex:
            print('ex = {!r}'.format(ex))
            raise

        # Reset detections
        dmet.clear()
        dmet._pred_aidbase = 1
        dmet._true_aidbase = 1
        return metrics_dict

    def draw_batch(harn, batch, outputs, batch_dets, idx=None, thresh=None,
                   orig_img=None):
        """
        Returns:
            np.ndarray: numpy image

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_yolo_harness(bsize=1)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, loss)
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> batch_dets = harn.model.module.postprocess(outputs)
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> stacked = harn.draw_batch(batch, outputs, batch_dets, thresh=0.01)
            >>> kwplot.imshow(stacked)
            >>> kwplot.show_if_requested()
        """
        import cv2
        import kwimage
        inputs, labels = batch

        targets = labels['targets']
        orig_sizes = labels['orig_sizes']

        if idx is None:
            idxs = range(len(inputs))
        else:
            idxs = [idx]

        imgs = []
        for idx in idxs:
            chw01 = inputs[idx]
            target = targets[idx].view(-1, 5)

            pred_dets = batch_dets[idx]
            label_names = harn.datasets[harn.current_tag].label_names
            pred_dets.meta['classes'] = label_names

            true_dets = kwimage.Detections(
                boxes=kwimage.Boxes(target[:, 1:5], 'cxywh'),
                class_idxs=target[:, 0].int(),
                classes=label_names
            )

            pred_dets = pred_dets.numpy()
            true_dets = true_dets.numpy()

            true_dets = true_dets.compress(true_dets.class_idxs != -1)

            if thresh is not None:
                pred_dets = pred_dets.compress(pred_dets.scores > thresh)

            hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
            inp_size = np.array(hwc01.shape[0:2][::-1])

            true_dets.boxes.scale(inp_size, inplace=True)
            pred_dets.boxes.scale(inp_size, inplace=True)

            letterbox = harn.datasets[harn.current_tag].letterbox
            orig_size = orig_sizes[idx].cpu().numpy()
            target_size = inp_size
            img = letterbox._img_letterbox_invert(hwc01, orig_size, target_size)
            img = np.clip(img, 0, 1)
            # we are given the original image, to avoid artifacts from
            # inverting a downscale
            assert orig_img is None or orig_img.shape == img.shape

            true_dets.data['boxes'] = letterbox._boxes_letterbox_invert(
                true_dets.boxes, orig_size, target_size)
            pred_dets.data['boxes'] = letterbox._boxes_letterbox_invert(
                pred_dets.boxes, orig_size, target_size)

            # shift, scale, embed_size = letterbox._letterbox_transform(orig_size, target_size)
            # fig = kwplot.figure(doclf=True, fnum=1)
            # kwplot.imshow(img, colorspace='rgb')
            canvas = (img * 255).astype(np.uint8)
            canvas = true_dets.draw_on(canvas, color='green')
            canvas = pred_dets.draw_on(canvas, color='blue')

            canvas = cv2.resize(canvas, (300, 300))
            imgs.append(canvas)

        # if IS_PROFILING:
        #     torch.cuda.synchronize()

        stacked = imgs[0] if len(imgs) == 1 else kwimage.stack_images_grid(imgs)
        return stacked


def setup_yolo_harness(bsize=16, workers=0):
    """
    CommandLine:
        python -m netharn.examples.yolo_voc setup_yolo_harness

    Example:
        >>> # DISABLE_DOCTSET
        >>> harn = setup_yolo_harness()
        >>> harn.initialize()
    """

    xpu = nh.XPU.coerce('argv')

    nice = ub.argval('--nice', default='Yolo2Baseline')
    batch_size = int(ub.argval('--batch_size', default=bsize))
    bstep = int(ub.argval('--bstep', 4))
    workers = int(ub.argval('--workers', default=workers))
    decay = float(ub.argval('--decay', default=0.0005))
    lr = float(ub.argval('--lr', default=0.001))
    ovthresh = 0.5
    simulated_bsize = bstep * batch_size

    nh.configure_hacks(workers=workers)

    # We will divide the learning rate by the simulated batch size
    datasets = {
        'train': YoloVOCDataset(years=[2007, 2012], split='trainval'),
        # 'test': YoloVOCDataset(years=[2007], split='test'),
    }
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=True,
                              resize_rate=10 * bstep, drop_last=True)
        for key, dset in datasets.items()
    }

    anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                        (5.05587, 8.09892), (9.47112, 4.84053),
                        (11.2364, 10.0071)])

    if not ub.argflag('--eav'):
        lr_step_points = {
            # 0:   lr * 0.1 / simulated_bsize,  # burnin
            # 4:   lr * 1.0 / simulated_bsize,
            0:   lr * 1.0 / simulated_bsize,
            154: lr * 1.0 / simulated_bsize,
            155: lr * 0.1 / simulated_bsize,
            232: lr * 0.1 / simulated_bsize,
            233: lr * 0.01 / simulated_bsize,
        }
        max_epoch = 311
        scheduler_ = (nh.schedulers.core.YOLOScheduler, {
            'points': lr_step_points,
            # 'interpolate': False,
            'interpolate': True,
            'burn_in': 0.96899225 if ub.argflag('--eav') else 3.86683584,  # number of epochs to burn_in for. approx 1000 batches?
            'dset_size': len(datasets['train']),  # when drop_last=False
            # 'dset_size': (len(datasets['train']) // simulated_bsize) * simulated_bsize,  # make a multiple of batch_size because drop_last=True
            'batch_size': batch_size,
        })
        from netharn.models.yolo2 import light_region_loss
        criterion_ = (light_region_loss.RegionLoss, {
            'num_classes': datasets['train'].num_classes,
            'anchors': anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,

            # eav version originally had a random *2 in cls loss,
            # we removed, that but we can replicate it here.
            'class_scale': 1.0 if not ub.argflag('--eav') else 2.0,
            'coord_scale': 1.0,

            'thresh': 0.6,  # iou_thresh
            'seen_thresh': 12800,
            # 'small_boxes': not ub.argflag('--eav'),
            'small_boxes': True,
            'mse_factor': 0.5 if not ub.argflag('--eav') else 1.0,
        })
    else:
        lr_step_points = {
            # dividing by batch size was one of those unpublished details
            0:   lr * 0.1 / simulated_bsize,
            1:   lr * 1.0 / simulated_bsize,

            96:  lr * 1.0 / simulated_bsize,
            97:  lr * 0.1 / simulated_bsize,

            135: lr * 0.1 / simulated_bsize,
            136: lr * 0.01 / simulated_bsize,
        }
        max_epoch = 176
        scheduler_ = (nh.schedulers.ListedLR, {
            'points': lr_step_points,
            'interpolate': False,
        })
        from netharn.models.yolo2 import region_loss2
        criterion_ = (region_loss2.RegionLoss, {
            'num_classes': datasets['train'].num_classes,
            'anchors': anchors,
            'reduction': 32,
            'seen': 0,
            'coord_scale'    : 1.0,
            'noobject_scale' : 1.0,
            'object_scale'   : 5.0,
            'class_scale'    : 1.0,
            'thresh'         : 0.6,  # iou_thresh
            # 'seen_thresh': 12800,
        })

    weights = ub.argval('--weights', default=None)
    if weights is None or weights == 'imagenet':
        weights = light_yolo.initial_imagenet_weights()
    elif weights == 'lightnet':
        weights = light_yolo.demo_voc_weights()
    else:
        print('weights = {!r}'.format(weights))

    hyper = nh.HyperParams(**{
        'nice': nice,
        'workdir': ub.expandpath('~/work/voc_yolo2'),

        'datasets': datasets,
        'loaders': loaders,

        'xpu': xpu,

        'model': (light_yolo.Yolo, {
            'num_classes': datasets['train'].num_classes,
            'anchors': anchors,
            'conf_thresh': 0.001,
            # 'conf_thresh': 0.1,  # make training a bit faster
            'nms_thresh': 0.5 if not ub.argflag('--eav') else 0.4
        }),

        'criterion': criterion_,

        'initializer': (nh.initializers.Pretrained, {
            'fpath': weights,
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': lr_step_points[0],
            'momentum': 0.9,
            'dampening': 0,
            # multiplying by batch size was one of those unpublished details
            'weight_decay': decay * simulated_bsize,
        }),

        'scheduler': scheduler_,

        'monitor': (nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['mAP'],
            'patience': max_epoch,
            'max_epoch': max_epoch,
        }),

        # 'augment': datasets['train'].augmenter,

        'dynamics': {
            # Controls how many batches to process before taking a step in the
            # gradient direction. Effectively simulates a batch_size that is
            # `bstep` times bigger.
            'batch_step': bstep,
        },

        'other': {
            # Other params are not used internally, so you are free to set any
            # extra params specific to your algorithm, and still have them
            # logged in the hyperparam structure. For YOLO this is `ovthresh`.
            'batch_size': batch_size,
            'nice': nice,
            'ovthresh': ovthresh,  # used in mAP computation
            'input_range': 'norm01',
        },
    })
    print('max_epoch = {!r}'.format(max_epoch))
    harn = YoloHarn(hyper=hyper)
    harn.preferences['prog_backend'] = 'progiter'
    harn.intervals['log_iter_train'] = None
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None
    harn.preferences['large_loss'] = 1000  # tell netharn when to check for divergence
    return harn


def train():
    harn = setup_yolo_harness()
    harn.initialize()
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        srun -c 4 -p priority --gres=gpu:1 \
            python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=16 --nice=rescaled --lr=0.001 --bstep=4 --workers=4

        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=16 --nice=new_loss_v2 --lr=0.001 --bstep=4 --workers=4

        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=16 --nice=eav_run --lr=0.001 --bstep=4 --workers=6 --eav
        python -m netharn.examples.yolo_voc train --gpu=1 --batch_size=16 --nice=pjr_run2 --lr=0.001 --bstep=4 --workers=6

        python -m netharn.examples.yolo_voc train --gpu=1 --batch_size=16 --nice=fixed_nms --lr=0.001 --bstep=4 --workers=6

        python -m netharn.examples.yolo_voc train --gpu=1 --batch_size=16 --nice=fixed_lrs --lr=0.001 --bstep=4 --workers=6


        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=eav_run2 --lr=0.001 --bstep=4 --workers=8 --eav
        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=pjr_run2 --lr=0.001 --bstep=4 --workers=4

        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=4 --nice=pjr_run2 --lr=0.001 --bstep=8 --workers=4

        python -m netharn.examples.yolo_voc train --gpu=0,1 --batch_size=32 --nice=july23 --lr=0.001 --bstep=2 --workers=8
        python -m netharn.examples.yolo_voc train --gpu=2 --batch_size=16 --nice=july23_lr_x8 --lr=0.008 --bstep=4 --workers=6

        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=batchaware2 --lr=0.001 --bstep=8 --workers=3

        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=july_eav_run3 --lr=0.001 --bstep=8 --workers=6 --eav
        python -m netharn.examples.yolo_voc train --gpu=1 --batch_size=8 --nice=july_eav_run4 --lr=0.002 --bstep=8 --workers=6 --eav
        python -m netharn.examples.yolo_voc train --gpu=2 --batch_size=16 --nice=july_pjr_run4 --lr=0.001 --bstep=4 --workers=6


        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=july_eav_run4_hack1 --lr=0.001 --bstep=8 --workers=6 --eav --weights=/home/local/KHQ/jon.crall/work/voc_yolo2/fit/nice/july_eav_run_hack/torch_snapshots/_epoch_00000150.pt

        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=lightnet_start --lr=0.001 --bstep=8 --workers=6 --eav --weights=lightnet


        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=HOPE --lr=0.001 --bstep=8 --workers=6 --eav --weights=imagenet
        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=HOPE2 --lr=0.001 --bstep=8 --workers=6 --eav --weights=imagenet
        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=HOPE3 --lr=0.001 --bstep=8 --workers=4 --eav --weights=imagenet

        python -m netharn.examples.yolo_voc train --gpu=0 --batch_size=8 --nice=HOPE4 --lr=0.001 --bstep=8 --workers=4 --eav --weights=imagenet


        python -m netharn.examples.yolo_voc train --gpu=0 --workers=4 --weights=lightnet
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
