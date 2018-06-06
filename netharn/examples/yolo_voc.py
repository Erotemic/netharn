# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import torch
import ubelt as ub
import numpy as np
import pandas as pd
import netharn as nh
from netharn import util
import imgaug.augmenters as iaa
from netharn.util import profiler  # NOQA
import torch.utils.data as torch_data
from netharn.models.yolo2 import multiscale_batch_sampler
from netharn.models.yolo2 import light_region_loss
from netharn.models.yolo2 import light_yolo


class YoloVOCDataset(nh.data.voc.VOCDataset):
    """
    Extends VOC localization dataset (which simply loads the images in VOC2008
    with minimal processing) for multiscale training.

    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py YoloVOCDataset

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

        # Original YOLO Anchors
        # self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
        #                            (6.63, 11.38), (9.42, 5.11),
        #                            (16.62, 10.52)],
        #                           dtype=np.float)

        # Lightnet Anchors
        self.anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                                 (5.05587, 8.09892), (9.47112, 4.84053),
                                 (11.2364, 10.0071)])
        self.num_anchors = len(self.anchors)
        self.augmenter = None

        if 'train' in split:
            import netharn.data.transforms  # NOQA
            from netharn.data.transforms import HSVShift
            augmentors = [
                # iaa.Flipud(p=.5),
                # iaa.Affine(
                #     # scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
                #     # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                #     rotate=(-3.6, 3.6),
                #     # rotate=(-15, 15),
                #     # shear=(-7, 7),
                #     # order=[0, 1, 3],
                #     order=1,
                #     # cval=(0, 255),
                #     cval=127,
                #     mode=ia.ALL,
                #     backend='cv2',
                # ),
                # Order used in lightnet is hsv, rc, rf, lb
                # lb is applied externally to augmenters
                HSVShift(hue=0.1, sat=1.5, val=1.5),
                iaa.Crop(percent=(0, .2)),
                iaa.Fliplr(p=.5),
                # iaa.AddToHueAndSaturation((-20, 20)),
                # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                # iaa.AddToHueAndSaturation((-15, 15)),
                # iaa.ContrastNormalization((0.75, 1.5))
                # iaa.ContrastNormalization((0.75, 1.5), per_channel=0.5),
            ]
            self.augmenter = iaa.Sequential(augmentors)

        # Used to resize images to the appropriate inp_size without changing
        # the aspect ratio.
        self.letterbox = nh.data.transforms.Resize(None, mode='letterbox')

    # def __len__(self):
    #     # hack
    #     if 'train' in self.split:
    #         return 100
    #     else:
    #         return super().__len__()

    @profiler.profile
    def __getitem__(self, index):
        """
        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloVOCDataset.__getitem__ --show

        Example:
            >>> # DISABLE_DOCTSET
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.truepath('~/code/netharn/netharn/examples'))
            >>> from yolo_voc import *
            >>> self = YoloVOCDataset(split='train')
            >>> index = 1
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> norm_boxes = label[0].numpy().reshape(-1, 5)[:, 1:5]
            >>> inp_size = hwc01.shape[-2::-1]
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> inp_boxes = util.Boxes(norm_boxes, 'cxywh').scale(inp_size).data
            >>> mplutil.draw_boxes(inp_boxes, box_format='cxywh')
            >>> mplutil.show_if_requested()

        Example:
            >>> # DISABLE_DOCTSET
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.truepath('~/code/netharn/netharn/examples'))
            >>> from yolo_voc import *
            >>> self = YoloVOCDataset(split='test')
            >>> index = 0
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> norm_boxes = label[0].numpy().reshape(-1, 5)[:, 1:5]
            >>> inp_size = hwc01.shape[-2::-1]
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> inp_boxes = util.Boxes(norm_boxes, 'cxywh').scale(inp_size).data
            >>> mplutil.draw_boxes(inp_boxes, box_format='cxywh')
            >>> mplutil.show_if_requested()

        Ignore:
            >>> self = YoloVOCDataset(split='train')

            for index in ub.ProgIter(range(len(self))):
                chw01, label = self[index]
                target = label[0]
                wh = target[:, 3:5]
                if np.any(wh == 0):
                    raise ValueError()
                pass

            >>> # Check that we can collate this data
            >>> self = YoloVOCDataset(split='train')
            >>> inbatch = [self[index] for index in range(0, 16)]
            >>> from netharn.data import collate
            >>> batch = collate.padded_collate(inbatch)
            >>> inputs, labels = batch
            >>> assert len(labels) == len(inbatch[0][1])
            >>> targets = labels['targets']
            >>> orig_sizes = labels['orig_sizes']
            >>> gt_weights = labels['gt_weights']
            >>> indices = labels['indices']
            >>> bg_weights = labels['bg_weights']
            >>> assert list(target.shape) == [16, 6, 5]
            >>> assert list(gt_weights.shape) == [16, 6]
            >>> assert list(origsize.shape) == [16, 2]
            >>> assert list(index.shape) == [16, 1]
        """
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
        bbs = util.Boxes(tlbr, 'tlbr').to_imgaug(shape=image.shape)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            image = seq_det.augment_image(image)
            bbs = seq_det.augment_bounding_boxes([bbs])[0]

            # Clip any bounding boxes that went out of bounds
            h, w = image.shape[0:2]
            tlbr = util.Boxes.from_imgaug(bbs)
            tlbr = tlbr.clip(0, 0, w - 1, h - 1, inplace=True)

            # Remove any boxes that are no longer visible or out of bounds
            flags = (tlbr.area > 0).ravel()
            tlbr = tlbr.compress(flags, inplace=True)
            gt_classes = gt_classes[flags]
            gt_weights = gt_weights[flags]

            bbs = tlbr.to_imgaug(shape=image.shape)

        # Apply letterbox resize transform to train and test
        self.letterbox.target_size = inp_size
        image = self.letterbox.augment_image(image)
        bbs = self.letterbox.augment_bounding_boxes([bbs])[0]
        tlbr_inp = util.Boxes.from_imgaug(bbs)

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
                    pin_memory=False):
        """
        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloVOCDataset.make_loader

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
            resample_freq = 10
        else:
            sampler = torch_sampler.SequentialSampler(self)
            resample_freq = None

        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            sampler, batch_size=batch_size, resample_freq=resample_freq,
        )
        # torch.utils.data.sampler.WeightedRandomSampler
        loader = torch_data.DataLoader(self, batch_sampler=batch_sampler,
                                       collate_fn=nh.data.collate.padded_collate,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)
        if loader.batch_size != batch_size:
            try:
                loader.batch_size = batch_size
            except Exception:
                pass
        return loader


class YoloHarn(nh.FitHarn):
    def __init__(harn, **kw):
        super().__init__(**kw)
        harn.batch_confusions = []
        harn.aps = {}

        harn.chosen_indices = {}

    # def initialize(harn):
    #     super().initialize()
    #     harn.datasets['train']._augmenter = harn.datasets['train'].augmenter
    #     if harn.epoch <= 0:
    #         # disable augmenter for the first epoch
    #         harn.datasets['train'].augmenter = None

    @profiler.profile
    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        batch_inputs, batch_labels = raw_batch

        inputs = harn.xpu.variable(batch_inputs)
        labels = {k: harn.xpu.variable(d) for k, d in batch_labels.items()}

        batch = (inputs, labels)
        return batch

    @profiler.profile
    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader

        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloHarn.run_batch

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_harness(bsize=2)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
        """

        # Compute how many images have been seen before
        bsize = harn.loaders['train'].batch_sampler.batch_size
        nitems = len(harn.datasets['train'])
        bx = harn.bxs['train']
        n_seen = (bx * bsize) + (nitems * harn.epoch)

        inputs, labels = batch
        outputs = harn.model(inputs)
        # torch.cuda.synchronize()
        target = labels['targets']
        gt_weights = labels['gt_weights']
        loss = harn.criterion(outputs, target, seen=n_seen,
                              gt_weights=gt_weights)
        # torch.cuda.synchronize()
        return outputs, loss

    @profiler.profile
    def on_batch(harn, batch, outputs, loss):
        """
        custom callback

        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloHarn.on_batch --gpu=0 --show

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_harness(bsize=1)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, loss)
            >>> # xdoc: +REQUIRES(--show)
            >>> postout = harn.model.module.postprocess(outputs)
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> harn.visualize_prediction(batch, outputs, postout, idx=0, thresh=0.01)
            >>> mplutil.show_if_requested()
        """
        if harn.current_tag != 'train':
            # Dont worry about computing mAP on the training set for now
            inputs, labels = batch
            inp_size = np.array(inputs.shape[-2:][::-1])

            try:
                postout = harn.model.module.postprocess(outputs)
            except Exception as ex:
                harn.error('\n\n\n')
                harn.error('ERROR: FAILED TO POSTPROCESS OUTPUTS')
                harn.error('DETAILS: {!r}'.format(ex))
                raise

            for y in harn._measure_confusion(postout, labels, inp_size):
                harn.batch_confusions.append(y)

        metrics_dict = ub.odict()
        metrics_dict['L_bbox'] = float(harn.criterion.loss_coord)
        metrics_dict['L_iou'] = float(harn.criterion.loss_conf)
        metrics_dict['L_cls'] = float(harn.criterion.loss_cls)
        for k, v in metrics_dict.items():
            if not np.isfinite(v):
                raise ValueError('{}={} is not finite'.format(k, v))
        return metrics_dict

    @profiler.profile
    def on_epoch(harn):
        """
        custom callback

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_harness(bsize=4)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> # run a few batches
            >>> harn.on_batch(batch, outputs, loss)
            >>> harn.on_batch(batch, outputs, loss)
            >>> harn.on_batch(batch, outputs, loss)
            >>> # then finish the epoch
            >>> harn.on_epoch()
        """
        tag = harn.current_tag

        if tag in {'test', 'vali'}:
            harn._dump_chosen_indices()

        if harn.batch_confusions:
            y = pd.concat([pd.DataFrame(y) for y in harn.batch_confusions])

            precision, recall, ap = nh.metrics.detections._multiclass_ap(y)

            # TODO: write out a few visualizations
            loader = harn.loaders[tag]
            num_classes = len(loader.dataset.label_names)
            labels = list(range(num_classes))
            aps = nh.metrics.ave_precisions(y, labels, use_07_metric=True)
            harn.aps[tag] = aps
            mean_ap = np.nanmean(aps['ap'])
            max_ap = np.nanmax(aps['ap'])
            harn.log_value(tag + ' epoch mAP', mean_ap, harn.epoch)
            harn.log_value(tag + ' epoch max-AP', max_ap, harn.epoch)
            harn.batch_confusions.clear()
            metrics_dict = ub.odict()
            metrics_dict['max-AP'] = max_ap
            metrics_dict['mAP'] = mean_ap
            metrics_dict['AP'] = ap
            return metrics_dict

    # Non-standard problem-specific custom methods

    @profiler.profile
    def _measure_confusion(harn, postout, labels, inp_size, **kw):
        targets = labels['targets']
        gt_weights = labels['gt_weights']
        bg_weights = labels['bg_weights']
        # orig_sizes = labels['orig_sizes']
        # indices = labels['indices']

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        bsize = len(targets)
        for bx in range(bsize):
            postitem = asnumpy(postout[bx])
            target = asnumpy(targets[bx]).reshape(-1, 5)
            true_cxywh   = target[:, 1:5]
            true_cxs     = target[:, 0]
            true_weight  = asnumpy(gt_weights[bx])

            # Remove padded truth
            flags = true_cxs != -1
            true_cxywh  = true_cxywh[flags]
            true_cxs    = true_cxs[flags]
            true_weight = true_weight[flags]

            # orig_size    = asnumpy(orig_sizes[bx])
            # gx           = int(asnumpy(indices[bx]))

            # how much do we care about the background in this image?
            bg_weight = float(asnumpy(bg_weights[bx]))

            # Unpack postprocessed predictions
            sboxes = postitem.reshape(-1, 6)
            pred_cxywh = sboxes[:, 0:4]
            pred_scores = sboxes[:, 4]
            pred_cxs = sboxes[:, 5].astype(np.int)

            true_tlbr = util.Boxes(true_cxywh, 'cxywh').to_tlbr()
            pred_tlbr = util.Boxes(pred_cxywh, 'cxywh').to_tlbr()

            true_tlbr = true_tlbr.scale(inp_size)
            pred_tlbr = pred_tlbr.scale(inp_size)

            # TODO: can we invert the letterbox transform here and clip for
            # some extra mAP?
            true_boxes = true_tlbr.data
            pred_boxes = pred_tlbr.data

            y = nh.metrics.detection_confusions(
                true_boxes=true_boxes,
                true_cxs=true_cxs,
                true_weights=true_weight,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_cxs=pred_cxs,
                bg_weight=bg_weight,
                bg_cls=-1,
                ovthresh=harn.hyper.other['ovthresh'],
                **kw
            )
            # y['gx'] = gx
            yield y

    def _postout_to_coco(harn, postout, labels, inp_size):
        """
        -[ ] TODO: dump predictions for the test set to disk and score using
             someone elses code.
        """
        targets = labels['targets']
        gt_weights = labels['gt_weights']
        # orig_sizes = labels['orig_sizes']
        indices = labels['indices']
        orig_sizes = labels['orig_sizes']
        # bg_weights = labels['bg_weights']

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        def undo_letterbox(cxywh):
            boxes = util.Boxes(cxywh, 'cxywh')
            letterbox = harn.datasets['train'].letterbox
            return letterbox._boxes_letterbox_invert(boxes, orig_size, inp_size)

        predictions = []
        truth = []

        bsize = len(targets)
        for bx in range(bsize):
            postitem = asnumpy(postout[bx])
            target = asnumpy(targets[bx]).reshape(-1, 5)
            true_cxywh   = target[:, 1:5]
            true_cxs     = target[:, 0]
            true_weight  = asnumpy(gt_weights[bx])

            # Remove padded truth
            flags = true_cxs != -1
            true_cxywh  = true_cxywh[flags]
            true_cxs    = true_cxs[flags]
            true_weight = true_weight[flags]

            orig_size = asnumpy(orig_sizes[bx])
            gx        = int(asnumpy(indices[bx]))

            # how much do we care about the background in this image?
            # bg_weight = float(asnumpy(bg_weights[bx]))

            # Unpack postprocessed predictions
            sboxes = postitem.reshape(-1, 6)
            pred_cxywh = sboxes[:, 0:4]
            pred_scores = sboxes[:, 4]
            pred_cxs = sboxes[:, 5].astype(np.int)

            true_xywh = undo_letterbox(true_cxywh).toformat('xywh').data
            pred_xywh = undo_letterbox(pred_cxywh).toformat('xywh').data

            for xywh, cx, score in zip(pred_xywh, pred_cxs, pred_scores):
                pred = {
                    'image_id': gx,
                    'category_id': cx,
                    'bbox': list(xywh),
                    'score': score,
                }
                predictions.append(pred)

            for xywh, cx, weight in zip(true_xywh, true_cxs, gt_weights):
                true = {
                    'image_id': gx,
                    'category_id': cx,
                    'bbox': list(xywh),
                    'weight': weight,
                }
                truth.append(true)
        return predictions, truth

    def visualize_prediction(harn, batch, outputs, postout, idx=0, thresh=None,
                             orig_img=None):
        """
        Returns:
            np.ndarray: numpy image
        """
        # xdoc: +REQUIRES(--show)
        inputs, labels = batch

        targets = labels['targets']
        orig_sizes = labels['orig_sizes']

        chw01 = inputs[idx]
        target = targets[idx].cpu().numpy().reshape(-1, 5)
        postitem = postout[idx].cpu().numpy().reshape(-1, 6)
        orig_size = orig_sizes[idx].cpu().numpy()
        # ---
        hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
        # TRUE
        true_cxs = target[:, 0].astype(np.int)
        true_cxywh = target[:, 1:5]
        flags = true_cxs != -1
        true_cxywh = true_cxywh[flags]
        true_cxs = true_cxs[flags]
        # PRED
        pred_cxywh = postitem[:, 0:4]
        pred_scores = postitem[:, 4]
        pred_cxs = postitem[:, 5].astype(np.int)

        if thresh is not None:
            flags = pred_scores > thresh
            pred_cxs = pred_cxs[flags]
            pred_cxywh = pred_cxywh[flags]
            pred_scores = pred_scores[flags]

        label_names = harn.datasets['train'].label_names

        true_clsnms = list(ub.take(label_names, true_cxs))
        pred_clsnms = list(ub.take(label_names, pred_cxs))
        pred_labels = ['{}@{:.2f}'.format(n, s)
                       for n, s in zip(pred_clsnms, pred_scores)]
        # ---
        inp_size = np.array(hwc01.shape[0:2][::-1])
        target_size = inp_size

        true_boxes_ = nh.util.Boxes(true_cxywh, 'cxywh').scale(inp_size)
        pred_boxes_ = nh.util.Boxes(pred_cxywh, 'cxywh').scale(inp_size)

        letterbox = harn.datasets['train'].letterbox
        img = letterbox._img_letterbox_invert(hwc01, orig_size, target_size)
        img = np.clip(img, 0, 1)
        if orig_img is not None:
            # we are given the original image, to avoid artifacts from
            # inverting a downscale
            assert orig_img.shape == img.shape

        true_cxywh_ = letterbox._boxes_letterbox_invert(true_boxes_, orig_size, target_size)
        pred_cxywh_ = letterbox._boxes_letterbox_invert(pred_boxes_, orig_size, target_size)

        shift, scale, embed_size = letterbox._letterbox_transform(orig_size, target_size)

        fig = nh.util.figure(doclf=True, fnum=1)
        nh.util.imshow(img, colorspace='rgb')
        nh.util.draw_boxes(true_cxywh_.data, color='green', box_format='cxywh', labels=true_clsnms)
        nh.util.draw_boxes(pred_cxywh_.data, color='blue', box_format='cxywh', labels=pred_labels)
        return fig

    def _choose_indices(harn):
        """
        Hack to pick several images from the validation set to monitor each
        epoch.
        """
        tag = harn.current_tag
        dset = harn.loaders[tag].dataset

        cid_to_gids = ub.ddict(set)
        empty_gids = []
        for gid in range(len(dset)):
            annots = dset._load_annotation(gid)
            if len(annots['gt_classes']) == 0:
                empty_gids.append(gid)
            for cid, ishard in zip(annots['gt_classes'], annots['gt_ishard']):
                if not ishard:
                    cid_to_gids[cid].add(gid)

        # Choose an image with each category
        chosen_gids = set()
        for cid, gids in cid_to_gids.items():
            for gid in gids:
                if gid not in chosen_gids:
                    chosen_gids.add(gid)
                    break

        # Choose an image with nothing in it (if it exists)
        if empty_gids:
            chosen_gids.add(empty_gids[0])

        chosen_indices = chosen_gids
        harn.chosen_indices[tag] = sorted(chosen_indices)

    def _dump_chosen_indices(harn):
        """
        Dump a visualization of the validation images to disk
        """
        tag = harn.current_tag
        harn.debug('DUMP CHOSEN INDICES')

        if tag not in harn.chosen_indices:
            harn._choose_indices()

        nh.util.mplutil.aggensure()

        dset = harn.loaders[tag].dataset
        for indices in ub.chunks(harn.chosen_indices[tag], 16):
            harn.debug('PREDICTING CHUNK')
            inbatch = [dset[index] for index in indices]
            raw_batch = nh.data.collate.padded_collate(inbatch)
            batch = harn.prepare_batch(raw_batch)
            outputs, loss = harn.run_batch(batch)
            postout = harn.model.module.postprocess(outputs)

            for idx, index in enumerate(indices):
                orig_img = dset._load_image(index)
                fig = harn.visualize_prediction(batch, outputs, postout, idx=idx,
                                                thresh=0.1, orig_img=orig_img)
                img = nh.util.mplutil.render_figure_to_image(fig)
                dump_dpath = ub.ensuredir((harn.train_dpath, 'dump', tag))
                dump_fname = 'pred_{:04d}_{:08d}.png'.format(index, harn.epoch)
                fpath = os.path.join(dump_dpath, dump_fname)
                harn.debug('dump viz fpath = {}'.format(fpath))
                nh.util.imwrite(fpath, img)

    def dump_batch_item(harn, batch, outputs, postout):
        fig = harn.visualize_prediction(batch, outputs, postout, idx=0,
                                        thresh=0.2)
        img = nh.util.mplutil.render_figure_to_image(fig)
        dump_dpath = ub.ensuredir((harn.train_dpath, 'dump'))
        dump_fname = 'pred_{:08d}.png'.format(harn.epoch)
        fpath = os.path.join(dump_dpath, dump_fname)
        nh.util.imwrite(fpath, img)

    def deploy(harn):
        """
        Experimental function that will deploy a standalone predictor
        """
        pass


def setup_harness(bsize=16, workers=0):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py setup_harness

    Example:
        >>> # DISABLE_DOCTSET
        >>> harn = setup_harness()
        >>> harn.initialize()
    """

    xpu = nh.XPU.cast('argv')

    nice = ub.argval('--nice', default='Yolo2Baseline')
    batch_size = int(ub.argval('--batch_size', default=bsize))
    bstep = int(ub.argval('--bstep', 4))
    workers = int(ub.argval('--workers', default=workers))
    decay = float(ub.argval('--decay', default=0.0005))
    lr = float(ub.argval('--lr', default=0.001))
    ovthresh = 0.5

    # We will divide the learning rate by the simulated batch size
    datasets = {
        'train': YoloVOCDataset(years=[2007, 2012], split='trainval'),
        'test': YoloVOCDataset(years=[2007], split='test'),
    }
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=True)
        for key, dset in datasets.items()
    }

    if workers > 0:
        import cv2
        cv2.setNumThreads(0)

    simulated_bsize = bstep * batch_size
    hyper = nh.HyperParams(**{
        'nice': nice,
        'workdir': ub.truepath('~/work/voc_yolo2'),
        'datasets': datasets,

        # 'xpu': 'distributed(todo: fancy network stuff)',
        # 'xpu': 'cpu',
        # 'xpu': 'gpu:0,1,2,3',
        'xpu': xpu,

        # a single dict is applied to all datset loaders
        'loaders': loaders,

        'model': (light_yolo.Yolo, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors,
            'conf_thresh': 0.001,
            # 'nms_thresh': 0.5,  # reproduce original yolo
            'nms_thresh': 0.4,  # reproduce lightnet
        }),

        'criterion': (light_region_loss.RegionLoss, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,
            'class_scale': 1.0,
            'coord_scale': 1.0,
            'thresh': 0.6,  # iou_thresh
        }),

        'initializer': (nh.initializers.Pretrained, {
            # 'fpath': light_yolo.demo_voc_weights(),
            'fpath': light_yolo.initial_imagenet_weights(),
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': lr / 10,
            'momentum': 0.9,
            'dampening': 0,
            # multiplying by batch size was one of those unpublished details
            'weight_decay': decay * simulated_bsize,
        }),

        # Pascal 2007 + 2012 trainval has 16551 images
        # Pascal 2007 test has 4952 images
        # In the original YOLO, one batch is 64 images,
        # so one epoch is 16551 / 64 = 259 iterations.
        #
        # From the original YOLO VOC v2 config
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
        #     learning_rate=0.001
        #     burn_in=1000
        #     max_batches = 80200
        #     policy=steps
        #     steps=40000,60000
        #     scales=.1,.1
        #
        # However, the LIGHTNET values are
        #   LR_STEPS = [250, 25000, 35000]
        #
        # Based in this, the iter to batch conversion is
        #
        # ((np.array([250, 25000, 35000, 1000, 40000, 60000, 80200]) / 256) + 1).astype(np.int)
        # array([  1,  98, 137,   4, 157, 235, 314])


        'scheduler': (nh.schedulers.ListedLR, {
            'points': {
                # dividing by batch size was one of those unpublished details
                # 0:  lr * 0.1 / simulated_bsize,  # burnin
                # 4:  lr * 1.0 / simulated_bsize,
                # 157: lr * 0.1 / simulated_bsize,
                # 235: lr * 0.001 / simulated_bsize,
                0:  lr * 0.1 / simulated_bsize,
                1:  lr * 1.0 / simulated_bsize,
                60: lr * 0.1 / simulated_bsize,
                90: lr * 0.001 / simulated_bsize,
            },
            'interpolate': False
        }),

        'monitor': (nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['mAP'],
            'patience': 314,
            'max_epoch': 314,
        }),

        'augment': datasets['train'].augmenter,

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
    harn = YoloHarn(hyper=hyper)
    harn.config['use_tqdm'] = False
    harn.intervals['log_iter_train'] = 1
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None
    return harn


def train():
    harn = setup_harness()
    # util.ensure_ulimit()
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        srun -c 4 -p priority --gres=gpu:1 \
            python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=rescaled --lr=0.001 --bstep=4 --workers=4
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
