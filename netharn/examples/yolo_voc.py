# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import ubelt as ub
import numpy as np
import pandas as pd
import netharn as nh
from netharn import util
import imgaug.augmenters as iaa
from netharn.util import profiler  # NOQA
from netharn.data import collate
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
        >>> assert len(YoloVOCDataset(split='train', years=[2007])) == 2501
        >>> assert len(YoloVOCDataset(split='test', years=[2007])) == 4952
        >>> assert len(YoloVOCDataset(split='val', years=[2007])) == 2510
        >>> assert len(YoloVOCDataset(split='trainval', years=[2007])) == 5011

        >>> assert len(YoloVOCDataset(split='train', years=[2007, 2012])) == 8218
        >>> assert len(YoloVOCDataset(split='test', years=[2007, 2012])) == 4952
        >>> assert len(YoloVOCDataset(split='val', years=[2007, 2012])) == 8333

    Example:
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

    @profiler.profile
    def __getitem__(self, index):
        """
        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloVOCDataset.__getitem__ --show

        Example:
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
            >>> target, gt_weights, origsize, index = labels
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
        label = (target, gt_weights, orig_size, index, bg_weight)

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

    @ub.memoize_method  # remove this if RAM is a problem
    def _load_image(self, index):
        return super(YoloVOCDataset, self)._load_image(index)

    @ub.memoize_method
    def _load_annotation(self, index):
        return super(YoloVOCDataset, self)._load_annotation(index)

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=False):
        """
        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloVOCDataset.make_loader

        Example:
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
        assert len(self) > 0, 'must have some data'
        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            self, batch_size=batch_size, shuffle=shuffle
        )
        loader = torch_data.DataLoader(self, batch_sampler=batch_sampler,
                                       collate_fn=collate.padded_collate,
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
        labels = [harn.xpu.variable(d) for d in batch_labels]

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
            >>> harn = setup_harness(bsize=2)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_weights()
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
        target, gt_weights, orig_sizes, indices, bg_weights = labels
        loss = harn.criterion(outputs, target, seen=n_seen)
        # torch.cuda.synchronize()
        return outputs, loss

    @profiler.profile
    def on_batch(harn, batch, outputs, loss):
        """
        custom callback

        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloHarn.on_batch --gpu=0 --show

        Example:
            >>> harn = setup_harness(bsize=1)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_weights()
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

            postout = harn.model.module.postprocess(outputs)

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

    def _postout_to_coco(harn, postout, labels, inp_size):
        """
        -[ ] TODO: dump predictions for the test set to disk and score using
             someone elses code.
        """
        targets = labels[0]
        gt_weights = labels[1]
        orig_sizes = labels[2]
        indices = labels[3]
        # bg_weights = labels[4]

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        def undo_letterbox(cxywh):
            boxes = util.Boxes(cxywh, 'cxywh')
            letterbox = harn.datasets['train'].letterbox
            return letterbox._boxes_letterbox_invert(boxes, orig_size, inp_size)

        predictions = []
        truth = []

        bsize = len(labels[0])
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

    @profiler.profile
    def on_epoch(harn):
        """
        custom callback

        Example:
            >>> harn = setup_harness(bsize=4)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_weights()
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
        if harn.batch_confusions:
            y = pd.concat([pd.DataFrame(y) for y in harn.batch_confusions])
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
            return metrics_dict

    # Non-standard problem-specific custom methods

    @profiler.profile
    def _measure_confusion(harn, postout, labels, inp_size, **kw):
        targets = labels[0]
        gt_weights = labels[1]
        # orig_sizes = labels[2]
        # indices = labels[3]
        bg_weights = labels[4]

        # def clip_boxes_to_letterbox(boxes, letterbox_tlbr):
        #     if boxes.shape[0] == 0:
        #         return boxes

        #     boxes = boxes.copy()
        #     left, top, right, bot = letterbox_tlbr
        #     x1, y1, x2, y2 = boxes.T
        #     np.minimum(x1, right, out=x1)
        #     np.minimum(y1, bot, out=y1)
        #     np.minimum(x2, right, out=x2)
        #     np.minimum(y2, bot, out=y2)

        #     np.maximum(x1, left, out=x1)
        #     np.maximum(y1, top, out=y1)
        #     np.maximum(x2, left, out=x2)
        #     np.maximum(y2, top, out=y2)
        #     return boxes

        def asnumpy(tensor):
            return tensor.data.cpu().numpy()

        bsize = len(labels[0])
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

    def visualize_prediction(harn, batch, outputs, postout, idx=0, thresh=None):
        """
        Returns:
            np.ndarray: numpy image
        """
        # xdoc: +REQUIRES(--show)
        inputs, labels = batch
        targets, gt_weights, orig_sizes, indices, bg_weights = labels
        chw01 = inputs[idx]
        target = targets[idx]
        postitem = postout[idx]
        orig_size = orig_sizes[idx].cpu().numpy()
        # ---
        hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
        # TRUE
        true_cxs = target[:, 0].long()
        true_cxywh = target[:, 1:5]
        flags = true_cxs != -1
        true_cxywh = true_cxywh[flags]
        true_cxs = true_cxs[flags]
        # PRED
        pred_cxywh = postitem[:, 0:4]
        pred_scores = postitem[:, 4]
        pred_cxs = postitem[:, 5]

        if thresh is not None:
            flags = pred_scores > thresh
            pred_cxs = pred_cxs[flags]
            pred_cxywh = pred_cxywh[flags]
            pred_scores = pred_scores[flags]

        pred_clsnms = list(ub.take(harn.datasets['train'].label_names,
                                   pred_cxs.long().cpu().numpy()))
        pred_labels = ['{}@{:.2f}'.format(n, s)
                       for n, s in zip(pred_clsnms, pred_scores)]

        true_labels = list(ub.take(harn.datasets['train'].label_names,
                                   true_cxs.long().cpu().numpy()))
        # ---
        inp_size = np.array(hwc01.shape[0:2][::-1])
        target_size = inp_size

        true_boxes_ = util.Boxes(true_cxywh.cpu().numpy(), 'cxywh').scale(inp_size)
        pred_boxes_ = util.Boxes(pred_cxywh.cpu().numpy(), 'cxywh').scale(inp_size)

        letterbox = harn.datasets['train'].letterbox
        img = letterbox._img_letterbox_invert(hwc01, orig_size, target_size)
        img = np.clip(img, 0, 1)
        true_cxywh_ = letterbox._boxes_letterbox_invert(true_boxes_, orig_size, target_size)
        pred_cxywh_ = letterbox._boxes_letterbox_invert(pred_boxes_, orig_size, target_size)

        from netharn.util import mplutil
        shift, scale, embed_size = letterbox._letterbox_transform(orig_size, target_size)

        mplutil.figure(doclf=True, fnum=1)
        mplutil.imshow(img, colorspace='rgb')
        mplutil.draw_boxes(true_cxywh_.data, color='green', box_format='cxywh', labels=true_labels)
        mplutil.draw_boxes(pred_cxywh_.data, color='blue', box_format='cxywh', labels=pred_labels)

        # mplutil.show_if_requested()


def compare_loss():
    harn = setup_harness(bsize=2)
    harn.hyper.xpu = nh.XPU(0)
    harn.initialize()

    weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')

    # Load weights into a lightnet model
    import lightnet as ln
    import os
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))
    ln_net = ln.models.Yolo(ln_test.CLASSES, weights_fpath,
                            ln_test.CONF_THRESH, ln_test.NMS_THRESH)
    ln_net = harn.xpu.move(ln_net)
    os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
    TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
    ln_dset = ln_test.CustomDataset(TESTFILE, ln_net)

    # Load weights into a netharn model
    state_dict = harn.xpu.load(weights_fpath)['weights']
    harn.model.module.load_state_dict(state_dict)

    ln_img, ln_label = ln_dset[0]
    nh_img, nh_label = harn.datasets['test'][0]
    nh_targets = nh_label[0][None, :]
    ln_targets = [ln_label]

    # Test model forward is the same for my image
    ln_outputs = ln_net._forward(harn.xpu.move(nh_img[None, :]))
    nh_outputs = harn.model(harn.xpu.move(nh_img[None, :]))

    seen = ln_net.loss.seen = 99999999
    ln_loss = ln_net.loss(ln_outputs, nh_targets)
    nh_loss = harn.criterion(ln_outputs, nh_targets, seen=seen)
    print('nh_loss = {!r}'.format(nh_loss))
    print('ln_loss = {!r}'.format(ln_loss))

    ln_brambox_loss = ln_net.loss(ln_outputs, ln_targets)
    print('ln_brambox_loss = {!r}'.format(ln_brambox_loss))

    inp_size = tuple(nh_img.shape[-2:][::-1])

    ln_tf_target = []
    for anno in ln_targets[0]:
        anno.class_label = anno.class_id
        tf = ln.data.preprocess.BramboxToTensor._tf_anno(anno, inp_size, None)
        ln_tf_target.append(tf)

    ln_boxes = nh.util.Boxes(np.array(ln_tf_target)[:, 1:], 'cxywh').scale(inp_size)
    nh_boxes = nh.util.Boxes(np.array(nh_targets[0])[:, 1:], 'cxywh').scale(inp_size)

    nh.util.imshow(ln_img.numpy(), colorspace='rgb', fnum=1)
    nh.util.draw_boxes(ln_boxes, color='blue')
    nh.util.draw_boxes(nh_boxes, color='red')


def _test_with_lnstyle_data():
    """
    Uses pretrained lightnet weights, and the lightnet data loader.

    Uses my critrion and net implementations.
    (already verified to produce the same outputs)

    Checks to see if my loss and map calcluations are the same as lightnet
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py _test_with_lnstyle_data

    Using LighNet Trained Weights:

        LightNet Results:
            TEST 30000 mAP:74.18% Loss:3.16839 (Coord:0.38 Conf:1.61 Cls:1.17)

        My Results:
            # Worse losses (due to different image loading)
            loss: 5.00 {coord: 0.69, conf: 2.05, cls: 2.26}
            mAP = 0.6227
            The MAP is quite a bit worse... Why is that?

    USING THE SAME WEIGHTS! I must be doing something wrong.

        Results using the same Data Loader:
            {cls: 2.22, conf: 2.05, coord: 0.65, loss: 4.91}

            # Checking with extra info
            {loss_bram: 3.17, loss_ten1: 4.91, loss_ten2: 4.91}

        OH!, Is is just that BramBox has an ignore function?
            - [X] Add ignore / weight to tensor version to see if its the same
            YUP! {loss_bram: 3.17, loss_ten1: 4.91, loss_ten2: 4.91, nh_unweighted: 4.92, nh_weighted: 3.16}

    TO CHECK:
        - [ ] why is the loss different?
            - [X] network input size is 416 in both
            - [x] networks output the same data given the same input
            - [x] loss outputs the same data given the same input (they do if seen is the same)
            - [x] CONCLUSION: The loss is not actually different


        - [ ] why is the mAP different?
            - [x] does brambox compute AP differently?
                ... YES
            CONCLUSION: several factors are at play
                * brambox has a different AP computation
                * netharn and lightnet non-max supressions are different
                * The NMS seems to have the most significant impact

    # """
    import brambox.boxes as bbb
    CHECK_SANITY = False

    # Path to weights that we will be testing
    # (These were trained using the lightnet harness and achived a good mAP)
    weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')

    # Load weights into a netharn model
    harn = setup_harness(bsize=2)
    harn.hyper.xpu = nh.XPU(0)
    harn.initialize()
    state_dict = harn.xpu.load(weights_fpath)['weights']
    harn.model.module.load_state_dict(state_dict)
    harn.model.eval()
    nh_net = harn.model.module

    # Load weights into a lightnet model
    import os
    import lightnet as ln
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))
    ln_net = ln.models.Yolo(ln_test.CLASSES, weights_fpath,
                            ln_test.CONF_THRESH, ln_test.NMS_THRESH)
    ln_net = harn.xpu.move(ln_net)
    ln_net.eval()

    # Sanity check, the weights should be the same
    if CHECK_SANITY:
        state1 = nh_net.state_dict()
        state2 = ln_net.state_dict()
        assert state1.keys() == state2.keys()
        for k in state1.keys():
            v1 = state1[k]
            v2 = state2[k]
            assert np.all(v1 == v2)

    # Create a lightnet dataset loader
    TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
    os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
    ln_dset = ln_test.CustomDataset(TESTFILE, ln_net)
    ln_loader = torch.utils.data.DataLoader(
        ln_dset, batch_size=2, shuffle=False, drop_last=False, num_workers=0,
        pin_memory=True, collate_fn=ln.data.list_collate,
    )

    # Create a netharn dataset loader
    nh_loader = harn.loaders['test']

    # ----------------------
    # Postprocessing to transform yolo outputs into detections
    # Basic difference here is the implementation of NMS
    ln_postprocess = ln_net.postprocess
    nh_postprocess = harn.model.module.postprocess

    # ----------------------
    # Define helper functions to deal with bramboxes
    LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
              'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']
    NETWORK_SIZE = (416, 416)
    detection_to_brambox = ln.data.TensorToBrambox(NETWORK_SIZE, LABELS)

    def brambox_to_labels(ln_bramboxes, inp_size):
        """ convert brambox to netharn style labels """
        max_anno = max(map(len, ln_bramboxes))
        ln_targets = [
            ln.data.preprocess.BramboxToTensor.apply(
                annos, inp_size, max_anno=max_anno, class_label_map=LABELS)
            for annos in ln_bramboxes]
        ln_targets = torch.stack(ln_targets)

        gt_weights = -np.ones((len(ln_bramboxes), max_anno), dtype=np.float32)
        for i, annos in enumerate(ln_bramboxes):
            weights = 1.0 - np.array([anno.ignore for anno in annos], dtype=np.float32)
            gt_weights[i, 0:len(annos)] = weights
        gt_weights = torch.Tensor(gt_weights)

        bg_weights = torch.FloatTensor(np.ones(len(ln_targets)))
        indices = None
        orig_sizes = None
        ln_labels = ln_targets, gt_weights, orig_sizes, indices, bg_weights
        return ln_labels

    # ----------------------
    CHECK_LOSS = False

    with torch.no_grad():

        # Track netharn and lightnet results that will be scored
        ln_batch_confusions = []
        nh_batch_confusions = []
        nh_batch_confusions0 = []

        ln_results = []
        nh_results = []
        nh_results0 = []

        anno = {}
        ln_det = {}
        nh_det = {}
        nh_det0 = {}

        moving_ave = nh.util.util_averages.CumMovingAve()

        coco_truth = []
        ln_coco_detections = []
        nh_coco_detections0 = []

        prog = ub.ProgIter(zip(ln_loader, nh_loader), desc='')
        for bx, (ln_batch, nh_batch) in enumerate(prog):
            ln_inputs, ln_bramboxes = ln_batch
            inp_size = tuple(ln_inputs.shape[-2:][::-1])
            nh_inputs, nh_labels = nh_batch

            nh_targets = nh_labels[0]
            nh_gt_weights = nh_labels[1]

            # Convert brambox into components understood by netharn
            ln_labels = brambox_to_labels(ln_bramboxes, inp_size)
            ln_inputs = harn.xpu.variable(ln_inputs)
            ln_targets = harn.xpu.variable(ln_labels[0])
            ln_gt_weights = harn.xpu.variable(ln_labels[1])  # NOQA

            ln_net.loss.seen = 1000000
            ln_outputs = ln_net._forward(ln_inputs)

            if CHECK_SANITY:
                nh_outputs = harn.model(ln_inputs)
                assert np.all(nh_outputs == ln_outputs)

            ln_loss_bram = ln_net.loss(ln_outputs, ln_bramboxes)
            moving_ave.update(ub.odict([
                ('loss_bram', float(ln_loss_bram.sum())),
            ]))

            if CHECK_LOSS:
                seen = ln_net.loss.seen
                ln_loss_ten1 = harn.criterion(ln_outputs, ln_targets,
                                              seen=seen)
                ln_loss_ten2 = ln_net.loss(ln_outputs, ln_targets)

                nh_weighted = harn.criterion(ln_outputs, nh_targets,
                                             gt_weights=nh_gt_weights,
                                             seen=seen)
                nh_unweighted = harn.criterion(ln_outputs, nh_targets,
                                               seen=seen)

                moving_ave.update(ub.odict([
                    ('loss_ten1', float(ln_loss_ten1.sum())),
                    ('loss_ten2', float(ln_loss_ten2.sum())),
                    ('nh_weighted', float(nh_weighted.sum())),
                    ('nh_unweighted', float(nh_unweighted.sum())),
                    # ('coord', harn.criterion.loss_coord),
                    # ('conf', harn.criterion.loss_conf),
                    # ('cls', harn.criterion.loss_cls),
                ]))
            # Display progress information
            average_losses = moving_ave.average()
            description = ub.repr2(average_losses, nl=0, precision=2, si=True)
            prog.set_description(description, refresh=False)

            # nh_outputs and ln_outputs should be the same, so no need to
            # differentiate between them here.
            ln_postout = ln_postprocess(ln_outputs.clone())
            nh_postout = nh_postprocess(ln_outputs.clone())

            # Should use the original NMS strategy
            nh_postout0 = nh_postprocess(ln_outputs.clone(), mode=0)

            ln_brambox_postout = detection_to_brambox([x.clone() for x in ln_postout])
            nh_brambox_postout = detection_to_brambox([x.clone() for x in nh_postout])
            nh_brambox_postout0 = detection_to_brambox([x.clone() for x in nh_postout0])

            # Record data scored by brambox
            offset = len(anno)
            def img_to_box(boxes, offset):
                gname_lut = ln_loader.dataset.keys
                return {gname_lut[offset + k]: v for k, v in enumerate(boxes)}
            anno.update(img_to_box(ln_bramboxes, offset))
            ln_det.update(img_to_box(ln_brambox_postout, offset))
            nh_det.update(img_to_box(nh_brambox_postout, offset))
            nh_det0.update(img_to_box(nh_brambox_postout0, offset))

            bg_weights = torch.FloatTensor(np.ones(len(ln_targets)))
            indices = None
            orig_sizes = None

            # Record data scored by netharn

            ln_labels = ln_targets, ln_gt_weights, orig_sizes, indices, bg_weights

            ln_results.append((ln_postout, ln_labels, inp_size))
            nh_results.append((nh_postout, nh_labels, inp_size))
            nh_results0.append((nh_postout0, nh_labels, inp_size))

            # preds, truths = harn._postout_to_coco(ln_postout, ln_labels, inp_size)
            # ln_coco_detections.append(preds)

            preds, truths = harn._postout_to_coco(nh_postout0, nh_labels, inp_size)
            nh_coco_detections0.append(preds)
            coco_truth.append(truths)

            # kw = dict(bias=0)
            # for y in harn._measure_confusion(ln_postout, ln_labels, inp_size, **kw):
            #     ln_batch_confusions.append(y)

            # for y in harn._measure_confusion(nh_postout, nh_labels, inp_size, **kw):
            #     nh_batch_confusions.append(y)

            # for y in harn._measure_confusion(nh_postout0, nh_labels, inp_size, **kw):
            #     nh_batch_confusions0.append(y)

            if bx > 50:
                break

    # Compute mAP using brambox / lightnet
    ln_mAP = round(bbb.ap(*bbb.pr(ln_det, anno)) * 100, 2)
    nh_mAP = round(bbb.ap(*bbb.pr(nh_det, anno)) * 100, 2)
    nh_mAP0 = round(bbb.ap(*bbb.pr(nh_det0, anno)) * 100, 2)
    print('\n----')
    print('ln_mAP = {!r}'.format(ln_mAP))
    print('nh_mAP = {!r}'.format(nh_mAP))
    print('nh_mAP0 = {!r}'.format(nh_mAP0))

    num_classes = len(LABELS)
    cls_labels = list(range(num_classes))

    # # Compute mAP using netharn
    # if False:
    #     is_tp = (y.true == y.pred) & (y.pred >= 0)
    #     is_fp = (y.true != y.pred) & (y.pred >= 0)
    #     rest = ~(is_fp | is_tp)

    #     y.true[is_tp] = 1
    #     y.pred[is_tp] = 1

    #     y.true[is_fp] = 0
    #     y.pred[is_fp] = 1

    #     y.true[rest] = 0
    #     y.pred[rest] = 0

    #     import sklearn
    #     import sklearn.metrics
    #     sklearn.metrics.average_precision_score(y.true, y.score, 'weighted',
    #                                             y.weight)

    for bias in [0, 1]:
        ln_batch_confusions = []
        nh_batch_confusions = []
        nh_batch_confusions0 = []
        print('\n\n======\n\nbias = {!r}'.format(bias))
        kw = dict(bias=bias, PREFER_WEIGHTED_TRUTH=False)
        for ln_postout, ln_labels, inp_size in ln_results:
            for y in harn._measure_confusion(ln_postout, ln_labels, inp_size, **kw):
                ln_batch_confusions.append(y)

        for nh_postout, nh_labels, inp_size in nh_results:
            for y in harn._measure_confusion(nh_postout, nh_labels, inp_size, **kw):
                nh_batch_confusions.append(y)

        for nh_postout0, nh_labels, inp_size in nh_results0:
            for y in harn._measure_confusion(nh_postout0, nh_labels, inp_size, **kw):
                nh_batch_confusions0.append(y)

        confusions = {
            'lh': ln_batch_confusions,
            # 'nh': nh_batch_confusions,
            'nh0': nh_batch_confusions0,
        }

        for lbl, batch_confusions in confusions.items():
            print('----')
            print('\nlbl = {!r}'.format(lbl))
            y = pd.concat([pd.DataFrame(c) for c in batch_confusions])

            # aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=True)
            # aps = aps.rename(dict(zip(cls_labels, LABELS)), axis=0)
            # mean_ap = np.nanmean(aps['ap'])
            # print('mean_ap_07 = {:.2f}'.format(mean_ap * 100))

            # aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=False)
            # aps = aps.rename(dict(zip(cls_labels, LABELS)), axis=0)
            # mean_ap = np.nanmean(aps['ap'])
            # print('mean_ap_12 = {:.2f}'.format(mean_ap * 100))

            # Try the other way
            from netharn.metrics.detections import _confusion_pr_ap
            prec, recall, ap2 = _confusion_pr_ap(y)
            print('ap2 = {!r}'.format(round(ap2 * 100, 2)))

            # max_ap = np.nanmax(aps['ap'])
            # print(aps)
            # print('max_ap = {!r}'.format(max_ap))


def _run_quick_test():
    harn = setup_harness(bsize=2)
    harn.hyper.xpu = nh.XPU(0)
    harn.initialize()

    if 0:
        # Load up pretrained VOC weights
        weights_fpath = light_yolo.demo_weights()
        state_dict = harn.xpu.load(weights_fpath)['weights']
        harn.model.module.load_state_dict(state_dict)
    else:
        weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')
        state_dict = harn.xpu.load(weights_fpath)['weights']
        harn.model.module.load_state_dict(state_dict)

    harn.model.eval()

    with torch.no_grad():
        postprocess = harn.model.module.postprocess
        # postprocess.conf_thresh = 0.001
        # postprocess.nms_thresh = 0.5
        batch_confusions = []
        moving_ave = nh.util.util_averages.CumMovingAve()
        loader = harn.loaders['test']
        prog = ub.ProgIter(iter(loader), desc='')
        for batch in prog:
            inputs, labels = harn.prepare_batch(batch)
            inp_size = np.array(inputs.shape[-2:][::-1])
            outputs = harn.model(inputs)

            target, gt_weights, orig_sizes, indices, bg_weights = labels
            loss = harn.criterion(outputs, target, gt_weights=gt_weights,
                                  seen=1000000000)
            moving_ave.update(ub.odict([
                ('loss', float(loss.sum())),
                ('coord', harn.criterion.loss_coord),
                ('conf', harn.criterion.loss_conf),
                ('cls', harn.criterion.loss_cls),
            ]))

            average_losses = moving_ave.average()
            desc = ub.repr2(average_losses, nl=0, precision=2, si=True)
            prog.set_description(desc, refresh=False)

            postout = postprocess(outputs)
            for y in harn._measure_confusion(postout, labels, inp_size):
                batch_confusions.append(y)

            # batch_output.append((outputs.cpu().data.numpy().copy(), inp_size))
            # batch_labels.append([x.cpu().data.numpy().copy() for x in labels])

        average_losses = moving_ave.average()
        print('average_losses {}'.format(ub.repr2(average_losses)))

    # batch_confusions = []
    # for (outputs, inp_size), labels in ub.ProgIter(zip(batch_output, batch_labels), total=len(batch_labels)):
    #     labels = [torch.Tensor(x) for x in labels]
    #     outputs = torch.Tensor(outputs)
    #     postout = postprocess(outputs)
    #     for y in harn._measure_confusion(postout, labels, inp_size):
    #         batch_confusions.append(y)

    if False:
        from netharn.util import mplutil
        mplutil.qtensure()  # xdoc: +SKIP
        harn.visualize_prediction(batch, outputs, postout, thresh=.1)

    y = pd.concat([pd.DataFrame(c) for c in batch_confusions])
    # TODO: write out a few visualizations
    num_classes = len(loader.dataset.label_names)
    cls_labels = list(range(num_classes))

    aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=True)
    aps = aps.rename(dict(zip(cls_labels, loader.dataset.label_names)), axis=0)
    mean_ap = np.nanmean(aps['ap'])
    max_ap = np.nanmax(aps['ap'])
    print(aps)
    print('mean_ap = {!r}'.format(mean_ap))
    print('max_ap = {!r}'.format(max_ap))

    aps = nh.metrics.ave_precisions(y[y.score > .01], cls_labels, use_07_metric=True)
    aps = aps.rename(dict(zip(cls_labels, loader.dataset.label_names)), axis=0)
    mean_ap = np.nanmean(aps['ap'])
    max_ap = np.nanmax(aps['ap'])
    print(aps)
    print('mean_ap = {!r}'.format(mean_ap))
    print('max_ap = {!r}'.format(max_ap))

    # import sklearn.metrics
    # sklearn.metrics.accuracy_score(y.true, y.pred)
    # sklearn.metrics.precision_score(y.true, y.pred, average='weighted')


def setup_harness(bsize=16, workers=0):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py setup_harness

    Example:
        >>> harn = setup_harness()
        >>> harn.initialize()
    """

    xpu = nh.XPU.cast('argv')

    nice = ub.argval('--nice', default='Yolo2Baseline')
    batch_size = int(ub.argval('--batch_size', default=bsize))
    bstep = int(ub.argval('--bstep', 1))
    workers = int(ub.argval('--workers', default=workers))
    decay = float(ub.argval('--decay', default=0.0005))
    lr = float(ub.argval('--lr', default=0.001))
    ovthresh = 0.5

    # We will divide the learning rate by the simulated batch size
    datasets = {
        'train': YoloVOCDataset(split='trainval'),
        'test': YoloVOCDataset(split='test'),
    }
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=True)
        for key, dset in datasets.items()
    }

    # simulated_bsize = bstep * batch_size
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
            'nms_thresh': 0.5,
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
            # 'fpath': light_yolo.demo_weights(),
            'fpath': light_yolo.initial_imagenet_weights(),
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': lr / 10,
            'momentum': 0.9,
            'weight_decay': decay,
        }),

        'scheduler': (nh.schedulers.ListedLR, {
            'points': {
                # dividing by batch size was one of those unpublished details
                # 0: lr / simulated_bsize,
                # 5:  .01 / simulated_bsize,
                # 60: .011 / simulated_bsize,
                # 90: .001 / simulated_bsize,
                0:  lr / 10,
                1:  lr,
                59: lr * 1.1,
                60: lr / 10,
                90: lr / 100,
            },
            'interpolate': True
        }),

        'monitor': (nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['mAP'],
            'patience': 160,
            'max_epoch': 160,
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
    util.ensure_ulimit()
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=Small --lr=.00005
        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0,1,2,3 --batch_size=64 --workers=4 --nice=Warmup64 --lr=.0001

        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0,1,2,3 --batch_size=64 --workers=4 --nice=ColdOpen64 --lr=.001


        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=dynamic --lr=.001 --bstep=4

        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=letterboxed_copylr --bstep=4

        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=letterboxed_copylr_reworkaug --bstep=4

        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=1 --batch_size=16 --nice=better_lr1 --lr=0.0001 --decay=0.0005 --bstep=4 --workers=4
        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=2 --batch_size=16 --nice=better_lr2 --lr=0.000015625 --decay=0.0000078125 --bstep=4 --workers=4
        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=3 --batch_size=16 --nice=better_lr3 --lr=0.00002 --decay=0.00001 --bstep=4 --workers=4

        python ~/code/netharn/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=copy_aug --bstep=4 --lr=0.000015625 --decay=0.0000078125

        python ~/code/netharn/netharn/examples/yolo_voc.py all
        python ~/code/netharn/netharn/examples/yolo_voc.py setup_harness
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
