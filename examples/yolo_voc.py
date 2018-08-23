# -*- coding: utf-8 -*-
"""
References:
    https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import itertools as it
import torch
import ubelt as ub
import numpy as np
import netharn as nh
from netharn import util
import imgaug.augmenters as iaa
import torch.utils.data as torch_data
from netharn.models.yolo2 import multiscale_batch_sampler
from netharn.models.yolo2 import light_yolo


class YoloVOCDataset(nh.data.voc.VOCDataset):
    """
    Extends VOC localization dataset (which simply loads the images in VOC2008
    with minimal processing) for multiscale training.

    CommandLine:
        python ~/code/netharn/examples/yolo_voc.py YoloVOCDataset

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
            import netharn.data.transforms  # NOQA
            from netharn.data.transforms import HSVShift
            augmentors = [
                # Order used in lightnet is hsv, rc, rf, lb
                # lb is applied externally to augmenters
                iaa.Sometimes(.9, HSVShift(hue=0.1, sat=1.5, val=1.5)),
                # iaa.Crop(percent=(0, .2)),
                iaa.Crop(percent=(0, .2), keep_size=False),
                iaa.Fliplr(p=.5),
            ]
            self.augmenter = iaa.Sequential(augmentors)

        # Used to resize images to the appropriate inp_size without changing
        # the aspect ratio.
        self.letterbox = nh.data.transforms.Resize(None, mode='letterbox')

    @util.profiler.profile
    def __getitem__(self, index):
        """
        CommandLine:
            python ~/code/netharn/examples/yolo_voc.py YoloVOCDataset.__getitem__ --show

        Example:
            >>> # DISABLE_DOCTSET
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.truepath('~/code/netharn/examples'))
            >>> from yolo_voc import *
            >>> self = YoloVOCDataset(split='train')
            >>> index = 7
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> norm_boxes = label['targets'].numpy().reshape(-1, 5)[:, 1:5]
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
            >>> sys.path.append(ubelt.truepath('~/code/netharn/examples'))
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
                    pin_memory=False, resize_rate=10, drop_last=False):
        """
        CommandLine:
            python ~/code/netharn/examples/yolo_voc.py YoloVOCDataset.make_loader

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
        super(YoloHarn, self).__init__(**kw)
        # harn.batch_confusions = []
        # harn.aps = {}

        # Dictionary of detection metrics
        harn.dmets = {}  # Dict[str, nh.metrics.detections.DetectionMetrics]
        harn.chosen_indices = {}

    def after_initialize(harn):
        # Prepare structures we will use to measure and quantify quality
        for tag, voc_dset in harn.datasets.items():
            cacher = ub.Cacher('dmet2', cfgstr=tag, appname='netharn')
            dmet = cacher.tryload()
            if dmet is None:
                dmet = nh.metrics.detections.DetectionMetrics()
                dmet.true = voc_dset.to_coco()
                # Truth and predictions share the same images and categories
                dmet.pred.dataset['images'] = dmet.true.dataset['images']
                dmet.pred.dataset['categories'] = dmet.true.dataset['categories']
                dmet.pred.dataset['annotations'] = []  # start empty
                dmet.true.dataset['annotations'] = []
                dmet.pred._clear_index()
                dmet.true._clear_index()
                dmet.true._build_index()
                dmet.pred._build_index()
                dmet.true._ensure_imgsize()
                dmet.pred._ensure_imgsize()
                cacher.save(dmet)

            dmet.true._build_index()
            dmet.pred._build_index()
            harn.dmets[tag] = dmet

    @util.profiler.profile
    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        batch_inputs, batch_labels = raw_batch

        inputs = harn.xpu.variable(batch_inputs)
        labels = {k: harn.xpu.variable(d) for k, d in batch_labels.items()}

        batch = (inputs, labels)
        return batch

    @util.profiler.profile
    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader

        CommandLine:
            python ~/code/netharn/examples/yolo_voc.py YoloHarn.run_batch

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
        # n_seen = 10000

        inputs, labels = batch
        outputs = harn.model(inputs)
        # if ub.argflag('--profile'):
        #     torch.cuda.synchronize()
        target = labels['targets']
        gt_weights = labels['gt_weights']

        target2 = {
            'target': target,
            'gt_weights': gt_weights,
        }
        loss = harn.criterion(outputs, target2, seen=n_seen)

        # loss = harn.criterion(outputs, target, seen=n_seen,
        #                       gt_weights=gt_weights)

        # if ub.argflag('--profile'):
        #     torch.cuda.synchronize()
        return outputs, loss

    @util.profiler.profile
    def on_batch(harn, batch, outputs, loss):
        """
        custom callback

        CommandLine:
            python ~/code/netharn/examples/yolo_voc.py YoloHarn.on_batch --gpu=0 --show

        Example:
            >>> # DISABLE_DOCTSET
            >>> harn = setup_yolo_harness(bsize=1)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn.on_batch(batch, outputs, loss)
            >>> # xdoc: +REQUIRES(--show)
            >>> postout = harn.model.module.postprocess(outputs, nms_mode=4)
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> harn.visualize_prediction(batch, outputs, postout, idx=0, thresh=0.01)
            >>> mplutil.show_if_requested()
        """
        harn._record_predictions(batch, outputs)

        metrics_dict = ub.odict()
        metrics_dict['L_bbox'] = float(harn.criterion.loss_coord)
        metrics_dict['L_iou'] = float(harn.criterion.loss_conf)
        metrics_dict['L_cls'] = float(harn.criterion.loss_cls)
        for k, v in metrics_dict.items():
            if not np.isfinite(v):
                raise ValueError('{}={} is not finite'.format(k, v))
        return metrics_dict

    def _postout_to_pred_ann(harn, inp_size, labels, postout, _aidbase=1,
                             undo_lb=True):
        """ Convert batch predictions to coco-style annotations for scoring """
        indices = labels['indices']
        orig_sizes = labels['orig_sizes']
        letterbox = harn.datasets[harn.current_tag].letterbox

        MAX_DETS = None

        bsize = len(indices)

        _aids = it.count(_aidbase)
        for bx in range(bsize):
            postitem = postout[bx].data.cpu().numpy()
            orig_size = orig_sizes[bx].data.cpu().numpy()
            gx = int(indices[bx].data.cpu().numpy())

            # Unpack postprocessed predictions
            sboxes = postitem.reshape(-1, 6)
            pred_boxes_ = util.Boxes(sboxes[:, 0:4], 'cxywh').scale(inp_size)
            pred_scores = sboxes[:, 4]
            pred_cxs = sboxes[:, 5].astype(np.int)

            if undo_lb:
                pred_boxes = letterbox._boxes_letterbox_invert(
                    pred_boxes_, orig_size, inp_size)
            else:
                pred_boxes = pred_boxes_

            # sort predictions by descending score

            # Take at most MAX_DETS detections to evaulate
            _pred_sortx = pred_scores.argsort()[::-1][:MAX_DETS]

            _pred_boxes = pred_boxes.take(_pred_sortx, axis=0).to_xywh().data.tolist()
            _pred_cxs = pred_cxs.take(_pred_sortx, axis=0).tolist()
            _pred_scores = pred_scores.take(_pred_sortx, axis=0).tolist()
            for box, cx, score, aid in zip(_pred_boxes, _pred_cxs, _pred_scores, _aids):
                yield {
                    'id': aid,
                    'image_id': gx,
                    'category_id': cx,
                    'bbox': box,
                    'score': score
                }

    def _labels_to_true_ann(harn, inp_size, labels, _aidbase=1, undo_lb=True):
        """ Convert batch groundtruth to coco-style annotations for scoring """
        indices = labels['indices']
        orig_sizes = labels['orig_sizes']
        targets = labels['targets']
        gt_weights = labels['gt_weights']

        letterbox = harn.datasets[harn.current_tag].letterbox
        # On the training set, we need to add truth due to augmentation
        bsize = len(indices)
        _aids = it.count(_aidbase)
        for bx in range(bsize):
            target = targets[bx].cpu().numpy().reshape(-1, 5)
            true_weights = gt_weights[bx].cpu().numpy()
            orig_size = orig_sizes[bx].cpu().numpy()
            true_cxs = target[:, 0].astype(np.int)
            true_cxywh = target[:, 1:5]
            flags = true_cxs != -1
            true_weights = true_weights[flags]
            true_cxywh = true_cxywh[flags]
            true_cxs = true_cxs[flags]

            gx = int(indices[bx].data.cpu().numpy())

            true_boxes = nh.util.Boxes(true_cxywh, 'cxywh').scale(inp_size)
            if undo_lb:
                true_boxes = letterbox._boxes_letterbox_invert(true_boxes, orig_size, inp_size)

            _true_boxes = true_boxes.to_xywh().data.tolist()
            _true_cxs = true_cxs.tolist()
            _true_weights = true_weights.tolist()
            for box, cx, weight, aid in zip(_true_boxes, _true_cxs, _true_weights, _aids):
                yield {
                    'id': aid,
                    'image_id': gx,
                    'category_id': cx,
                    'bbox': box,
                    'weight': weight
                }

    @util.profiler.profile
    def _record_predictions(harn, batch, outputs):
        """
        Transform batch predictions into coco-style detections for scoring

        Ignore:
            >>> harn = setup_yolo_harness(bsize=1)
            >>> harn.initialize()
            >>> harn.current_tag = tag = 'test'
            >>> batch = harn._demo_batch(0, tag)
            >>> weights_fpath = light_yolo.demo_voc_weights()
            >>> state_dict = harn.xpu.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn.run_batch(batch)
            >>> harn._record_predictions(batch, outputs)
        """
        dmet = harn.dmets[harn.current_tag]
        inputs, labels = batch
        inp_size = np.array(inputs.shape[-2:][::-1])

        try:
            postout = harn.model.module.postprocess(outputs, nms_mode=4)
        except Exception as ex:
            harn.error('\n\n\n')
            harn.error('ERROR: FAILED TO POSTPROCESS OUTPUTS')
            harn.error('DETAILS: {!r}'.format(ex))
            raise

        pred_anns = list(harn._postout_to_pred_ann(
            inp_size, labels, postout, _aidbase=dmet.pred.n_annots + 1
        ))
        dmet.pred.add_annotations(pred_anns)

        true_anns = list(harn._labels_to_true_ann(
            inp_size, labels, _aidbase=dmet.true.n_annots + 1
        ))
        dmet.true.add_annotations(true_anns)

    @util.profiler.profile
    def on_epoch(harn):
        """
        custom callback

        CommandLine:
            python ~/code/netharn/examples/yolo_voc.py YoloHarn.on_epoch

        Example:
            >>> # DISABLE_DOCTSET
            >>> import sys, os
            >>> sys.path.append(os.path.expanduser('~/code/netharn/examples'))
            >>> from yolo_voc import *
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

        # harn.log('Epoch evaluation: {}'.format(harn.current_tag))

        # Measure quality
        dmet = harn.dmets[harn.current_tag]
        try:
            coco_scores = dmet.score_coco()
            metrics_dict['coco-mAP'] = coco_scores['mAP']
        except ImportError:
            pass
        except Exception as ex:
            print('ex = {!r}'.format(ex))

        try:
            nh_scores = dmet.score_netharn()
            metrics_dict['nh-mAP'] = nh_scores['mAP']
            metrics_dict['nh-AP'] = nh_scores['peritem']['ap']
        except Exception as ex:
            print('ex = {!r}'.format(ex))

        try:
            voc_scores = dmet.score_voc()
            metrics_dict['voc-mAP'] = voc_scores['mAP']
        except Exception as ex:
            print('ex = {!r}'.format(ex))

        # Reset detections
        dmet.pred.remove_all_annotations()
        dmet.true.remove_all_annotations()

        if harn.current_tag in {'test', 'vali'}:
            if harn.epoch > 20:
                # Dont bother testing the early iterations
                if (harn.epoch % 10 == 5 or harn.epoch > 300):
                    harn._dump_chosen_indices()
        return metrics_dict

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

        label_names = harn.datasets[harn.current_tag].label_names

        true_clsnms = list(ub.take(label_names, true_cxs))
        pred_clsnms = list(ub.take(label_names, pred_cxs))
        pred_labels = ['{}@{:.2f}'.format(n, s)
                       for n, s in zip(pred_clsnms, pred_scores)]
        # ---
        inp_size = np.array(hwc01.shape[0:2][::-1])
        target_size = inp_size

        true_boxes_ = nh.util.Boxes(true_cxywh, 'cxywh').scale(inp_size)
        pred_boxes_ = nh.util.Boxes(pred_cxywh, 'cxywh').scale(inp_size)

        letterbox = harn.datasets[harn.current_tag].letterbox
        img = letterbox._img_letterbox_invert(hwc01, orig_size, target_size)
        img = np.clip(img, 0, 1)
        if orig_img is not None:
            # we are given the original image, to avoid artifacts from
            # inverting a downscale
            assert orig_img.shape == img.shape

        true_cxywh_ = letterbox._boxes_letterbox_invert(true_boxes_, orig_size, target_size)
        pred_cxywh_ = letterbox._boxes_letterbox_invert(pred_boxes_, orig_size, target_size)

        # shift, scale, embed_size = letterbox._letterbox_transform(orig_size, target_size)

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
            postout = harn.model.module.postprocess(outputs, nms_mode=4)

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


def setup_yolo_harness(bsize=16, workers=0):
    """
    CommandLine:
        python ~/code/netharn/examples/yolo_voc.py setup_yolo_harness

    Example:
        >>> # DISABLE_DOCTSET
        >>> harn = setup_yolo_harness()
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
    simulated_bsize = bstep * batch_size

    # We will divide the learning rate by the simulated batch size
    datasets = {
        'train': YoloVOCDataset(years=[2007, 2012], split='trainval'),
        'test': YoloVOCDataset(years=[2007], split='test'),
    }
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=True,
                              resize_rate=10 * bstep, drop_last=True)
        for key, dset in datasets.items()
    }

    if workers > 0:
        import cv2
        cv2.setNumThreads(0)

    # assert simulated_bsize == 64, 'must be 64'

    # Pascal 2007 + 2012 trainval has 16551 images
    # Pascal 2007 test has 4952 images
    # In the original YOLO, one batch is 64 images, therefore:
    #
    # ONE EPOCH is 16551 / 64 = 258.609375 = 259 iterations.
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
    # The DARNKET STEPS ARE:
    #   DN_STEPS = 1000, 40000, 60000, 80200
    #
    # Based in this, the iter to batch conversion is
    #
    # Key lightnet batch numbers
    # >>> np.array([250, 25000, 30000, 35000, 45000]) / (16512 / 64)
    # array([0.9689,  96.899, 116.2790, 135.658, 174.4186])
    # -> Round
    # array([  1.,  97., 135.])
    # >>> np.array([1000, 40000, 60000, 80200]) / 258
    # array([  3.86683584, 154.67343363, 232.01015044, 310.12023443])
    # -> Round
    # array(4, 157, 232, 310])
    # array([  3.87596899, 155.03875969, 232.55813953, 310.85271318])
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

    weights = ub.argval('--weights', default=None)
    if weights is None or weights == 'imagenet':
        weights = light_yolo.initial_imagenet_weights()
    elif weights == 'lightnet':
        weights = light_yolo.demo_voc_weights()
    else:
        print('weights = {!r}'.format(weights))

    # Anchors
    anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                        (5.05587, 8.09892), (9.47112, 4.84053),
                        (11.2364, 10.0071)])

    from netharn.models.yolo2 import region_loss2
    # from netharn.models.yolo2 import light_region_loss

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
            'anchors': anchors,
            'conf_thresh': 0.001,
            # 'conf_thresh': 0.1,  # make training a bit faster
            'nms_thresh': 0.5 if not ub.argflag('--eav') else 0.4
        }),

        'criterion': (region_loss2.RegionLoss, {
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
        }),

        # 'criterion': (light_region_loss.RegionLoss, {
        #     'num_classes': datasets['train'].num_classes,
        #     'anchors': anchors,
        #     'object_scale': 5.0,
        #     'noobject_scale': 1.0,

        #     # eav version originally had a random *2 in cls loss,
        #     # we removed, that but we can replicate it here.
        #     'class_scale': 1.0 if not ub.argflag('--eav') else 2.0,
        #     'coord_scale': 1.0,

        #     'thresh': 0.6,  # iou_thresh
        #     'seen_thresh': 12800,
        #     # 'small_boxes': not ub.argflag('--eav'),
        #     'small_boxes': True,
        #     'mse_factor': 0.5 if not ub.argflag('--eav') else 1.0,
        # }),

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
    print('max_epoch = {!r}'.format(max_epoch))
    harn = YoloHarn(hyper=hyper)
    harn.config['prog_backend'] = 'progiter'
    harn.intervals['log_iter_train'] = None
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None
    harn.config['large_loss'] = 1000  # tell netharn when to check for divergence
    return harn


def train():
    harn = setup_yolo_harness()
    # util.ensure_ulimit()
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        srun -c 4 -p priority --gres=gpu:1 \
            python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=rescaled --lr=0.001 --bstep=4 --workers=4

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=new_loss_v2 --lr=0.001 --bstep=4 --workers=4

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=16 --nice=eav_run --lr=0.001 --bstep=4 --workers=6 --eav
        python ~/code/netharn/examples/yolo_voc.py train --gpu=1 --batch_size=16 --nice=pjr_run2 --lr=0.001 --bstep=4 --workers=6

        python ~/code/netharn/examples/yolo_voc.py train --gpu=1 --batch_size=16 --nice=fixed_nms --lr=0.001 --bstep=4 --workers=6

        python ~/code/netharn/examples/yolo_voc.py train --gpu=1 --batch_size=16 --nice=fixed_lrs --lr=0.001 --bstep=4 --workers=6

        # python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=test --lr=0.001 --bstep=4 --workers=0 --profile
        # python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=test --lr=0.001 --bstep=4 --workers=0 --profile

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=eav_run2 --lr=0.001 --bstep=4 --workers=8 --eav
        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=pjr_run2 --lr=0.001 --bstep=4 --workers=4

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=4 --nice=pjr_run2 --lr=0.001 --bstep=8 --workers=4

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0,1 --batch_size=32 --nice=july23 --lr=0.001 --bstep=2 --workers=8
        python ~/code/netharn/examples/yolo_voc.py train --gpu=2 --batch_size=16 --nice=july23_lr_x8 --lr=0.008 --bstep=4 --workers=6

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=batchaware2 --lr=0.001 --bstep=8 --workers=3

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=july_eav_run3 --lr=0.001 --bstep=8 --workers=6 --eav
        python ~/code/netharn/examples/yolo_voc.py train --gpu=1 --batch_size=8 --nice=july_eav_run4 --lr=0.002 --bstep=8 --workers=6 --eav
        python ~/code/netharn/examples/yolo_voc.py train --gpu=2 --batch_size=16 --nice=july_pjr_run4 --lr=0.001 --bstep=4 --workers=6


        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=july_eav_run4_hack1 --lr=0.001 --bstep=8 --workers=6 --eav --weights=/home/local/KHQ/jon.crall/work/voc_yolo2/fit/nice/july_eav_run_hack/torch_snapshots/_epoch_00000150.pt

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=lightnet_start --lr=0.001 --bstep=8 --workers=6 --eav --weights=lightnet


        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=HOPE --lr=0.001 --bstep=8 --workers=6 --eav --weights=imagenet
        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=HOPE2 --lr=0.001 --bstep=8 --workers=6 --eav --weights=imagenet
        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=HOPE3 --lr=0.001 --bstep=8 --workers=4 --eav --weights=imagenet

        python ~/code/netharn/examples/yolo_voc.py train --gpu=0 --batch_size=8 --nice=HOPE4 --lr=0.001 --bstep=8 --workers=4 --eav --weights=imagenet
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
