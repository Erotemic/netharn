# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import ubelt as ub
import numpy as np
import pandas as pd
import netharn as nh
from netharn import util
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from netharn.util import profiler  # NOQA
from netharn.data import collate
import torch.utils.data as torch_data
from netharn.models.yolo2 import multiscale_batch_sampler
from netharn.models.yolo2 import light_region_loss
from netharn.models.yolo2 import light_yolo

# def s(d):
#     """ sorts a dict, returns as an OrderedDict """
#     return ub.odict(sorted(d.items()))


def asnumpy(tensor):
    return tensor.data.cpu().numpy()


def asfloat(t):
    return float(asnumpy(t))


def letterbox_transform(orig_size, target_size):
    """
    aspect ratio preserving scaling + extra padding to equal target size
    """
    ow, oh = orig_size
    tw, th = target_size

    factor = orig_size / target_size
    fw, fh = factor
    sf = 1 / fw if fw >= fh else 1 / fh

    embed_size = np.round(orig_size * sf).astype(np.int)
    shift = np.round((target_size - embed_size) / 2).astype(np.int)
    return shift, embed_size


def apply_img_letterbox(image, embed_size, shift, target_size):
    """
    Example:
        >>> #image = np.ones((100, 50), dtype=np.uint8) * 255
        >>> image = np.ones((100, 100), dtype=np.uint8) * 255
        >>> target_size = np.array((100, 120))
        >>> orig_size = np.array(image.shape[0:2][::-1])
        >>> shift, embed_size = letterbox_transform(orig_size, target_size)
        >>> hwc255 = apply_img_letterbox(image, embed_size, shift, target_size)
        >>> # xdoc: +REQUIRES(--show)
        >>> from netharn.util import mplutil
        >>> mplutil.figure(doclf=True, fnum=1)
        >>> mplutil.qtensure()  # xdoc: +SKIP
        >>> mplutil.imshow(hwc255, colorspace='gray')
        >>> mplutil.show_if_requested()
    """

    pad_lefttop = shift
    pad_rightbot = target_size - (embed_size + shift)

    left, top = pad_lefttop
    right, bot = pad_rightbot

    fill_color = 127
    channels = nh.util.get_num_channels(image)

    # NO DONT SQUISH: squish the image into network input coordinates
    # USE LETTERBOX TO MAINTAIN ASPECT RATIO
    interpolation = cv2.INTER_AREA
    scaled = cv2.resize(image, tuple(embed_size),
                        interpolation=interpolation)

    hwc255 = cv2.copyMakeBorder(scaled, top, bot, left, right,
                                cv2.BORDER_CONSTANT,
                                value=(fill_color,) * channels)
    return hwc255


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

        # self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
        #                            (6.63, 11.38), (9.42, 5.11),
        #                            (16.62, 10.52)],
        #                           dtype=np.float)

        self.anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                                 (5.05587, 8.09892), (9.47112, 4.84053),
                                 (11.2364, 10.0071)])
        self.num_anchors = len(self.anchors)
        self.augmenter = None

        if 'train' in split:
            augmentors = [
                iaa.Fliplr(p=.5),
                # iaa.Flipud(p=.5),
                iaa.Affine(
                    # scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
                    # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    # rotate=(-15, 15),
                    # shear=(-7, 7),
                    # order=[0, 1, 3],
                    order=1,
                    # cval=(0, 255),
                    cval=127,
                    mode=ia.ALL,
                    backend='cv2',
                ),
                # iaa.AddToHueAndSaturation((-20, 20)),
                # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.AddToHueAndSaturation((-15, 15)),
                iaa.ContrastNormalization((0.75, 1.5))
                # iaa.ContrastNormalization((0.75, 1.5), per_channel=0.5),
            ]
            self.augmenter = iaa.Sequential(augmentors)

    @profiler.profile
    def __getitem__(self, index):
        """
        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloVOCDataset.__getitem__ --show

        Example:
            >>> self = YoloVOCDataset(split='train')
            >>> index = 1
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> norm_boxes = label[0][:, 1:5].numpy()
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

        (hwc255, orig_size,
         tlbr, gt_classes, gt_weights) = self._load_item(index, inp_size)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            hwc255 = seq_det.augment_image(hwc255)

            bbs = ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1, y1, x2, y2)
                 for x1, y1, x2, y2 in tlbr], shape=hwc255.shape)
            bbs = seq_det.augment_bounding_boxes([bbs])[0]

            tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                             for bb in bbs.bounding_boxes])
            tlbr = util.clip_boxes(tlbr, hwc255.shape[0:2])

            # REMOVE ANY BOXES THAT ARE NOW OUT OF BOUNDS
            tlbr = util.Boxes(tlbr, 'tlbr')
            flags = (tlbr.area > 0).ravel()
            tlbr = tlbr.data[flags]
            gt_classes = gt_classes[flags]
            gt_weights = gt_weights[flags]

        # Remove boxes that are too small
        # ONLY DO THIS FOR THE SMALL DEMO TASK
        # if False:
        #     tlbr = util.Boxes(tlbr, 'tlbr')
        #     flags = (tlbr.area > 10).ravel()
        #     tlbr = tlbr.data[flags]
        #     gt_classes = gt_classes[flags]
        #     gt_weights = gt_weights[flags]

        chw01 = torch.FloatTensor(hwc255.transpose(2, 0, 1) / 255.0)

        # Lightnet YOLO accepts truth tensors in the format:
        # [class_id, center_x, center_y, w, h]
        # where coordinates are noramlized between 0 and 1
        tlbr_inp = util.Boxes(tlbr, 'tlbr')
        cxywh_norm = tlbr_inp.asformat('cxywh').scale(1 / inp_size)

        datas = [gt_classes[:, None], cxywh_norm.data]
        # [d.shape for d in datas]
        target = np.concatenate(datas, axis=-1)
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

    def _load_sized_image(self, index, inp_size):
        """
        Example:
            >>> self = YoloVOCDataset(split='test')
            >>> index = 0
            >>> target_size = inp_size = np.array((416, 416))
            >>> hwc255 = self._load_sized_image(0, (inp_size))[0]
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc255, colorspace='rgb')
        """
        # load the raw data from VOC

        # cacher = ub.Cacher('voc_img', cfgstr=ub.repr2([index, inp_size]),
        #                    appname='netharn', enabled=0)
        # data = cacher.tryload()
        # if data is None:
        image = self._load_image(index)

        orig_size = np.array(image.shape[0:2][::-1])

        shift, embed_size = letterbox_transform(orig_size, inp_size)
        hwc255 = apply_img_letterbox(image, embed_size, shift, inp_size)

        scale = embed_size / orig_size

        # cacher.save(data)
        return hwc255, orig_size, scale, shift

    def _load_item(self, index, inp_size):
        # load the raw data from VOC
        inp_size = np.array(inp_size)
        hwc255, orig_size, scale, shift = self._load_sized_image(index, inp_size)

        # VOC loads annotations in tlbr
        annot = self._load_annotation(index)
        tlbr_orig = util.Boxes(annot['boxes'].astype(np.float), 'tlbr')
        gt_classes = annot['gt_classes']
        # Weight samples so we dont care about difficult cases
        gt_weights = 1.0 - annot['gt_ishard'].astype(np.float)
        # squish the bounding box into network input coordinates
        tlbr = tlbr_orig.scale(scale).shift(shift).data

        return hwc255, orig_size, tlbr, gt_classes, gt_weights

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
        loader.batch_size = batch_size
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
            >>> state_dict = torch.load(weights_fpath)['weights']
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

        target, gt_weights, orig_sizes, indices, bg_weights = labels
        loss = harn.criterion(outputs, target, seen=n_seen)
        return outputs, loss

    @profiler.profile
    def on_batch(harn, batch, outputs, loss):
        """
        custom callback

        CommandLine:
            python ~/code/netharn/netharn/examples/yolo_voc.py YoloHarn.on_batch --gpu=0 --show

        Example:
            >>> harn = setup_harness(bsize=3)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'train')
            >>> weights_fpath = light_yolo.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['weights']
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
        inputs, labels = batch
        postout = harn.model.module.postprocess(outputs)
        inp_size = np.array(inputs.shape[-2:][::-1])

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
            >>> harn = setup_harness(bsize=4)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'test')
            >>> weights_fpath = light_yolo.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['weights']
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
        # if tag == 'train':
        #     harn.datasets['train'].augmenter = harn.datasets['train']._augmenter

        loader = harn.loaders[tag]
        y = pd.concat(harn.batch_confusions)
        # TODO: write out a few visualizations
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
    def _measure_confusion(harn, postout, labels, inp_size):
        targets, gt_weights, orig_sizes, indices, bg_weights = labels

        def clip_boxes_to_letterbox(boxes, letterbox_tlbr):
            if boxes.shape[0] == 0:
                return boxes

            boxes = boxes.copy()
            left, top, right, bot = letterbox_tlbr
            x1, y1, x2, y2 = boxes.T
            np.minimum(x1, right, out=x1)
            np.minimum(y1, bot, out=y1)
            np.minimum(x2, right, out=x2)
            np.minimum(y2, bot, out=y2)

            np.maximum(x1, left, out=x1)
            np.maximum(y1, top, out=y1)
            np.maximum(x2, left, out=x2)
            np.maximum(y2, top, out=y2)
            return boxes

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

            orig_size    = asnumpy(orig_sizes[bx])
            gx           = int(asnumpy(indices[bx]))

            # how much do we care about the background in this image?
            bg_weight = float(asnumpy(bg_weights[bx]))

            # Unpack postprocessed predictions
            sboxes = postitem.reshape(-1, 6)
            pred_cxywh = sboxes[:, 0:4]
            pred_scores = sboxes[:, 4]
            pred_cxs = sboxes[:, 5].astype(np.int)

            true_tlbr = util.Boxes(true_cxywh, 'cxywh').as_tlbr()
            pred_tlbr = util.Boxes(pred_cxywh, 'cxywh').as_tlbr()

            # use max because of letterbox transform
            lettered_orig_size = orig_size.max()
            true_boxes = true_tlbr.scale(lettered_orig_size).data
            pred_boxes = pred_tlbr.scale(lettered_orig_size).data

            # Clip predicted boxes to the letterbox
            shift, embed_size = letterbox_transform(orig_size, inp_size)
            orig_lefttop = (shift / inp_size) * orig_size.max()
            orig_rightbot = lettered_orig_size - orig_lefttop
            letterbox_tlbr = list(orig_lefttop) + list(orig_rightbot)

            pred_boxes = clip_boxes_to_letterbox(pred_boxes, letterbox_tlbr)

            y = nh.metrics.detection_confusions(
                true_boxes=true_boxes,
                true_cxs=true_cxs,
                true_weights=true_weight,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_cxs=pred_cxs,
                bg_weight=bg_weight,
                bg_cls=-1,
                ovthresh=harn.hyper.other['ovthresh']
            )

            y['gx'] = gx
            yield y

    @classmethod
    def _run_quick_test(cls):
        harn = setup_harness(bsize=2)
        harn.hyper.xpu = nh.XPU(0)
        harn.initialize()

        # Load up pretrained VOC weights
        weights_fpath = light_yolo.demo_weights()
        state_dict = torch.load(weights_fpath)['weights']
        harn.model.module.load_state_dict(state_dict)

        harn.model.eval()

        postprocess = harn.model.module.postprocess
        # postprocess.conf_thresh = 0.001
        # postprocess.nms_thresh = 0.5

        batch_confusions = []
        # batch_labels = []
        # batch_output = []
        loader = harn.loaders['test']
        for batch in ub.ProgIter(loader):
            inputs, labels = harn.prepare_batch(batch)
            inp_size = np.array(inputs.shape[-2:][::-1])
            outputs = harn.model(inputs)

            postout = postprocess(outputs)
            for y in harn._measure_confusion(postout, labels, inp_size):
                batch_confusions.append(y)

            # batch_output.append((outputs.cpu().data.numpy().copy(), inp_size))
            # batch_labels.append([x.cpu().data.numpy().copy() for x in labels])

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

        y = pd.concat(batch_confusions)
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
        # ---
        hwc01 = chw01.cpu().numpy().transpose(1, 2, 0)
        # TRUE
        true_cxs = target[:, 0].long()
        true_boxes = target[:, 1:5]
        flags = true_cxs != -1
        true_boxes = true_boxes[flags]
        true_cxs = true_cxs[flags]
        # PRED
        pred_boxes = postitem[:, 0:4]
        pred_scores = postitem[:, 4]
        pred_cxs = postitem[:, 5]

        if thresh is not None:
            flags = pred_scores > thresh
            pred_cxs = pred_cxs[flags]
            pred_boxes = pred_boxes[flags]
            pred_scores = pred_scores[flags]

        pred_clsnms = list(ub.take(harn.datasets['train'].label_names,
                                   pred_cxs.long().cpu().numpy()))
        pred_labels = ['{}@{:.2f}'.format(n, s)
                       for n, s in zip(pred_clsnms, pred_scores)]

        true_labels = list(ub.take(harn.datasets['train'].label_names,
                                   true_cxs.long().cpu().numpy()))

        # ---
        inp_size = np.array(hwc01.shape[0:2][::-1])
        true_boxes_ = util.Boxes(true_boxes.cpu().numpy(), 'cxywh').scale(inp_size).data
        pred_boxes_ = util.Boxes(pred_boxes.cpu().numpy(), 'cxywh').scale(inp_size).data
        from netharn.util import mplutil

        mplutil.figure(doclf=True, fnum=1)
        mplutil.imshow(hwc01, colorspace='rgb')
        mplutil.draw_boxes(true_boxes_, color='green', box_format='cxywh', labels=true_labels)
        mplutil.draw_boxes(pred_boxes_, color='blue', box_format='cxywh', labels=pred_labels)

        # mplutil.show_if_requested()


def setup_harness(bsize=16, workers=0):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py setup_harness

    Example:
        >>> harn = setup_harness()
        >>> harn.initialize()
    """

    xpu = nh.XPU.cast('argv')

    batch_size = int(ub.argval('--batch_size', default=bsize))
    bstep = int(ub.argval('--bstep', 1))
    workers = int(ub.argval('--workers', default=workers))
    # lr = float(ub.argval('--lr', default=0.001))
    nice = ub.argval('--nice', default='Yolo2Baseline')

    # We will divide the learning rate by the simulated batch size
    simulated_bsize = bstep * batch_size

    datasets = {
        'train': YoloVOCDataset(split='trainval'),
        'test': YoloVOCDataset(split='test'),
    }
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=True)
        for key, dset in datasets.items()
    }
    ovthresh = 0.5

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
            'lr': .0001 / simulated_bsize,
            'momentum': 0.9,
            'weight_decay': 0.0005 / simulated_bsize,
        }),

        'scheduler': (nh.schedulers.ListedLR, {
            'points': {
                # dividing by batch size was one of those unpublished details
                # 0: lr / simulated_bsize,
                # 5:  .01 / simulated_bsize,
                # 60: .011 / simulated_bsize,
                # 90: .001 / simulated_bsize,
                0:  .0001 / simulated_bsize,
                1:  .001 / simulated_bsize,
                60: .00101 / simulated_bsize,
                90: .00001 / simulated_bsize,
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
            # currently a special param. TODO: incorporate more generally
            'batch_step': bstep,
        },

        'other': {
            'batch_size': batch_size,
            'nice': nice,
            'ovthresh': ovthresh,  # used in mAP computation
            'input_range': 'norm01',
            # 'anyway': 'you want it',
            # 'thats the way': 'you need it',
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

        python ~/code/netharn/netharn/examples/yolo_voc.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
