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

# def s(d):
#     """ sorts a dict, returns as an OrderedDict """
#     return ub.odict(sorted(d.items()))


def asnumpy(tensor):
    return tensor.data.cpu().numpy()


def asfloat(t):
    return float(asnumpy(t))


class YoloVOCDataset(nh.data.voc.VOCDataset):
    """
    Extends VOC localization dataset (which simply loads the images in VOC2008
    with minimal processing) for multiscale training.

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

        self.factor = factor  # downsample factor of yolo grid

        self.base_wh = np.array(base_wh, dtype=np.int)

        assert np.all(self.base_wh % self.factor == 0)

        self.multi_scale_inp_size = np.array([
            self.base_wh + (self.factor * i) for i in range(*scales)])
        self.multi_scale_out_size = self.multi_scale_inp_size // self.factor

        self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                                   (6.63, 11.38), (9.42, 5.11),
                                   (16.62, 10.52)],
                                  dtype=np.float)
        self.num_anchors = len(self.anchors)
        self.augmenter = None

        if split == 'train':
            augmentors = [
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Affine(
                    scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-15, 15),
                    shear=(-7, 7),
                    order=[0, 1, 3],
                    cval=(0, 255),
                    mode=ia.ALL,
                    backend='cv2',
                ),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            ]
            self.augmenter = iaa.Sequential(augmentors)

    def _load_sized_image(self, index, inp_size):
        # load the raw data from VOC

        cacher = ub.Cacher('voc_img', cfgstr=ub.repr2([index, inp_size]),
                           appname='netharn')
        data = cacher.tryload()
        if data is None:
            image = self._load_image(index)
            orig_size = np.array(image.shape[0:2][::-1])
            factor = inp_size / orig_size
            # squish the image into network input coordinates
            interpolation = (cv2.INTER_AREA if factor.sum() <= 2 else
                             cv2.INTER_CUBIC)
            hwc255 = cv2.resize(image, tuple(inp_size),
                                interpolation=interpolation)
            data = hwc255, orig_size, factor
            cacher.save(data)

        hwc255, orig_size, factor = data
        return hwc255, orig_size, factor

    def _load_item(self, index, inp_size):
        # load the raw data from VOC
        hwc255, orig_size, factor = self._load_sized_image(index, inp_size)

        # VOC loads annotations in tlbr
        annot = self._load_annotation(index)
        tlbr_orig = util.Boxes(annot['boxes'].astype(np.float), 'tlbr')
        gt_classes = annot['gt_classes']
        # Weight samples so we dont care about difficult cases
        gt_weights = 1.0 - annot['gt_ishard'].astype(np.float)
        # squish the bounding box into network input coordinates
        tlbr = tlbr_orig.scale(factor).data

        return hwc255, orig_size, tlbr, gt_classes, gt_weights

    @profiler.profile
    def __getitem__(self, index):
        """
        Example:
            >>> self = YoloVOCDataset(split='train')
            >>> index = 1
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> boxes, class_idxs = label[0:2]
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.numpy(), box_format='tlbr')

        Ignore:
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
            inp_size = self.multi_scale_inp_size[size_index]
        else:
            inp_size = self.base_size

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

        # Remove boxes that are too small
        # ONLY DO THIS FOR THE SMALL DEMO TASK
        if False:
            tlbr = util.Boxes(tlbr, 'tlbr')
            flags = (tlbr.area > 10).ravel()
            tlbr = tlbr.data[flags]
            gt_classes = gt_classes[flags]
            gt_weights = gt_weights[flags]

        chw01 = torch.FloatTensor(hwc255.transpose(2, 0, 1) / 255)

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
        gt_weights = torch.FloatTensor(gt_weights)
        label = (target, gt_weights, orig_size, index)

        return chw01, label

    def _load_image(self, index):
        return super(YoloVOCDataset, self)._load_image(index)

    @ub.memoize_method
    def _load_annotation(self, index):
        return super(YoloVOCDataset, self)._load_annotation(index)


def make_loaders(datasets, batch_size=16, workers=0):
    """
    Example:
        >>> datasets = {'train': YoloVOCDataset(split='train')}
        >>> torch.random.manual_seed(0)
        >>> loaders = make_loaders(datasets)
        >>> train_iter = iter(loaders['train'])
        >>> # training batches should have multiple shapes
        >>> shapes = set()
        >>> for batch in train_iter:
        >>>     shapes.add(batch[0].shape[-1])
        >>>     if len(shapes) > 1:
        >>>         break
        >>> assert len(shapes) > 1
    """
    loaders = {}
    for key, dset in datasets.items():
        assert len(dset) > 0, 'must have some data'
        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            dset, batch_size=batch_size, shuffle=(key == 'train')
        )
        loader = torch_data.DataLoader(dset, batch_sampler=batch_sampler,
                                       collate_fn=collate.padded_collate,
                                       num_workers=workers)
        loader.batch_size = batch_size
        loaders[key] = loader
    return loaders


class YoloHarn(nh.FitHarn):
    def __init__(harn, **kw):
        super().__init__(**kw)
        harn.batch_confusions = []

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader
        """
        inputs, labels = batch
        outputs = harn.model(inputs)
        loss = harn.criterion(outputs, labels, seen=harn.n_seen)
        return outputs, loss

    def on_batch(harn, batch, outputs, loss):
        """ custom callback """
        inputs, labels = batch

        postout = harn.model.postproces(outputs)

        for y in harn._measure_confusion(postout, labels):
            harn.batch_confusions.append(y)

        metrics_dict = ub.odict()
        metrics_dict['L_bbox'] = asfloat(harn.criterion.bbox_loss)
        metrics_dict['L_iou'] = asfloat(harn.criterion.iou_loss)
        metrics_dict['L_cls'] = asfloat(harn.criterion.cls_loss)
        return metrics_dict

    def on_epoch(harn):
        """ custom callback """
        loader = harn.current.loader
        tag = harn.current.tag
        y = pd.concat(harn.batch_confusions)
        # TODO: write out a few visualizations
        num_classes = len(loader.dataset.label_names)
        ap_list = nh.evaluate.detection_ave_precision(y, num_classes)
        mean_ap = np.mean(ap_list)
        max_ap = np.nanmax(ap_list)
        harn.log_value(tag + ' epoch mAP', mean_ap, harn.epoch)
        harn.log_value(tag + ' epoch max-AP', max_ap, harn.epoch)
        harn.batch_confusions.clear()

    # Non-standard problem-specific custom methods

    def _measure_confusion(harn, postout, labels):
        bsize = len(labels[0])
        for bx in range(bsize):
            true_cxywh   = asnumpy(labels[0][bx])
            true_cxs     = asnumpy(labels[1][bx])
            # how much do we care about each annotation in this image?
            true_weight  = asnumpy(labels[2][bx])
            orig_size    = asnumpy(labels[3][bx])
            gx           = asnumpy(labels[4][bx])
            # how much do we care about the background in this image?
            bg_weight = asnumpy(labels[5][bx])

            pred_boxes = asnumpy(postout[0][bx])
            pred_scores = asnumpy(postout[1][bx])
            pred_cxs = asnumpy(postout[2][bx])

            tlbr = util.Boxes(true_cxywh, 'cywh').to_tlbr()
            true_boxes = tlbr.scale(orig_size).data

            y = nh.evaluate.image_confusions(
                true_boxes, true_cxs,
                pred_boxes, pred_scores, pred_cxs,
                true_weights=true_weight,
                bg_weight=bg_weight,
                ovthresh=harn.hyper.other['ovthresh'])

            y['gx'] = gx
            yield y

    def visualize_prediction(harn, inputs, outputs, labels):
        """
        Returns:
            np.ndarray: numpy image
        """
        pass


def setup_harness():
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py setup_harness

    Example:
        >>> harn = setup_harness()
        >>> harn.initialize()
    """
    from netharn.models.yolo2 import light_region_loss
    from netharn.models.yolo2 import light_yolo

    datasets = {
        'train': nh.data.YoloVOCDataset(tag='trainval'),
        'test': nh.data.YoloVOCDataset(tag='test'),
    }

    hyper = nh.HyperParams({
        'nice': 'Yolo2Baseline',
        'workdir': ub.truepath('~/work/voc_yolo2'),
        'datasets': datasets,

        # 'xpu': 'distributed(todo: fancy network stuff)',
        # 'xpu': 'cpu',
        'xpu': 'gpu',
        # 'xpu': 'gpu:0,1,2,3',

        # a single dict is applied to all datset loaders
        'loaders': {
            'batch_size': 16,
            'collate_fn': nh.collate.padded_collate,
            'workers': 0,
            # 'init_fn': None,
        },

        'model': (light_yolo.Yolo, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors,
            'conf_thresh': 0.001,
            'nms_thresh': 0.5,
            'ovthresh': 0.5,
        }),

        'criterion': (light_region_loss.RegionLoss, {
            'num_classes': datasets['train'].num_classes,
            'anchors': datasets['train'].anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,
            'class_scale': 1.0,
            'coord_scale': 1.0,
            'iou_thresh': 0.6,
        }),

        'initializer': (nh.Pretrained, {
            'fpath': light_yolo.initial_imagenet_weights(),
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': .001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }),

        'schedule': (nh.ListedLR, {
            'points': {0: .001, 10: .01,  60: .015, 90: .001},
            'interpolate': True
        }),

        'monitor': (nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['mAP'],
            'patience': 160,
            'max_epoch': 160,
        }),

        'augment': datasets['train'].augment,

        'other': {
            'input_range': 'norm01',
            'anyway': 'you want it',
            'thats the way': 'you need it',
        },
    })

    harn = YoloHarn(hyper=hyper)
    return harn


def train():
    harn = setup_harness()
    util.ensure_ulimit()
    with harn.xpu:
        harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py train
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
