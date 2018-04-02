# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import ubelt as ub
import numpy as np
import pandas as pd
import netharn as nh
from netharn import util

# def s(d):
#     """ sorts a dict, returns as an OrderedDict """
#     return ub.odict(sorted(d.items()))


def asnumpy(tensor):
    return tensor.data.cpu().numpy()


def asfloat(t):
    return float(asnumpy(t))


def train():
    harn = setup_harness()
    with harn.xpu:
        harn.run()


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
        'train': nh.data.VOC2012(tag='trainval'),
        'test': nh.data.VOC2012(tag='test'),
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
            'fpath': nh.Yolo2.grab_initial_weights('imagenet')
            # 'fpath': nh.initializers.KaimingNormal()
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': .001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }),

        'schedule': (nh.ListedLR, {
            'points': {0: .001, 10: .01,  60: .01, 90: .001},
            'interpolation': 'linear',
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py train
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
