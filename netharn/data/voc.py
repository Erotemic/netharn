"""
Simple dataset for loading the VOC 2007 object detection dataset without extra
bells and whistles. Simply loads the images, boxes, and class labels and
resizes images to a standard size.
"""
from os.path import exists
from os.path import join
import re
import scipy
import scipy.sparse
import cv2
import torch
import glob
import ubelt as ub
import numpy as np
from . import collate
import torch.utils.data as torch_data


class VOCDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Example:
        >>> assert len(VOCDataset(split='train', years=[2007])) == 2501
        >>> assert len(VOCDataset(split='test', years=[2007])) == 4952
        >>> assert len(VOCDataset(split='val', years=[2007])) == 2510
        >>> assert len(VOCDataset(split='trainval', years=[2007])) == 5011

        >>> assert len(VOCDataset(split='train', years=[2007, 2012])) == 8218
        >>> assert len(VOCDataset(split='test', years=[2007, 2012])) == 4952
        >>> assert len(VOCDataset(split='val', years=[2007, 2012])) == 8333

    Example:
        >>> years = [2007, 2012]
        >>> self = VOCDataset()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)

    Example:
        >>> self = VOCDataset()

    """
    def __init__(self, devkit_dpath=None, split='train', years=[2007, 2012]):
        if devkit_dpath is None:
            # ub.truepath('~/data/VOC/VOCdevkit')
            devkit_dpath = self.ensure_voc_data(years=years)

        self.devkit_dpath = devkit_dpath
        self.years = years

        # determine train / test splits
        self.gpaths = []
        self.apaths = []
        if split == 'test':
            assert 2007 in years, 'test set is hacked to be only 2007'
            gps, aps = self._read_split_paths('test', 2007)
            self.gpaths += gps
            self.apaths += aps
        else:
            for year in sorted(years):
                gps, aps = self._read_split_paths(split, year)
                self.gpaths += gps
                self.apaths += aps

        self.label_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                            'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train',
                            'tvmonitor')
        self._class_to_ind = ub.invert_dict(dict(enumerate(self.label_names)))
        self.base_size = [320, 320]

        self.num_classes = len(self.label_names)

        import os
        hashid = ub.hash_data(list(map(os.path.basename, self.gpaths)))
        yearid = '_'.join(map(str, years))
        self.input_id = 'voc_{}_{}_{}'.format(yearid, split, hashid)

    def _read_split_paths(self, split, year):
        """
        split = 'train'
        self = VOCDataset('test')
        year = 2007
        year = 2012
        """
        split_idstrs = []
        data_dpath = join(self.devkit_dpath, 'VOC{}'.format(year))
        split_dpath = join(data_dpath, 'ImageSets', 'Main')
        pattern = join(split_dpath, '*_' + split + '.txt')
        for p in sorted(glob.glob(pattern)):
            rows = [list(re.split(' +', t)) for t in ub.readfrom(p).split('\n') if t]
            # code = -1 if the image does not contain the object
            # code = 1 if the image contains at least one instance
            # code = 0 if the image contains only hard instances of the object
            idstrs = [idstr for idstr, code in rows if int(code) == 1]
            split_idstrs.extend(idstrs)
        split_idstrs = sorted(set(split_idstrs))

        image_dpath = join(data_dpath, 'JPEGImages')
        annot_dpath = join(data_dpath, 'Annotations')
        gpaths = [join(image_dpath, '{}.jpg'.format(idstr))
                  for idstr in split_idstrs]
        apaths = [join(annot_dpath, '{}.xml'.format(idstr))
                  for idstr in split_idstrs]
        return gpaths, apaths
        # for p in gpaths:
        #     assert exists(p)
        # for p in apaths:
        #     assert exists(p)
        # return split_idstrs

    @classmethod
    def ensure_voc_data(VOCDataset, dpath=None, force=False, years=[2007, 2012]):
        """
        Download the Pascal VOC 2007 data if it does not already exist.

        CommandLine:
            python -m clab.data.voc VOCDataset.ensure_voc_data

        Example:
            >>> # SCRIPT
            >>> from clab.data.voc import *  # NOQA
            >>> VOCDataset.ensure_voc_data()
        """
        if dpath is None:
            dpath = ub.truepath('~/data/VOC')
        devkit_dpath = join(dpath, 'VOCdevkit')
        # if force or not exists(devkit_dpath):
        ub.ensuredir(dpath)

        fpath1 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar', dpath=dpath)
        if force or not exists(join(dpath, 'VOCdevkit', 'VOCcode')):
            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath1, dpath), verbout=1)

        if 2007 in years:
            # VOC 2007 train+validation data
            fpath2 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', dpath=dpath)
            if force or not exists(join(dpath, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'bird_trainval.txt')):
                ub.cmd('tar xvf "{}" -C "{}"'.format(fpath2, dpath), verbout=1)

            # VOC 2007 test data
            fpath3 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar', dpath=dpath)
            if force or not exists(join(dpath, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'bird_test.txt')):
                ub.cmd('tar xvf "{}" -C "{}"'.format(fpath3, dpath), verbout=1)

        if 2012 in years:
            # VOC 2012 train+validation data
            fpath4 = ub.grabdata('https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar', dpath=dpath)
            if force or not exists(join(dpath, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'bird_trainval.txt')):
                ub.cmd('tar xvf "{}" -C "{}"'.format(fpath4, dpath), verbout=1)
        return devkit_dpath

    def __nice__(self):
        return '{} {}'.format(self.split, len(self))

    def __len__(self):
        return len(self.gpaths)

    def __getitem__(self, index):
        """
        Returns:
            image, (bbox, class_idxs)

            bbox and class_idxs are variable-length
            bbox is in x1,y1,x2,y2 (i.e. tlbr) format

        Example:
            >>> self = VOCDataset()
            >>> chw, label = self[1]
            >>> hwc = chw.numpy().transpose(1, 2, 0)
            >>> boxes, class_idxs = label
            >>> # xdoc: +REQUIRES(--show)
            >>> from clab.util import mplutil
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> mplutil.imshow(hwc, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.numpy(), box_format='tlbr')
            >>> mplutil.show_if_requested()
        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, inp_size = index
        else:
            inp_size = self.base_size
        hwc, boxes, gt_classes = self._load_item(index, inp_size)

        chw = torch.FloatTensor(hwc.transpose(2, 0, 1))
        gt_classes = torch.LongTensor(gt_classes)
        boxes = torch.FloatTensor(boxes)
        label = (boxes, gt_classes,)
        return chw, label

    def _load_item(self, index, inp_size=None):
        # from clab.models.yolo2.utils.yolo import _offset_boxes
        image = self._load_image(index)
        annot = self._load_annotation(index)

        boxes = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        # squish the bounding box and image into a standard size
        if inp_size is None:
            return image, boxes, gt_classes
        else:
            w, h = inp_size
            sx = float(w) / image.shape[1]
            sy = float(h) / image.shape[0]
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
            interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
            hwc = cv2.resize(image, (w, h), interpolation=interpolation)
            return hwc, boxes, gt_classes

    def _load_image(self, index):
        fpath = self.gpaths[index]
        imbgr = cv2.imread(fpath)
        imrgb_255 = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)
        return imrgb_255

    def _load_annotation(self, index):
        import xml.etree.ElementTree as ET
        fpath = self.apaths[index]
        tree = ET.parse(fpath)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc is None else int(diffc.text)
            ishards[ix] = difficult

            clsname = obj.find('name').text.lower().strip()
            cls = self._class_to_ind[clsname]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        annots = {'boxes': boxes,
                  'gt_classes': gt_classes,
                  'gt_ishard': ishards,
                  'gt_overlaps': overlaps,
                  'flipped': False,
                  'fpath': fpath,
                  'seg_areas': seg_areas}
        return annots

    def make_loader(self, *args, **kwargs):
        """
        We need to do special collation to deal with different numbers of
        bboxes per item.

        Args:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data
                reshuffled at every epoch (default: False).
            sampler (Sampler, optional): defines the strategy to draw samples
                from the dataset. If specified, ``shuffle`` must be False.
            batch_sampler (Sampler, optional): like sampler, but returns a
                batch of indices at a time. Mutually exclusive with batch_size,
                shuffle, sampler, and drop_last.
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main
                process.  (default: 0)
            pin_memory (bool, optional): If ``True``, the data loader will copy
                tensors into CUDA pinned memory before returning them.
            drop_last (bool, optional): set to ``True`` to drop the last
                incomplete batch, if the dataset size is not divisible by the
                batch size. If ``False`` and the size of dataset is not
                divisible by the batch size, then the last batch will be
                smaller. (default: False)
            timeout (numeric, optional): if positive, the timeout value for
                collecting a batch from workers. Should always be non-negative.
                (default: 0)
            worker_init_fn (callable, optional): If not None, this will be
                called on each worker subprocess with the worker id (an int in
                ``[0, num_workers - 1]``) as input, after seeding and before
                data loading. (default: None)

        References:
            https://github.com/pytorch/pytorch/issues/1512

        Example:
            >>> self = VOCDataset()
            >>> #inbatch = [self[i] for i in range(10)]
            >>> loader = self.make_loader(batch_size=10)
            >>> batch = next(iter(loader))
            >>> images, labels = batch
            >>> assert len(images) == 10
            >>> assert len(labels) == 2
            >>> assert len(labels[0]) == len(images)
        """
        kwargs['collate_fn'] = collate.list_collate
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader


class EvaluateVOC(object):
    """
    Example:
        >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes()
        >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
    """
    def __init__(self, all_true_boxes, all_pred_boxes):
        self.all_true_boxes = all_true_boxes
        self.all_pred_boxes = all_pred_boxes

    @classmethod
    def perterb_boxes(EvaluateVOC, boxes, perterb_amount=.5, rng=None, cxs=None,
                      num_classes=None):
        n = boxes.shape[0]
        if boxes.shape[0] == 0:
            boxes = np.array([[10, 10, 50, 50, 1]])

        # add twice as many boxes,
        boxes = np.vstack([boxes, boxes])
        n_extra = len(boxes) - n
        # perterb the positions
        xywh = np.hstack([boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2]])
        scale = np.sqrt(xywh.max()) * perterb_amount
        pred_xywh = xywh + rng.randn(*xywh.shape) * scale

        # randomly keep some
        keep1 = rng.rand(n) >= min(perterb_amount, .5)
        keep2 = rng.rand(n_extra) < min(perterb_amount, .5)
        keep = np.hstack([keep1, keep2])

        if cxs is not None:
            # randomly change class indexes
            cxs2 = list(cxs) + list(rng.randint(0, num_classes, n_extra))
            cxs2 = np.array(cxs2)
            change = rng.rand(n) < min(perterb_amount, 1.0)
            cxs2[:n][change] = list(rng.randint(0, num_classes, sum(change)))
            cxs2 = cxs2[keep]

        pred = pred_xywh[keep].astype(np.uint8)
        pred_boxes = np.hstack([pred[:, 0:2], pred[:, 0:2] + pred[:, 2:4]])
        # give dummy scores
        pred_boxes = np.hstack([pred_boxes, rng.rand(len(pred_boxes), 1)])
        if cxs is not None:
            return pred_boxes, cxs2
        else:
            return pred_boxes

    @classmethod
    def random_boxes(EvaluateVOC, n=None, c=4, rng=None):
        if rng is None:
            rng = np.random
        if n is None:
            n = rng.randint(0, 10)
        xywh = (rng.rand(n, 4) * 100).astype(np.int)
        tlbr = np.hstack([xywh[:, 0:2], xywh[:, 0:2] + xywh[:, 2:4]])
        cxs = (rng.rand(n) * c).astype(np.int)
        return tlbr, cxs

    @classmethod
    def demodata_boxes(EvaluateVOC, perterb_amount=.5, rng=0):
        """
        Example:
            >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(100, 0)
            >>> print(ub.repr2(all_true_boxes, nl=3, precision=2))
            >>> print(ub.repr2(all_pred_boxes, nl=3, precision=2))
            >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(0, 0)
            >>> print(ub.repr2(all_true_boxes, nl=3, precision=2))
            >>> print(ub.repr2(all_pred_boxes, nl=3, precision=2))
        """
        all_true_boxes = [
            # class 1
            [
                # image 1
                [[100, 100, 200, 200, 1]],
                # image 2
                np.empty((0, 5)),
                # image 3
                [[0, 10, 10, 20, 1], [10, 10, 20, 20, 1], [20, 10, 30, 20, 1]],
            ],
            # class 2
            [
                # image 1
                [[0, 0, 100, 100, 1], [0, 0, 50, 50, 1]],
                # image 2
                [[0, 0, 50, 50, 1], [50, 50, 100, 100, 1]],
                # image 3
                [[0, 0, 10, 10, 1], [10, 0, 20, 10, 1], [20, 0, 30, 10, 1]],
            ],
        ]
        # convert to numpy
        for cx, class_boxes in enumerate(all_true_boxes):
            for gx, boxes in enumerate(class_boxes):
                all_true_boxes[cx][gx] = np.array(boxes)

        # setup perterbubed demo predicted boxes
        rng = np.random.RandomState(rng)

        all_pred_boxes = []
        for cx, class_boxes in enumerate(all_true_boxes):
            all_pred_boxes.append([])
            for gx, boxes in enumerate(class_boxes):
                pred_boxes = EvaluateVOC.perterb_boxes(boxes, perterb_amount,
                                                       rng)
                all_pred_boxes[cx].append(pred_boxes)

        return all_true_boxes, all_pred_boxes

    @classmethod
    def find_overlap(EvaluateVOC, true_boxes, pred_box):
        """
        Compute iou of `pred_box` with each `true_box in true_boxes`.
        Return the index and score of the true box with maximum overlap.

        Example:
            >>> true_boxes = np.array([[ 0,  0, 10, 10, 1],
            >>>                        [10,  0, 20, 10, 1],
            >>>                        [20,  0, 30, 10, 1]])
            >>> pred_box = np.array([6, 2, 20, 10, .9])
            >>> ovmax, ovidx = EvaluateVOC.find_overlap(true_boxes, pred_box)
            >>> print('ovidx = {!r}'.format(ovidx))
            ovidx = 1
        """
        if True:
            from clab.models.yolo2.utils import yolo_utils
            true_boxes = np.array(true_boxes)
            pred_box = np.array(pred_box)
            overlaps = yolo_utils.bbox_ious(
                true_boxes[:, 0:4].astype(np.float),
                pred_box[None, :][:, 0:4].astype(np.float)).ravel()
        else:
            bb = pred_box
            # intersection
            ixmin = np.maximum(true_boxes[:, 0], bb[0])
            iymin = np.maximum(true_boxes[:, 1], bb[1])
            ixmax = np.minimum(true_boxes[:, 2], bb[2])
            iymax = np.minimum(true_boxes[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (true_boxes[:, 2] - true_boxes[:, 0] + 1.) *
                   (true_boxes[:, 3] - true_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
        ovidx = overlaps.argmax()
        ovmax = overlaps[ovidx]
        return ovmax, ovidx

    def compute(self, ovthresh=0.5):
        """
        Example:
            >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(.5)
            >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
            >>> ovthresh = 0.5
            >>> mean_ap = self.compute(ovthresh)[0]
            >>> print('mean_ap = {:.2f}'.format(mean_ap))
            mean_ap = 0.36
            >>> all_true_boxes, all_pred_boxes = EvaluateVOC.demodata_boxes(0)
            >>> self = EvaluateVOC(all_true_boxes, all_pred_boxes)
            >>> ovthresh = 0.5
            >>> mean_ap = self.compute(ovthresh)[0]
            >>> print('mean_ap = {:.2f}'.format(mean_ap))
            mean_ap = 1.00
        """
        num_classes = len(self.all_true_boxes)
        ap_list2 = []
        for cx in range(num_classes):
            rec, prec, ap = self.eval_class(cx, ovthresh)
            ap_list2.append(ap)
        mean_ap2 = np.nanmean(ap_list2)
        return mean_ap2, ap_list2

    def eval_class(self, cx, ovthresh=0.5):
        all_true_boxes = self.all_true_boxes
        all_pred_boxes = self.all_pred_boxes

        cls_true_boxes = all_true_boxes[cx]
        cls_pred_boxes = all_pred_boxes[cx]

        # Flatten the predicted boxes
        import pandas as pd
        flat_pred_boxes = []
        flat_pred_gxs = []
        for gx, pred_boxes in enumerate(cls_pred_boxes):
            flat_pred_boxes.extend(pred_boxes)
            flat_pred_gxs.extend([gx] * len(pred_boxes))
        flat_pred_boxes = np.array(flat_pred_boxes)

        npos = sum([(b.T[4] > 0).sum() for b in cls_true_boxes if len(b)])
        # npos = sum(map(len, cls_true_boxes))

        if npos == 0:
            return [], [], np.nan

        if len(flat_pred_boxes) > 0:
            flat_preds = pd.DataFrame({
                'box': flat_pred_boxes[:, 0:4].tolist(),
                'conf': flat_pred_boxes[:, 4],
                'gx': flat_pred_gxs
            })
            flat_preds = flat_preds.sort_values('conf', ascending=False)

            # Keep track of which true boxes have been assigned in this class /
            # image pair.
            assign = {}

            # Greedy assignment for scoring
            nd = len(flat_preds)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            # Iterate through predicted bounding boxes in order of descending
            # confidence
            for sx, (pred_id, pred) in enumerate(flat_preds.iterrows()):
                gx, pred_box = pred[['gx', 'box']]
                true_boxes = cls_true_boxes[gx]

                ovmax = -np.inf
                true_id = None
                if len(true_boxes):
                    true_weights = true_boxes.T[4]
                    ovmax, ovidx = self.find_overlap(true_boxes, pred_box)
                    true_id = (gx, ovidx)

                if ovmax > ovthresh and true_id not in assign:
                    if true_weights[ovidx] > 0:
                        assign[true_id] = pred_id
                        tp[sx] = 1
                else:
                    fp[sx] = 1

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)

            if npos == 0:
                rec = 1
            else:
                rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            ap = EvaluateVOC.voc_ap(rec, prec, use_07_metric=True)
            return rec, prec, ap
        else:
            if npos == 0:
                return [], [], np.nan
            else:
                return [], [], 0.0

    @classmethod
    def voc_ap(EvaluateVOC, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @classmethod
    def sanity_check(EvaluateVOC):
        """
        Example:
            >>> from clab.data.voc import *
            >>> EvaluateVOC.sanity_check()
        """
        import pandas as pd
        n_images = 100
        ovthresh = 0.8
        num_classes = 200
        rng = np.random.RandomState(0)
        for perterb_amount in [0, .00001, .0001, .0005, .001, .01, .1, .5]:
            img_ys = []

            all_true_boxes = [[] for cx in range(num_classes)]
            all_pred_boxes = [[] for cx in range(num_classes)]

            for index in range(n_images):
                n = rng.randint(0, 50)
                true_boxes, true_cxs = EvaluateVOC.random_boxes(n=n,
                                                                c=num_classes,
                                                                rng=rng)
                if len(true_boxes):
                    # flip every other box to have weight 0
                    true_boxes = np.hstack([true_boxes, np.ones((len(true_boxes), 1))])
                    true_boxes[::2, 4] = 0

                pred_sboxes, pred_cxs = EvaluateVOC.perterb_boxes(
                    true_boxes, perterb_amount=perterb_amount,
                    cxs=true_cxs, rng=rng, num_classes=num_classes)
                pred_scores = pred_sboxes[:, 4]
                pred_boxes = pred_sboxes[:, 0:4]
                y = EvaluateVOC.image_confusions(true_boxes, true_cxs,
                                                 pred_boxes, pred_scores,
                                                 pred_cxs, ovthresh=ovthresh)
                y['gx'] = index
                img_ys.append(y)

                # Build format2
                for cx in range(num_classes):
                    all_true_boxes[cx].append(true_boxes[true_cxs == cx])
                    all_pred_boxes[cx].append(pred_sboxes[pred_cxs == cx])

            y = pd.concat(img_ys)
            mean_ap1, ap_list1 = EvaluateVOC.compute_map(y, num_classes)

            self = EvaluateVOC(all_true_boxes, all_pred_boxes)
            mean_ap2, ap_list2 = self.compute(ovthresh=ovthresh)
            print('mean_ap1 = {!r}'.format(mean_ap1))
            print('mean_ap2 = {!r}'.format(mean_ap2))
            assert mean_ap2 == mean_ap1
            # print('ap_list1 = {!r}'.format(ap_list1))
            # print('ap_list2 = {!r}'.format(ap_list2))
            print('-------')

    @classmethod
    def image_confusions(EvaluateVOC, true_boxes, true_cxs, pred_boxes,
                         pred_scores, pred_cxs, ovthresh=0.5):
        """
        Given predictions and truth for an image return (y_pred, y_true,
        y_score), which is suitable for sklearn classification metrics

        Returns:
            pd.DataFrame: with relevant clf information

        Example:
            >>> true_boxes = np.array([[ 0,  0, 10, 10, 1],
            >>>                        [10,  0, 20, 10, 0],
            >>>                        [10,  0, 20, 10, 1],
            >>>                        [20,  0, 30, 10, 1]])
            >>> true_cxs = np.array([0, 0, 1, 1])
            >>> pred_boxes = np.array([[6, 2, 20, 10],
            >>>                        [3,  2, 9, 7],
            >>>                        [20,  0, 30, 10]])
            >>> pred_scores = np.array([.5, .5, .5])
            >>> pred_cxs = np.array([0, 0, 1])
            >>> y = EvaluateVOC.image_confusions(true_boxes, true_cxs,
            >>>                                  pred_boxes, pred_scores,
            >>>                                  pred_cxs)
        """
        import pandas as pd
        y_pred = []
        y_true = []
        y_score = []
        cxs = []

        cx_to_boxes = ub.group_items(true_boxes, true_cxs)
        cx_to_boxes = ub.map_vals(np.array, cx_to_boxes)
        # Keep track which true boxes are unused / not assigned
        cx_to_unused = {cx: [True] * len(boxes)
                        for cx, boxes in cx_to_boxes.items()}

        # sort predictions by score
        sortx = pred_scores.argsort()[::-1]
        pred_boxes  = pred_boxes.take(sortx, axis=0)
        pred_cxs    = pred_cxs.take(sortx, axis=0)
        pred_scores = pred_scores.take(sortx, axis=0)
        for cx, box, score in zip(pred_cxs, pred_boxes, pred_scores):
            cls_true_boxes = cx_to_boxes.get(cx, [])
            ovmax = -np.inf
            ovidx = None
            if len(cls_true_boxes):
                unused = cx_to_unused[cx]
                ovmax, ovidx = EvaluateVOC.find_overlap(cls_true_boxes, box)
                true_weights = cls_true_boxes.T[4]

            if ovmax > ovthresh and unused[ovidx]:
                # Mark this prediction as a true positive
                if true_weights[ovidx] > 0:
                    # Ignore matches to truth with weight 0
                    y_pred.append(cx)
                    y_true.append(cx)
                    y_score.append(score)
                    cxs.append(cx)
                    unused[ovidx] = False
            else:
                # Mark this prediction as a false positive
                y_pred.append(cx)
                y_true.append(-1)  # use -1 as ignore class
                y_score.append(score)
                cxs.append(cx)

        # Mark true boxes we failed to predict as false negatives
        for cx, unused in cx_to_unused.items():
            cls_true_boxes = cx_to_boxes.get(cx, [])
            for ovidx, flag in enumerate(unused):
                if flag:
                    # it has a nonzero weight
                    weight = cls_true_boxes.T[4][ovidx]
                    if  weight > 0:
                        # Mark this prediction as a false negative
                        y_pred.append(-1)
                        y_true.append(cx)
                        y_score.append(0.0)
                        cxs.append(cx)

        y = pd.DataFrame({
            'pred': y_pred,
            'true': y_true,
            'score': y_score,
            'cx': cxs,
        })
        return y

    @classmethod
    def compute_map(EvaluateVOC, y, num_classes):
        def group_metrics(group):
            if group is None:
                return np.nan
            group = group.sort_values('score', ascending=False)
            npos = sum(group.true >= 0)
            dets = group[group.pred > -1]
            if npos == 0:
                return np.nan
            if len(dets) == 0:
                if npos == 0:
                    return np.nan
                return 0.0
            tp = (dets.pred == dets.true).values.astype(np.uint8)
            fp = 1 - tp
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)

            eps = np.finfo(np.float64).eps
            if npos == 0:
                rec = 1
            else:
                rec = tp / npos
            prec = tp / np.maximum(tp + fp, eps)

            ap = EvaluateVOC.voc_ap(rec, prec, use_07_metric=True)
            return ap

        # because we use -1 to indicate a wrong prediction we can use max to
        # determine the class groupings.
        cx_to_group = dict(iter(y.groupby('cx')))
        ap_list1 = []
        for cx in range(num_classes):
            # for cx, group in cx_to_group.items():
            group = cx_to_group.get(cx, None)
            ap = group_metrics(group)
            ap_list1.append(ap)
        mean_ap1 = np.nanmean(ap_list1)
        return mean_ap1, ap_list1

    # === Original Method 1
    # def on_epoch1(harn, tag, loader):

    #     # Measure accumulated outputs
    #     num_images = len(loader.dataset)
    #     num_classes = loader.dataset.num_classes
    #     all_pred_boxes = [[[] for _ in range(num_images)]
    #                       for _ in range(num_classes)]
    #     all_true_boxes = [[[] for _ in range(num_images)]
    #                       for _ in range(num_classes)]

    #     # cx = 3
    #     # cx = 7
    #     print(ub.repr2([(gx, b) for gx, b in enumerate(all_true_boxes[cx]) if len(b)], nl=1))
    #     print(ub.repr2([(gx, b) for gx, b in enumerate(all_pred_boxes[cx]) if len(b)], nl=1))

    #     # Iterate over output from each batch
    #     for postout, labels in harn.accumulated:

    #         # Iterate over each item in the batch
    #         cls_pred_boxes, batch_pred_scores, batch_pred_cls_inds = postout
    #         cls_true_boxes, batch_true_cls_inds = labels[0:2]
    #         batch_orig_sz, batch_img_inds = labels[2:4]

    #         batch_size = len(labels[0])
    #         for bx in range(batch_size):
    #             gx = batch_img_inds[bx]

    #             true_boxes = cls_true_boxes[bx].data.cpu().numpy()
    #             true_cxs = batch_true_cls_inds[bx]

    #             pred_boxes  = cls_pred_boxes[bx]
    #             pred_scores = batch_pred_scores[bx]
    #             pred_cxs    = batch_pred_cls_inds[bx]

    #             for cx, boxes, score in zip(pred_cxs, pred_boxes, pred_scores):
    #                 all_pred_boxes[cx][gx].append(np.hstack([boxes, score]))

    #             for cx, boxes in zip(true_cxs, true_boxes):
    #                 all_true_boxes[cx][gx].append(boxes)

    #     all_boxes = all_true_boxes
    #     for cx, class_boxes in enumerate(all_boxes):
    #         for gx, boxes in enumerate(class_boxes):
    #             all_boxes[cx][gx] = np.array(boxes)
    #             if len(boxes):
    #                 boxes = np.array(boxes)
    #             else:
    #                 boxes = np.empty((0, 4))
    #             all_boxes[cx][gx] = boxes

    #     all_boxes = all_pred_boxes
    #     for cx, class_boxes in enumerate(all_boxes):
    #         for gx, boxes in enumerate(class_boxes):
    #             # Sort predictions by confidence
    #             if len(boxes):
    #                 boxes = np.array(boxes)
    #             else:
    #                 boxes = np.empty((0, 5))
    #             all_boxes[cx][gx] = boxes

    #     self = voc.EvaluateVOC(all_true_boxes, all_pred_boxes)
    #     ovthresh = 0.5
    #     mean_ap1 = self.compute(ovthresh)
    #     print('mean_ap1 = {!r}'.format(mean_ap1))

    #     num_classes = len(self.all_true_boxes)
    #     ap_list1 = []
    #     for cx in range(num_classes):
    #         rec, prec, ap = self.eval_class(cx, ovthresh)
    #         ap_list1.append(ap)
    #     print('ap_list1 = {!r}'.format(ap_list1))

    #     # reset accumulated for next epoch
    #     harn.accumulated.clear()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.data.voc all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
