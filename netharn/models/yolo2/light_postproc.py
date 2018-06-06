#
#   Lightnet related postprocessing
#   Thers are functions to transform the output of the network to brambox detection objects
#   Copyright EAVISE
#
import torch
# from torch.autograd import Variable
from netharn.util import profiler
from netharn import util
import numpy as np
import ubelt as ub


class GetBoundingBoxes(object):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        network (lightnet.network.Darknet): Network the converter will be used with
        conf_thresh (Number [0-1]): Confidence threshold to filter detections
        nms_thresh(Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion

    Returns:
        (Batch x Boxes x 6 tensor): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """

    def __init__(self, num_classes, anchors, conf_thresh=0.001, nms_thresh=0.4):
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    @profiler.profile
    def __call__(self, network_output, mode=0):
        """ Compute bounding boxes after thresholding and nms

            network_output (torch.autograd.Variable): Output tensor from the lightnet network

        Examples:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> output = torch.randn(8, 5, 5 + 20, 9, 9)
            >>> boxes = self(output)
            >>> assert len(boxes) == 8
            >>> assert all(b.shape[1] == 6 for b in boxes)

        CommandLine:
            python -m netharn.models.yolo2.light_postproc GetBoundingBoxes.__call__:1 --profile
            python -m netharn.models.yolo2.light_postproc GetBoundingBoxes.__call__:2 --profile

        Script:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> import ubelt
            >>> output = torch.randn(16, 5, 5 + 20, 9, 9)
            >>> output = output.cuda()
            >>> for timer in ubelt.Timerit(21, bestof=3, label='mode0+gpu'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         self(output_, mode=0)

        Script:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> import ubelt
            >>> output = torch.randn(16, 5, 5 + 20, 9, 9)
            >>> output = output.cuda()
            >>> for timer in ubelt.Timerit(21, bestof=3, label='mode1+gpu'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         self(output_, mode=1)

        Benchmark:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> import ubelt
            >>> output = torch.randn(16, 5, 5 + 20, 9, 9)
            >>> #
            >>> for timer in ubelt.Timerit(21, bestof=3, label='mode0+cpu'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         self(output_, mode=0)
            >>> #
            >>> for timer in ubelt.Timerit(21, bestof=3, label='mode1+cpu'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         self(output_, mode=1)
            >>> #
            >>> if torch.cuda.is_available():
            >>>     output = output.cuda()
            >>>     for timer in ubelt.Timerit(21, bestof=3, label='mode0+gpu'):
            >>>         output_ = output.clone()
            >>>         with timer:
            >>>             self(output_, mode=0)
            >>>     #
            >>>     for timer in ubelt.Timerit(21, bestof=3, label='mode1+gpu'):
            >>>         output_ = output.clone()
            >>>         with timer:
            >>>             self(output_, mode=1)

            %timeit self(output.data, mode=0)
            %timeit self(output.data, mode=1)
            %timeit self(output.data, mode=2)
        """
        boxes = self._get_boxes(network_output.data, mode=mode)
        boxes = [self._nms(box, mode=mode) for box in boxes]

        # force all boxes to be inside the image
        boxes = [self._clip_boxes(box) for box in boxes]
        postout = boxes
        return postout

    @profiler.profile
    def _clip_boxes(self, box):
        """
        CommandLine:
            python ~/code/netharn/netharn/models/yolo2/light_postproc.py GetBoundingBoxes._clip_boxes

        Example:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> # Make random boxes for one item in a batch
            >>> box = torch.randn(7, 6)
            >>> box[:, 0].sigmoid_()
            >>> box[:, 1].sigmoid_()
            >>> box.abs_()
            >>> new_box = self._clip_boxes(box)
            >>> box_tlbr = util.Boxes(box.cpu().numpy()[:, 0:4], 'cxywh').to_tlbr().data
            >>> new_tlbr = util.Boxes(new_box.cpu().numpy()[:, 0:4], 'cxywh').to_tlbr().data
            >>> #
            >>> print('old')
            >>> print(box_tlbr)
            >>> print('new')
            >>> print(new_tlbr)
            >>> #print('trim_w = {}'.format(ub.repr2(trim_w.numpy(), precision=4)))
            >>> #print('trim_h = {}'.format(ub.repr2(trim_h.numpy(), precision=4)))
            >>> assert np.all(new_tlbr.T[2] <= 1.01)
            >>> assert np.all(new_tlbr.T[2] >= -0.01)
            >>> assert np.all(new_tlbr.T[3] <= 1.01)
            >>> assert np.all(new_tlbr.T[3] >= -0.01)
        """
        if len(box) == 0:
            return box

        cx, cy, w, h = box.t()[0:4]

        x1 = cx - (w / 2)
        x2 = cx + (w / 2)

        y1 = cy - (h / 2)
        y2 = cy + (h / 2)

        trim_w1 = (0 - x1).clamp(0, None)
        trim_w2 = (x2 - 1).clamp(0, None)
        # multiply by 2 because we are trimming from both sides to ensure the
        # center prediction stays the same.
        trim_w = torch.max(trim_w1, trim_w2) * 2

        trim_h1 = (0 - y1).clamp(0, None)
        trim_h2 = (y2 - 1).clamp(0, None)
        trim_h = torch.max(trim_h1, trim_h2) * 2

        new_box = box.clone()
        new_box[:, 2] = new_box[:, 2] - trim_w
        new_box[:, 3] = new_box[:, 3] - trim_h
        return new_box

    @classmethod
    def apply(cls, network_output, num_classes, anchors, conf_thresh, nms_thresh):
        obj = cls(num_classes, anchors, conf_thresh, nms_thresh)
        return obj(network_output)

    @profiler.profile
    def _get_boxes(self, output, mode=0):
        """
        Returns array of detections for every image in batch

        CommandLine:
            python ~/code/netharn/netharn/models/yolo2/light_postproc.py GetBoundingBoxes._get_boxes

        Examples:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> output = torch.randn(16, 5, 5 + 20, 9, 9)
            >>> from netharn import XPU
            >>> output = XPU.cast('auto').move(output)
            >>> boxes = self._get_boxes(output.data)
            >>> assert len(boxes) == 16
            >>> assert all(len(b[0]) == 6 for b in boxes)

        Benchmark:
            >>> from netharn.models.yolo2.light_postproc import *
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> output = torch.randn(16, 5, 5 + 20, 9, 9)
            >>> from netharn import XPU
            >>> output = XPU.cast('auto').move(output)
            >>> for timer in ub.Timerit(100, bestof=10, label='mode 0'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         boxes0 = self._get_boxes(output_.data, mode=0)
            >>> for timer in ub.Timerit(100, bestof=10, label='mode 1'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         boxes1 = self._get_boxes(output_.data, mode=1)
            >>> for b0, b1 in zip(boxes0, boxes1):
            >>>     assert np.all(b0.cpu() == b1.cpu())
            >>> from lightnet.data.postprocess import GetBoundingBoxes as GetBoundingBoxesOrig
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> post = GetBoundingBoxesOrig(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> for timer in ub.Timerit(100, bestof=10, label='original'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         boxes3 = post._get_boxes(output_.data)
            >>> # Check that the output is the same
            >>> for b0, b3 in zip(boxes0, boxes3):
            >>>     b3_ = torch.Tensor(b3)
            >>>     assert np.all(b0.cpu() == b3_.cpu())
        """
        # dont modify inplace
        output = output.clone()

        # Check dimensions
        # if output.dim() == 3:
        #     output.unsqueeze_(0)

        # Variables
        bsize = output.shape[0]
        h, w = output.shape[-2:]

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
        lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
        anchor_w = self.anchors[:, 0].contiguous().view(1, self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(1, self.num_anchors, 1)
        if output.is_cuda:
            lin_x = lin_x.cuda(output.device)
            lin_y = lin_y.cuda(output.device)
            anchor_w = anchor_w.cuda(output.device)
            anchor_h = anchor_h.cuda(output.device)

        # -1 == 5+num_classes (we can drop feature maps if 1 class)
        output_ = output.view(bsize, self.num_anchors, -1, h * w)
        output_[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)          # X center
        output_[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)          # Y center
        output_[:, :, 2, :].exp_().mul_(anchor_w).div_(w)           # Width
        output_[:, :, 3, :].exp_().mul_(anchor_h).div_(h)           # Height
        output_[:, :, 4, :].sigmoid_()                              # Box score

        # Compute class_score
        if self.num_classes > 1:
            cls_scores = torch.nn.functional.softmax(output_[:, :, 5:, :], 2)
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max.mul_(output_[:, :, 4, :])
        else:
            cls_max = output_[:, :, 4, :]
            cls_max_idx = torch.zeros_like(cls_max)

        # Save detection if conf*class_conf is higher than threshold

        if mode == 0:
            output_ = output_.cpu()
            cls_max = cls_max.cpu()
            cls_max_idx = cls_max_idx.cpu()
            boxes = []
            for b in range(bsize):
                box_batch = []
                for a in range(self.num_anchors):
                    for i in range(h * w):
                        if cls_max[b, a, i] > self.conf_thresh:
                            box_batch.append([
                                output_[b, a, 0, i],
                                output_[b, a, 1, i],
                                output_[b, a, 2, i],
                                output_[b, a, 3, i],
                                cls_max[b, a, i],
                                cls_max_idx[b, a, i]
                            ])
                box_batch = torch.Tensor(box_batch)
                boxes.append(box_batch)
        elif mode == 1 or mode == 2:
            # Save detection if conf*class_conf is higher than threshold
            flags = cls_max > self.conf_thresh
            flat_flags = flags.view(-1)

            if not np.any(flat_flags):
                return [torch.FloatTensor([]) for _ in range(bsize)]

            # number of potential detections per batch
            item_size = np.prod(flags.shape[1:])
            slices = [slice((item_size * i), (item_size * (i + 1)))
                      for i in range(bsize)]
            # number of detections per batch (prepended with a zero)
            n_dets = torch.stack(
                [flat_flags[0].long() * 0] + [flat_flags[sl].long().sum() for sl in slices])
            # indices of splits between filtered detections
            filtered_split_idxs = torch.cumsum(n_dets, dim=0)

            # Do actual filtering of detections by confidence thresh
            flat_coords = output_.transpose(2, 3)[..., 0:4].clone().view(-1, 4)
            flat_class_max = cls_max.view(-1)
            flat_class_idx = cls_max_idx.view(-1)

            coords = flat_coords[flat_flags]
            scores = flat_class_max[flat_flags]
            cls_idxs = flat_class_idx[flat_flags]

            filtered_dets = torch.cat([coords, scores[:, None],
                                       cls_idxs[:, None].float()], dim=1)

            boxes2 = []
            for lx, rx in zip(filtered_split_idxs, filtered_split_idxs[1:]):
                batch_box = filtered_dets[lx:rx]
                boxes2.append(batch_box)

            boxes = boxes2

        return boxes

    @profiler.profile
    def _nms(self, boxes, mode=0):
        """ Non maximum suppression.
        Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
          boxes (tensor): Bounding boxes from get_detections

        Return:
          (tensor): Pruned boxes

        CommandLine:
            python -m netharn.models.yolo2.light_postproc GetBoundingBoxes._nms --profile

        Examples:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.01, nms_thresh=0.5)
            >>> output = torch.randn(8, 5, 5 + 20, 9, 9)
            >>> boxes_ = self._get_boxes(output.data)
            >>> boxes = torch.Tensor(boxes_[0])
            >>> ans0 = self._nms(boxes, mode=0)
            >>> ans1 = self._nms(boxes, mode=1)
            >>> ans2 = self._nms(boxes, mode=2)

        Ignore:
            >>> from netharn import util
            >>> scores = boxes[..., 4:5]
            >>> classes = boxes[..., 5:6]
            >>> cxywh = util.Boxes(boxes[..., 0:4], 'cxywh')
            >>> tlbr = cxywh.to_tlbr()
            >>> util.non_max_supression(tlbr.data.numpy(), scores.numpy().ravel(), self.nms_thresh)

        Benchmark:
            boxes = torch.Tensor(boxes_[0])
            import ubelt
            for timer in ubelt.Timerit(100, bestof=10, label='nms0+cpu'):
                with timer:
                    self._nms(boxes, mode=0)

            for timer in ubelt.Timerit(100, bestof=10, label='nms1+cpu'):
                with timer:
                    self._nms(boxes, mode=1)

            boxes = boxes.cuda()
            import ubelt
            for timer in ubelt.Timerit(100, bestof=10, label='nms0+gpu'):
                with timer:
                    self._nms(boxes, mode=0)

            for timer in ubelt.Timerit(100, bestof=10, label='nms1+gpu'):
                with timer:
                    self._nms(boxes, mode=1)
        """
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        # convert to tlbr
        tlbr_tensor = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]

        if mode == 0:
            # if torch.cuda.is_available:
            #     boxes = boxes.cuda()
            keep = _nms_torch(tlbr_tensor, scores, nms_thresh=self.nms_thresh)
            keep = sorted(keep)
        elif mode == 1:
            # Dont group by classes, just NMS
            tlbr_np = tlbr_tensor.cpu().numpy().astype(np.float32)
            scores_np = scores.cpu().numpy().astype(np.float32)
            keep = util.non_max_supression(tlbr_np, scores_np,
                                           self.nms_thresh)
            keep = sorted(keep)
        elif mode == 2:
            # Group and use NMS
            tlbr_np = tlbr_tensor.cpu().numpy().astype(np.float32)
            scores_np = scores.cpu().numpy().astype(np.float32)
            classes_np = boxes[..., 5].cpu().numpy().astype(np.int)
            keep = []
            for idxs in ub.group_items(range(len(classes_np)), classes_np).values():
                cls_tlbr_np = tlbr_np.take(idxs, axis=0)
                cls_scores_np = scores_np.take(idxs, axis=0)
                cls_keep = util.non_max_supression(cls_tlbr_np, cls_scores_np,
                                                   self.nms_thresh)
                keep.extend(list(ub.take(idxs, cls_keep)))
            keep = sorted(keep)
        else:
            raise KeyError(mode)
        return boxes[torch.LongTensor(keep)]


def _nms_torch(tlbr_tensor, scores, nms_thresh=.5):
    x1 = tlbr_tensor[:, 0]
    y1 = tlbr_tensor[:, 1]
    x2 = tlbr_tensor[:, 2]
    y2 = tlbr_tensor[:, 3]

    areas = ((x2 - x1) * (y2 - y1))
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            if torch.__version__.startswith('0.3'):
                i = order[0]
            else:
                i = order.item()
            i = order.item()
            keep.append(i)
            break

        i = order[0].item()
        keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (iou <= nms_thresh).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return keep


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.models.yolo2.light_postproc all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
