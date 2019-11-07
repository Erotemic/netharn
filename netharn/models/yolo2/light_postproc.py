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
    def __call__(self, network_output, nms_mode=4):
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
            >>> output = output.to(0)
            >>> for timer in ubelt.Timerit(21, bestof=3, label='mode0+gpu'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         self(output_, nms_mode=0)

        Script:
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> import ubelt
            >>> output = torch.randn(16, 5, 5 + 20, 9, 9)
            >>> output = output.to(0)
            >>> for timer in ubelt.Timerit(21, bestof=3, label='mode1+gpu'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         self(output_, nms_mode=1)

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
            >>>         self(output_, nms_mode=0)
            >>> #
            >>> for timer in ubelt.Timerit(21, bestof=3, label='mode1+cpu'):
            >>>     output_ = output.clone()
            >>>     with timer:
            >>>         self(output_, nms_mode=1)
            >>> #
            >>> if torch.cuda.is_available():
            >>>     output = output.to(0)
            >>>     for timer in ubelt.Timerit(21, bestof=3, label='mode0+gpu'):
            >>>         output_ = output.clone()
            >>>         with timer:
            >>>             self(output_, nms_mode=0)
            >>>     #
            >>>     for timer in ubelt.Timerit(21, bestof=3, label='mode1+gpu'):
            >>>         output_ = output.clone()
            >>>         with timer:
            >>>             self(output_, nms_mode=1)

            %timeit self(output.data, nms_mode=0)
            %timeit self(output.data, nms_mode=1)
            %timeit self(output.data, nms_mode=2)
        """
        # boxes = self._get_boxes(network_output.data, nms_mode=nms_mode)
        # mode1 is the same as 0, just lots faster
        boxes = self._get_boxes(network_output.data)
        boxes = [self._nms(box, nms_mode=nms_mode) for box in boxes]

        # force all boxes to be inside the image
        # boxes = [self._clip_boxes(box) for box in boxes]
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
    def _get_boxes(self, output):
        """
        Returns array of detections for every image in batch

        CommandLine:
            python ~/code/netharn/netharn/box_models/yolo2/light_postproc.py GetBoundingBoxes._get_boxes

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
        """
        # dont modify inplace
        output = output.clone()

        # Variables
        bsize = output.shape[0]
        h, w = output.shape[-2:]

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
        lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
        anchor_w = self.anchors[:, 0].contiguous().view(1, self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(1, self.num_anchors, 1)

        lin_x = lin_x.to(output.device)
        lin_y = lin_y.to(output.device)
        anchor_w = anchor_w.to(output.device)
        anchor_h = anchor_h.to(output.device)

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

        # Newst lightnet code, which is based on my mode1 code
        score_thresh = cls_max > self.conf_thresh
        score_thresh_flat = score_thresh.view(-1)

        if score_thresh.sum() == 0:
            boxes = []
            for i in range(bsize):
                boxes.append(torch.Tensor([]))
            return boxes

        # Mask select boxes > conf_thresh
        coords = output_.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]
        detections = torch.cat([coords, scores[:, None], idx[:, None].float()], dim=1)

        # Get indexes of splits between images of batch
        max_det_per_batch = len(self.anchors) * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(bsize)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        boxes = []
        start = 0
        for end in split_idx:
            boxes.append(detections[start: end])
            start = end

        return boxes

    @profiler.profile
    def _nms(self, cxywh_score_cls, nms_mode=4):
        """ Non maximum suppression.
        Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        Args:
          cxywh_score_cls (tensor): Bounding boxes and scores from
              get_detections. Assumes columns 0:4 are cx, cy, w, h, Column 4 is
              confidence, and column 5 is class id.

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
            >>> ans0 = self._nms(boxes, nms_mode=0)
            >>> ans1 = self._nms(boxes, nms_mode=1)
            >>> ans2 = self._nms(boxes, nms_mode=2)

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
                    self._nms(boxes, nms_mode=0)

            for timer in ubelt.Timerit(100, bestof=10, label='nms1+cpu'):
                with timer:
                    self._nms(boxes, nms_mode=1)

            boxes = boxes.to()
            import ubelt
            for timer in ubelt.Timerit(100, bestof=10, label='nms0+gpu'):
                with timer:
                    self._nms(boxes, nms_mode=0)

            for timer in ubelt.Timerit(100, bestof=10, label='nms1+gpu'):
                with timer:
                    self._nms(boxes, nms_mode=1)
        """
        if cxywh_score_cls.numel() == 0:
            return cxywh_score_cls

        a = cxywh_score_cls[:, :2]
        b = cxywh_score_cls[:, 2:4]
        # convert to tlbr
        tlbr_tensor = torch.cat([a - b / 2, a + b / 2], 1)
        scores = cxywh_score_cls[:, 4]

        if nms_mode == 0:
            # if torch.cuda.is_available:
            #     boxes = boxes.to(0)
            from netharn.util.nms.torch_nms import torch_nms
            cls_tensor = cxywh_score_cls[:, 5]
            keep = torch_nms(tlbr_tensor, scores, classes=cls_tensor,
                             thresh=self.nms_thresh, bias=0)
            return cxywh_score_cls[keep]
            # keep = _nms_torch(tlbr_tensor, scores, nms_thresh=self.nms_thresh)
            # keep = sorted(keep)
        elif nms_mode == 1:
            # Dont group by classes, just NMS
            tlbr_np = tlbr_tensor.cpu().numpy().astype(np.float32)
            scores_np = scores.cpu().numpy().astype(np.float32)
            keep = util.non_max_supression(tlbr_np, scores_np, self.nms_thresh,
                                           bias=0)
            keep = sorted(keep)
        elif nms_mode == 2:
            # Group and use NMS
            tlbr_np = tlbr_tensor.cpu().numpy().astype(np.float32)
            scores_np = scores.cpu().numpy().astype(np.float32)
            classes_np = cxywh_score_cls[:, 5].cpu().numpy().astype(np.int)

            keep = util.non_max_supression(tlbr_np, scores_np, self.nms_thresh,
                                           classes=classes_np, bias=0)
            # keep = []
            # for idxs in ub.group_items(range(len(classes_np)), classes_np).values():
            #     cls_tlbr_np = tlbr_np.take(idxs, axis=0)
            #     cls_scores_np = scores_np.take(idxs, axis=0)
            #     cls_keep = util.non_max_supression(cls_tlbr_np, cls_scores_np,
            #                                        self.nms_thresh, bias=0)
            #     keep.extend(list(ub.take(idxs, cls_keep)))
            keep = sorted(keep)
        elif nms_mode == 3:
            # Group and use NMS
            classes_np = cxywh_score_cls[:, 5].cpu().numpy().astype(np.int)
            keep = util.non_max_supression(tlbr_tensor, scores,
                                           self.nms_thresh, classes=classes_np,
                                           bias=0, impl='torch')
            keep = sorted(keep)
        elif nms_mode == 4:
            # Dont group, but use torch
            from netharn.util.nms.torch_nms import torch_nms
            keep = torch_nms(tlbr_tensor, scores,
                             thresh=self.nms_thresh, bias=0)
            return cxywh_score_cls[keep]
        else:
            raise KeyError(nms_mode)
        return cxywh_score_cls[torch.LongTensor(keep)]


def benchmark_nms_version():
    """
        xdoctset netharn.models.yolo2.light_postproc benchmark_nms_version
    """
    # Build random test boxes and scores
    from lightnet.data.transform._postprocess import NonMaxSupression
    import netharn as nh
    num = 16 * 16 * 5
    rng = nh.util.ensure_rng(0)
    cpu_boxes = nh.util.Boxes.random(num, scale=416.0, rng=rng, format='tlbr', tensor=True)
    cpu_tlbr = cpu_boxes.to_tlbr().data
    # cpu_scores = torch.Tensor(rng.rand(len(cpu_tlbr)))
    # make all scores unique to ensure comparability
    cpu_scores = torch.Tensor(np.linspace(0, 1, len(cpu_tlbr)))
    cpu_cls = torch.LongTensor(rng.randint(0, 20, len(cpu_tlbr)))

    # Format boxes in lightnet format
    cxywh_score_cls = torch.cat([cpu_boxes.to_cxywh().data,
                                 cpu_scores[:, None],
                                 cpu_cls.float()[:, None]], dim=-1)

    gpu = torch.device('cuda', 0)
    gpu_ln_boxes = cxywh_score_cls.to(gpu)

    thresh = .5

    def _ln_output_to_keep(ln_output, ln_boxes):
        keep = []
        for row in ln_output:
            # Find the index that we kept
            idxs = np.where(np.all(np.isclose(ln_boxes, row), axis=1))[0]
            assert len(idxs) == 1
            keep.append(idxs[0])
        assert np.all(np.isclose(ln_boxes[keep], ln_output))
        return keep

    N = 12
    bestof = 3

    t1 = ub.Timerit(N, bestof=bestof, label='lightnet()')
    for timer in t1:
        with timer:
            ln_output = NonMaxSupression._nms(gpu_ln_boxes, nms_thresh=thresh,
                                              class_nms=True, fast=False)
            torch.cuda.synchronize()
    ln_keep = _ln_output_to_keep(ln_output, gpu_ln_boxes)

    anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
    self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.01, nms_thresh=thresh)

    t1 = ub.Timerit(N, bestof=bestof, label='netharn(mode0)')
    for timer in t1:
        with timer:
            nh_output = self._nms(gpu_ln_boxes, nms_mode=0)
            torch.cuda.synchronize()
    nh_keep_0 = _ln_output_to_keep(nh_output, gpu_ln_boxes)

    t1 = ub.Timerit(N, bestof=bestof, label='netharn(mode1)')
    for timer in t1:
        with timer:
            nh_output = self._nms(gpu_ln_boxes, nms_mode=1)
            torch.cuda.synchronize()
    nh_keep_1 = _ln_output_to_keep(nh_output, gpu_ln_boxes)

    t1 = ub.Timerit(N, bestof=bestof, label='netharn(mode2)')
    for timer in t1:
        with timer:
            nh_output = self._nms(gpu_ln_boxes, nms_mode=2)
            torch.cuda.synchronize()
    nh_keep_2 = _ln_output_to_keep(nh_output, gpu_ln_boxes)

    t1 = ub.Timerit(N, bestof=bestof, label='netharn(mode3)')
    for timer in t1:
        with timer:
            nh_output = self._nms(gpu_ln_boxes, nms_mode=3)
            torch.cuda.synchronize()
    nh_keep_3 = _ln_output_to_keep(nh_output, gpu_ln_boxes)

    t1 = ub.Timerit(N, bestof=bestof, label='netharn(mode4)')
    for timer in t1:
        with timer:
            nh_output = self._nms(gpu_ln_boxes, nms_mode=4)
            torch.cuda.synchronize()
    nh_keep_4 = _ln_output_to_keep(nh_output, gpu_ln_boxes)

    nh_keep_0 == nh_keep_2
    nh_keep_0 == nh_keep_3

    print('ln_keep = {!r}'.format(len(ln_keep)))
    print('len(nh_keep_0) = {!r}'.format(len(nh_keep_0)))
    print('len(nh_keep_1) = {!r}'.format(len(nh_keep_1)))
    print('len(nh_keep_2) = {!r}'.format(len(nh_keep_2)))
    print('len(nh_keep_3) = {!r}'.format(len(nh_keep_3)))
    print('len(nh_keep_4) = {!r}'.format(len(nh_keep_4)))


# def _nms_torch(tlbr_tensor, scores, nms_thresh=.5):
#     x1 = tlbr_tensor[:, 0]
#     y1 = tlbr_tensor[:, 1]
#     x2 = tlbr_tensor[:, 2]
#     y2 = tlbr_tensor[:, 3]

#     areas = ((x2 - x1) * (y2 - y1))
#     _, order = scores.sort(0, descending=True)

#     keep = []
#     while order.numel() > 0:
#         if order.numel() == 1:
#             if torch.__version__.startswith('0.3'):
#                 i = order[0]
#             else:
#                 i = order.item()
#             i = order.item()
#             keep.append(i)
#             break

#         i = order[0].item()
#         keep.append(i)

#         xx1 = x1[order[1:]].clamp(min=x1[i])
#         yy1 = y1[order[1:]].clamp(min=y1[i])
#         xx2 = x2[order[1:]].clamp(max=x2[i])
#         yy2 = y2[order[1:]].clamp(max=y2[i])

#         w = (xx2 - xx1).clamp(min=0)
#         h = (yy2 - yy1).clamp(min=0)
#         inter = w * h

#         iou = inter / (areas[i] + areas[order[1:]] - inter)

#         ids = (iou <= nms_thresh).nonzero().squeeze()
#         if ids.numel() == 0:
#             break
#         order = order[ids + 1]
#     return keep


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.models.yolo2.light_postproc all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
