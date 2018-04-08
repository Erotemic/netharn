#
#   Loss modules
#   Copyright EAVISE
#
"""
Speedups
    [ ] - Preinitialize anchor tensors

"""

import math
import torch
import torch.nn as nn
import numpy as np  # NOQA
import functools
from torch.autograd import Variable
from netharn import util
from netharn.util import profiler

__all__ = ['RegionLoss']


# def math.log(x):
#     return math.log(max(x, np.finfo(np.float).eps))

@profiler.profile
def benchmark_region_loss():
    """
    CommandLine:
        python ~/code/netharn/netharn/models/yolo2/light_region_loss.py benchmark_region_loss --profile

    Benchmark:
        >>> benchmark_region_loss()
    """
    from netharn.models.yolo2.light_yolo import Yolo
    torch.random.manual_seed(0)
    network = Yolo(num_classes=2, conf_thresh=4e-2)
    self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
    Win, Hin = 96, 96
    # true boxes for each item in the batch
    # each box encodes class, center, width, and height
    # coordinates are normalized in the range 0 to 1
    # items in each batch are padded with dummy boxes with class_id=-1
    target = torch.FloatTensor([
        # boxes for batch item 1
        [[0, 0.50, 0.50, 1.00, 1.00],
         [1, 0.32, 0.42, 0.22, 0.12]],
        # boxes for batch item 2 (it has no objects, note the pad!)
        [[-1, 0, 0, 0, 0],
         [-1, 0, 0, 0, 0]],
    ])
    im_data = torch.randn(len(target), 3, Hin, Win)
    output = network.forward(im_data)
    import ubelt
    for timer in ubelt.Timerit(250, bestof=10, label='time'):
        with timer:
            loss = float(self.forward(output, target))
    print('loss = {!r}'.format(loss))


@profiler.profile
def profile_loss_speed():
    """
    python ~/code/netharn/netharn/models/yolo2/light_region_loss.py profile_loss_speed --profile

    Benchmark:
        >>> profile_loss_speed()
    """
    from netharn.models.yolo2.light_yolo import Yolo
    import netharn.models.yolo2.light_region_loss
    import lightnet.network
    import netharn as nh

    rng = util.ensure_rng(0)
    torch.random.manual_seed(0)
    network = Yolo(num_classes=2, conf_thresh=4e-2)

    self1 = netharn.models.yolo2.light_region_loss.RegionLoss(
        num_classes=network.num_classes, anchors=network.anchors)
    self2 = lightnet.network.RegionLoss(num_classes=network.num_classes,
                                        anchors=network.anchors)

    bsize = 8
    # Make a random semi-realistic set of groundtruth items
    n_targets = [rng.randint(0, 20) for _ in range(bsize)]
    target_list = [torch.FloatTensor(
        np.hstack([rng.randint(0, network.num_classes, nT)[:, None],
                   util.Boxes.random(nT, scale=1.0, rng=rng).data]))
        for nT in n_targets]
    target = nh.data.collate.padded_collate(target_list)

    Win, Hin = 416, 416
    im_data = torch.randn(len(target), 3, Hin, Win)
    output = network.forward(im_data)

    loss1 = float(self1(output, target))
    loss2 = float(self2(output, target))
    print('loss1 = {!r}'.format(loss1))
    print('loss2 = {!r}'.format(loss2))


def compare_loss_speed():
    """
    python ~/code/netharn/netharn/models/yolo2/light_region_loss.py compare_loss_speed

    Example:
        >>> compare_loss_speed()
    """
    from netharn.models.yolo2.light_yolo import Yolo
    import netharn.models.yolo2.light_region_loss
    import lightnet.network
    import ubelt as ub
    torch.random.manual_seed(0)
    network = Yolo(num_classes=2, conf_thresh=4e-2)

    self1 = netharn.models.yolo2.light_region_loss.RegionLoss(
        num_classes=network.num_classes, anchors=network.anchors)
    self2 = lightnet.network.RegionLoss(num_classes=network.num_classes,
                                        anchors=network.anchors)

    # Win, Hin = 416, 416
    Win, Hin = 96, 96

    # ----- More targets -----
    rng = util.ensure_rng(0)
    import netharn as nh

    bsize = 4
    # Make a random semi-realistic set of groundtruth items
    n_targets = [rng.randint(0, 10) for _ in range(bsize)]
    target_list = [torch.FloatTensor(
        np.hstack([rng.randint(0, network.num_classes, nT)[:, None],
                   util.Boxes.random(nT, scale=1.0, rng=rng).data]))
        for nT in n_targets]
    target = nh.data.collate.padded_collate(target_list)

    im_data = torch.randn(len(target), 3, Hin, Win)
    output = network.forward(im_data)

    self1.iou_mode = 'c'
    for timer in ub.Timerit(100, bestof=10, label='cython_ious'):
        with timer:
            loss_cy = float(self1(output, target))

    self1.iou_mode = 'py'
    for timer in ub.Timerit(100, bestof=10, label='python_ious'):
        with timer:
            loss_py = float(self1(output, target))

    for timer in ub.Timerit(100, bestof=10, label='original'):
        with timer:
            loss_orig = float(self2(output, target))

    print('loss_cy   = {!r}'.format(loss_cy))
    print('loss_py   = {!r}'.format(loss_py))
    print('loss_orig = {!r}'.format(loss_orig))


class BaseLossWithCudaState(torch.nn.modules.loss._Loss):
    """
    Keep track of if the module is in cpu or gpu mode
    """
    def __init__(self):
        super(BaseLossWithCudaState, self).__init__()
        self._iscuda = False
        self._device_num = None

    def cuda(self, device_num=None, **kwargs):
        self._iscuda = True
        self._device_num = device_num
        return super(BaseLossWithCudaState, self).cuda(device_num, **kwargs)

    def cpu(self):
        self._iscuda = False
        self._device_num = None
        return super(BaseLossWithCudaState, self).cpu()

    @property
    def is_cuda(self):
        return self._iscuda

    def get_device(self):
        return self._device_num


class RegionLoss(BaseLossWithCudaState):
    """ Computes region loss from darknet network output and target annotation.

    Args:
        network (lightnet.network.Darknet): Network that will be optimised with this loss function (optional) if not specified, then `num_classes` and `anchors` must be given.
        num_classes (int): number of categories
        anchors (dict): dict representing anchor boxes (see :class:`lightnet.network.Darknet`)
        coord_scale (float): weight of bounding box coordinates
        noobject_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou for a predicted box to be assigned to a target

    CommandLine:
        python ~/code/netharn/netharn/models/yolo2/light_region_loss.py RegionLoss

    Example:
        >>> from netharn.models.yolo2.light_yolo import Yolo
        >>> torch.random.manual_seed(0)
        >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
        >>> self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
        >>> Win, Hin = 96, 96
        >>> Wout, Hout = 1, 1
        >>> # true boxes for each item in the batch
        >>> # each box encodes class, center, width, and height
        >>> # coordinates are normalized in the range 0 to 1
        >>> # items in each batch are padded with dummy boxes with class_id=-1
        >>> target = torch.FloatTensor([
        >>>     # boxes for batch item 1
        >>>     [[0, 0.50, 0.50, 1.00, 1.00],
        >>>      [1, 0.32, 0.42, 0.22, 0.12]],
        >>>     # boxes for batch item 2 (it has no objects, note the pad!)
        >>>     [[-1, 0, 0, 0, 0],
        >>>      [-1, 0, 0, 0, 0]],
        >>> ])
        >>> im_data = torch.randn(len(target), 3, Hin, Win)
        >>> output = network.forward(im_data)
        >>> loss = float(self.forward(output, target))
        >>> print(f'output.sum() = {output.sum():.2f}')
        output.sum() = 2.15
        >>> print(f'loss = {loss:.2f}')
        loss = 20.18

    """

    def __init__(self, num_classes, anchors, coord_scale=1.0,
                 noobject_scale=1.0, object_scale=5.0, class_scale=1.0,
                 thresh=0.6):
        super().__init__()

        self.num_classes = num_classes

        self.anchors = anchors
        self.num_anchors = len(anchors)

        # self.anchor_step = len(self.anchors) // self.num_anchors
        self.reduction = 32             # input_dim/output_dim

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

        self.loss_coord = None
        self.loss_conf = None
        self.loss_cls = None
        self.loss_tot = None

        self.mse = nn.MSELoss(size_average=False)
        self.mse = nn.MSELoss(size_average=False)
        self.cls_critrion = nn.CrossEntropyLoss(size_average=False)

        nA = self.num_anchors
        self.anchor_w = torch.Tensor(self.anchors.T[0]).view(nA, 1)
        self.anchor_h = torch.Tensor(self.anchors.T[1]).view(nA, 1)

        rel_anchors_cxywh = util.Boxes(np.hstack([self.anchors * 0, self.anchors]).astype(np.float32), 'cxywh')
        self.rel_anchors_tlbr = rel_anchors_cxywh.toformat('tlbr').data

        self._prev_pred_init = None
        self._prev_pred_dim = None

        self.iou_mode = None

    @functools.lru_cache(maxsize=32)
    @profiler.profile
    def _init_pred_boxes(self, device, nB, nA, nH, nW):
        # NOTE: this might not actually be a bottleneck
        # I haven't tested.
        # pred_dim = nB * nA * nH * nW
        # if pred_dim == self._prev_pred_dim:
        #     pred_boxes, lin_x, lin_y = self._prev_pred_init
        # else:
        pred_boxes = torch.FloatTensor(nB * nA * nH * nW, 4)
        lin_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).view(nH * nW)
        lin_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().contiguous().view(nH * nW)
        if device is not None:
            lin_x = lin_x.cuda(device)
            lin_y = lin_y.cuda(device)
            self.anchor_w = self.anchor_w.cuda(device)
            self.anchor_h = self.anchor_h.cuda(device)
        return pred_boxes, lin_x, lin_y
        # self._prev_pred_init = pred_boxes, lin_x, lin_y
        # return pred_boxes, lin_x, lin_y

    @profiler.profile
    def forward(self, output, target, seen=0):
        """ Compute Region loss.

        Args:
            output (torch.autograd.Variable): Output from the network
            target (torch.Tensor): the shape should be [B, T, 5], where B is
                the batch size, T is the maximum number of boxes in an item,
                and the final dimension should correspond to [class_idx,
                center_x, center_y, width, height]. Items with fewer than T
                boxes should be padded with dummy boxes with class_idx=-1.
            seen (int): if specified, overrides the `seen` attribute read from `self.net` (default None)
        """
        # Parameters
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        if isinstance(target, Variable):
            target = target.data

        # Get x,y,w,h,conf,cls
        output = output.view(nB, nA, -1, nH * nW)
        # torch.cuda.synchronize()

        coord = torch.zeros_like(output[:, :, :4])
        # torch.cuda.synchronize()

        coord[:, :, 0:2] = output[:, :, 0:2].sigmoid()  # tx,ty
        # torch.cuda.synchronize()

        coord[:, :, 2:4] = output[:, :, 2:4]            # tw,th
        # torch.cuda.synchronize()

        conf = output[:, :, 4].sigmoid()
        # torch.cuda.synchronize()
        if nC > 1:
            cls = output[:, :, 5:].contiguous().view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(-1, nC)
            # torch.cuda.synchronize()

        # Create prediction boxes
        with torch.no_grad():
            device = self.get_device()
            pred_boxes, lin_x, lin_y = self._init_pred_boxes(device, nB, nA, nH, nW)
            # torch.cuda.synchronize()
            if device is not None:
                pred_boxes = pred_boxes.cuda(device)
                # torch.cuda.synchronize()
                self.anchor_w = self.anchor_w.cuda(device)
                # torch.cuda.synchronize()
                self.anchor_h = self.anchor_h.cuda(device)
                # torch.cuda.synchronize()

            pred_boxes[:, 0] = (coord[:, :, 0].data + lin_x).view(-1)
            torch.cuda.synchronize()
            pred_boxes[:, 1] = (coord[:, :, 1].data + lin_y).view(-1)
            torch.cuda.synchronize()
            pred_boxes[:, 2] = (coord[:, :, 2].data.exp() * self.anchor_w).view(-1)
            torch.cuda.synchronize()
            pred_boxes[:, 3] = (coord[:, :, 3].data.exp() * self.anchor_h).view(-1)
            torch.cuda.synchronize()
            pred_boxes = pred_boxes.cpu()

            # Get target values
            coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(
                pred_boxes, target, nH, nW, seen=seen)
            coord_mask = coord_mask.expand_as(tcoord)
            if nC > 1:
                tcls = tcls.view(-1)[cls_mask.view(-1)].long()
                cls_mask = cls_mask.view(-1, 1).repeat(1, nC)

            if device is not None:
                tcoord = tcoord.cuda(device)
                tconf = tconf.cuda(device)
                coord_mask = coord_mask.cuda(device)
                conf_mask = conf_mask.cuda(device)

                if nC > 1:
                    tcls = tcls.cuda(device)
                    cls_mask = cls_mask.cuda(device)

        tcoord = Variable(tcoord, requires_grad=False)
        tconf = Variable(tconf, requires_grad=False)
        coord_mask = Variable(coord_mask, requires_grad=False)
        conf_mask = Variable(conf_mask.sqrt(), requires_grad=False)
        if nC > 1:
            tcls = Variable(tcls, requires_grad=False)
            cls_mask = Variable(cls_mask, requires_grad=False)
            cls = cls[cls_mask].view(-1, nC)

        # Compute losses
        loss_coord = self.coord_scale * self.mse(coord * coord_mask, tcoord * coord_mask) / nB
        loss_conf = self.mse(conf * conf_mask, tconf * conf_mask) / nB
        if nC > 1:
            loss_cls = self.class_scale * 2 * self.cls_critrion(cls, tcls) / nB
            loss_tot = loss_coord + loss_conf + loss_cls
            self.loss_cls = float(loss_cls.data.cpu().numpy())
        else:
            self.loss_cls = 0
            loss_tot = loss_coord + loss_conf

        self.loss_tot = float(loss_tot.data.cpu().numpy())
        self.loss_coord = float(loss_coord.data.cpu().numpy())
        self.loss_conf = float(loss_conf.data.cpu().numpy())

        return loss_tot

    def build_targets(self, pred_boxes, ground_truth, nH, nW, seen=0):
        """ Compare prediction boxes and targets, convert targets to network output tensors """
        return self._build_targets_tensor(pred_boxes, ground_truth, nH, nW, seen=seen)

    @profiler.profile
    def _build_targets_tensor(self, pred_boxes, ground_truth, nH, nW, seen=0):
        """
        Compare prediction boxes and ground truths, convert ground truths to network output tensors

        Example:
            >>> from netharn.models.yolo2.light_yolo import Yolo
            >>> from netharn.models.yolo2.light_region_loss import RegionLoss
            >>> torch.random.manual_seed(0)
            >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
            >>> self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
            >>> Win, Hin = 96, 96
            >>> nW, nH = 3, 3
            >>> # true boxes for each item in the batch
            >>> # each box encodes class, center, width, and height
            >>> # coordinates are normalized in the range 0 to 1
            >>> # items in each batch are padded with dummy boxes with class_id=-1
            >>> ground_truth = torch.FloatTensor([
            >>>     # boxes for batch item 0 (it has no objects, note the pad!)
            >>>     [[-1, 0, 0, 0, 0],
            >>>      [-1, 0, 0, 0, 0],
            >>>      [-1, 0, 0, 0, 0]],
            >>>     # boxes for batch item 1
            >>>     [[0, 0.50, 0.50, 1.00, 1.00],
            >>>      [1, 0.34, 0.32, 0.12, 0.32],
            >>>      [1, 0.32, 0.42, 0.22, 0.12]],
            >>> ])
            >>> pred_boxes = torch.rand(90, 4)
            >>> seen = 0
        """
        # Parameters
        nB = ground_truth.size(0)
        nT = ground_truth.size(1)
        nA = self.num_anchors
        nAnchors = nA * nH * nW
        nPixels = nH * nW

        seen = seen + nB

        # Tensors
        conf_mask = torch.ones(nB, nA, nPixels) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, 1, nPixels)
        cls_mask = torch.zeros(nB, nA, nPixels).byte()
        tcoord = torch.zeros(nB, nA, 4, nPixels)
        tconf = torch.zeros(nB, nA, nPixels)
        tcls = torch.zeros(nB, nA, nPixels)

        if seen < 12800:
            coord_mask.fill_(1)
            tcoord[:, :, 0].fill_(0.5)
            tcoord[:, :, 1].fill_(0.5)

        pred_cxywh = pred_boxes
        pred_tlbr = util.Boxes(pred_cxywh.data.cpu().numpy(), 'cxywh').toformat('tlbr').data

        gt_class = ground_truth[..., 0].data.cpu().numpy()
        gt_cxywh = util.Boxes(ground_truth[..., 1:5].data.cpu().numpy().astype(np.float32), 'cxywh').scale([nW, nH])

        gt_tlbr = gt_cxywh.to_tlbr().data

        rel_gt_cxywh = gt_cxywh.copy()
        rel_gt_cxywh.data.T[0:2] = 0

        rel_gt_tlbr = rel_gt_cxywh.toformat('tlbr').data

        gt_isvalid = (gt_class >= 0)

        # Loop over ground_truths and construct tensors
        for bx in range(nB):
            # Get the actual groundtruth boxes for this batch item
            flags = gt_isvalid[bx]
            if not np.any(flags):
                continue

            # Create gt anchor assignments
            batch_rel_gt_tlbr = rel_gt_tlbr[bx][flags]
            anchor_ious = util.box_ious(self.rel_anchors_tlbr,
                                        batch_rel_gt_tlbr, bias=0,
                                        mode=self.iou_mode)
            best_ns = np.argmax(anchor_ious, axis=0)

            # Setting confidence mask
            cur_pred_tlbr = pred_tlbr[bx * nAnchors:(bx + 1) * nAnchors]
            cur_gt_tlbr = gt_tlbr[bx][flags]

            ious = util.box_ious(cur_pred_tlbr, cur_gt_tlbr, bias=0,
                                 mode=self.iou_mode)
            cur_ious = torch.FloatTensor(ious.max(-1))
            conf_mask[bx].view(-1)[cur_ious > self.thresh] = 0

            for t in range(nT):
                if not flags[t]:
                    break
                gx, gy, gw, gh = gt_cxywh.data[bx][t]
                gi = min(nW - 1, max(0, int(gx)))
                gj = min(nH - 1, max(0, int(gy)))

                best_n = best_ns[t]

                gt_box_ = gt_tlbr[bx][t]
                pred_box_ = pred_tlbr[bx * nAnchors + best_n * nPixels + gj * nW + gi]

                iou = float(util.box_ious(gt_box_[None, :], pred_box_[None, :],
                                          bias=0, mode=self.iou_mode)[0, 0])

                best_anchor = self.anchors[best_n]
                best_aw, best_ah = best_anchor

                coord_mask[bx, best_n, 0, gj * nW + gi] = 1
                cls_mask[bx, best_n, gj * nW + gi] = 1
                conf_mask[bx, best_n, gj * nW + gi] = self.object_scale

                tcoord[bx, best_n, 0, gj * nW + gi] = gx - gi
                tcoord[bx, best_n, 1, gj * nW + gi] = gy - gj
                tcoord[bx, best_n, 2, gj * nW + gi] = math.log(gw / best_aw)
                tcoord[bx, best_n, 3, gj * nW + gi] = math.log(gh / best_ah)
                tconf[bx, best_n, gj * nW + gi] = iou
                tcls[bx, best_n, gj * nW + gi] = ground_truth[bx, t, 0]

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.models.yolo2.light_region_loss all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
