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
# import functools
from torch.autograd import Variable
from netharn import util
from netharn.util import profiler

__all__ = ['RegionLoss']


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
        if self._device_num is None:
            return torch.device('cpu')
        return self._device_num


class RegionLoss(BaseLossWithCudaState):
    """ Computes region loss from darknet network output and target annotation.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
            These width and height values should be in network output coordinates.
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
        >>> print(f'loss = {loss:.2f}')
        loss = 20.18
        >>> print(f'output.sum() = {output.sum():.2f}')
        output.sum() = 2.15

    Example:
        >>> from netharn.models.yolo2.light_yolo import Yolo
        >>> torch.random.manual_seed(0)
        >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
        >>> self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
        >>> Win, Hin = 96, 96
        >>> Wout, Hout = 1, 1
        >>> target = torch.FloatTensor([])
        >>> im_data = torch.randn(2, 3, Hin, Win)
        >>> output = network.forward(im_data)
        >>> loss = float(self.forward(output, target))
        >>> print(f'output.sum() = {output.sum():.2f}')
        output.sum() = 2.15
        >>> print(f'loss = {loss:.2f}')
        loss = 16.47
    """

    def __init__(self, num_classes, anchors, coord_scale=1.0,
                 noobject_scale=1.0, object_scale=5.0, class_scale=1.0,
                 thresh=0.6):
        super().__init__()

        self.num_classes = num_classes

        self.anchors = torch.Tensor(anchors)
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

        # Precompute relative anchors in tlbr format for iou computation
        rel_anchors_cxywh = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
        self.rel_anchors_boxes = util.Boxes(rel_anchors_cxywh, 'cxywh')

        self._prev_pred_init = None
        self._prev_pred_dim = None

        self.iou_mode = None

    def _output_to_boxes(self, output):
        """
        Returns boxes in normalized cxywh space and logspace

        Maintains dimensions as much as possible

        Args:
            output (tensor): with shape [nB, nA, 5 + nC, nH, nW]
        """
        # Parameters
        # output = output.view(nB
        nB, nA, nC5, nH, nW = output.shape
        assert nA == self.num_anchors
        assert nC5 == self.num_classes + 5
        nC = nC5 - 5
        # assert nO == (nA * (5 + nC))

        device = self.get_device()
        self.anchors = self.anchors.to(device)
        anchor_w = self.anchors[:, 0].contiguous().view(1, nA, 1, 1, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(1, nA, 1, 1, 1)

        output_ = output.view(nB, nA, 5 + nC, nH, nW)

        # The 0, 1, 2, and 3-rd output in the 3rd dim are bbox coords
        coord = torch.zeros_like(output_[:, :, :4, :, :], device=device)
        coord[:, :, 0:2, :, :] = output_[:, :, 0:2, :, :].sigmoid()  # tx,ty
        coord[:, :, 2:4, :, :] = output_[:, :, 2:4, :, :]            # tw,th

        # The 4-th output in the 3rd dim is an iou score
        conf = output_[:, :, 4:5, :, :].sigmoid()

        def swap_third_dim(x):
            """ for yolo output makes the 3rd dim the last """
            x = x.view(nB * nA, -1, nH * nW)
            x = x.transpose(1, 2).contiguous()
            x = x.view(nB, nA, nH, nW, -1)
            return x

        # The the rest of the outputs are class probabilities
        if nC > 1:
            cls = swap_third_dim(output_[:, :, 5:, :, :])

        # Add center offsets to each grid cell in the network output.
        lin_x = torch.linspace(
            0, nW - 1, nW, device=device).repeat(nH, 1)
        lin_y = torch.linspace(
            0, nH - 1, nH, device=device).repeat(nW, 1).t().contiguous()

        # Create prediction boxes
        pred_cxywh = torch.FloatTensor(nB * nA * nH * nW, 4, device=device)
        pred_cxywh[:, 0] = (coord[:, :, 0:1, :, :].data + lin_x).view(-1)
        pred_cxywh[:, 1] = (coord[:, :, 1:2, :, :].data + lin_y).view(-1)
        pred_cxywh[:, 2] = (coord[:, :, 2:3, :, :].data.exp() * anchor_w).view(-1)
        pred_cxywh[:, 3] = (coord[:, :, 3:4, :, :].data.exp() * anchor_h).view(-1)

        # conf.view(nB * nA * nW * nH, 1)
        # coord.view(nB * nA, 4, nH * nC)
        # coord2 = swap_third_dim(coord).view(-1, 4)

        return coord, conf, cls, pred_cxywh

    def fw2(self):
        _tup = self.build_targets(
            pred_cxywh, target, nH, nW, seen=seen, gt_weights=gt_weights)
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = _tup

        # Replicate along the third dim
        coord_mask = coord_mask.expand_as(tcoord)

        if nC > 1:
            tcls = tcls.view(-1)[cls_mask.view(-1)].long()
            cls_mask = cls_mask.view(-1, 1).repeat(1, nC)

        tcoord = Variable(tcoord, requires_grad=False)
        tconf = Variable(tconf, requires_grad=False)
        coord_mask = Variable(coord_mask, requires_grad=False)
        conf_mask = Variable(conf_mask.sqrt(), requires_grad=False)

        tconf = tconf.view(nB, nA, 1, nH, nW)
        conf_mask = conf_mask.view(nB, nA, 1, nH, nW)
        coord_mask = coord_mask.view(nB, nA, 4, nH, nW)
        cls_mask = cls_mask.view(nB, nA, nH, nW, nC)  # note the difference

        if nC > 1:
            tcls = Variable(tcls, requires_grad=False)
            cls_mask = Variable(cls_mask, requires_grad=False)
            pcls = cls[cls_mask].view(-1, nC)

        # Compute losses

        # corresponds to delta_region_box
        loss_coord = self.coord_scale * self.mse(coord * coord_mask, tcoord * coord_mask) / nB
        loss_conf = self.mse(conf * conf_mask, tconf * conf_mask) / nB
        if nC > 1 and cls.numel():
            loss_cls = self.class_scale * 2 * self.cls_critrion(pcls, tcls) / nB
            loss_tot = loss_coord + loss_conf + loss_cls
            self.loss_cls = float(loss_cls.data.cpu().numpy())
        else:
            self.loss_cls = 0
            loss_tot = loss_coord + loss_conf

        self.loss_tot = float(loss_tot.data.cpu().numpy())
        self.loss_coord = float(loss_coord.data.cpu().numpy())
        self.loss_conf = float(loss_conf.data.cpu().numpy())
        pass

    @profiler.profile
    def forward(self, output, target, seen=0, gt_weights=None):
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

        coord = torch.zeros_like(output[:, :, :4])

        coord[:, :, 0:2] = output[:, :, 0:2].sigmoid()  # tx,ty
        coord[:, :, 2:4] = output[:, :, 2:4]            # tw,th

        conf = output[:, :, 4].sigmoid()
        if nC > 1:
            cls = output[:, :, 5:].contiguous().view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(-1, nC)

        with torch.no_grad():
            # Create prediction boxes
            pred_cxywh = torch.FloatTensor(nB * nA * nH * nW, 4)
            lin_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).view(nH * nW)
            lin_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().contiguous().view(nH * nW)
            anchor_w = self.anchors[:, 0].contiguous().view(nA, 1)
            anchor_h = self.anchors[:, 1].contiguous().view(nA, 1)

            device = self.get_device()
            if device is not None:
                self.rel_anchors_boxes.data = self.rel_anchors_boxes.data.to(device)
                self.anchors = self.anchors.to(device)
                pred_cxywh = pred_cxywh.to(device)
                lin_x = lin_x.to(device)
                lin_y = lin_y.to(device)
                anchor_w = anchor_w.to(device)
                anchor_h = anchor_h.to(device)

            # Convert raw network output to bounding boxes in network output coordinates
            pred_cxywh[:, 0] = (coord[:, :, 0].data + lin_x).view(-1)
            pred_cxywh[:, 1] = (coord[:, :, 1].data + lin_y).view(-1)
            pred_cxywh[:, 2] = (coord[:, :, 2].data.exp() * anchor_w).view(-1)
            pred_cxywh[:, 3] = (coord[:, :, 3].data.exp() * anchor_h).view(-1)

            # Get target values
            _tup = self.build_targets(
                pred_cxywh, target, nH, nW, seen=seen, gt_weights=gt_weights)
            coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = _tup

            coord_mask = coord_mask.view(*list(coord_mask.shape[0:-2]) + [coord_mask.shape[-2] * coord_mask.shape[-1]])
            conf_mask = conf_mask.view(*list(conf_mask.shape[0:-2]) + [conf_mask.shape[-2] * conf_mask.shape[-1]])
            cls_mask = cls_mask.view(*list(cls_mask.shape[0:-2]) + [cls_mask.shape[-2] * cls_mask.shape[-1]])
            tcoord = tcoord.view(*list(tcoord.shape[0:-2]) + [tcoord.shape[-2] * tcoord.shape[-1]])
            tconf = tconf.view(*list(tconf.shape[0:-2]) + [tconf.shape[-2] * tconf.shape[-1]])
            tcls = tcls.view(*list(tcls.shape[0:-2]) + [tcls.shape[-2] * tcls.shape[-1]])

            coord_mask = coord_mask.expand_as(tcoord)
            if nC > 1:
                tcls = tcls.view(-1)[cls_mask.view(-1)].long()
                cls_mask = cls_mask.view(-1, 1).repeat(1, nC)

            if device is not None:
                tcoord = tcoord.to(device)
                tconf = tconf.to(device)
                coord_mask = coord_mask.to(device)
                conf_mask = conf_mask.to(device)

                if nC > 1:
                    tcls = tcls.to(device)
                    cls_mask = cls_mask.to(device)

        tcoord = Variable(tcoord, requires_grad=False)
        tconf = Variable(tconf, requires_grad=False)
        coord_mask = Variable(coord_mask, requires_grad=False)
        conf_mask = Variable(conf_mask.sqrt(), requires_grad=False)
        if nC > 1:
            tcls = Variable(tcls, requires_grad=False)
            cls_mask = Variable(cls_mask, requires_grad=False)
            cls = cls[cls_mask].view(-1, nC)

        # Compute losses

        # corresponds to delta_region_box
        loss_coord = self.coord_scale * self.mse(coord * coord_mask, tcoord * coord_mask) / nB
        loss_conf = self.mse(conf * conf_mask, tconf * conf_mask) / nB
        if nC > 1 and cls.numel():
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

    def build_targets(self, pred_cxywh, ground_truth, nH, nW, seen=0, gt_weights=None):
        """ Compare prediction boxes and targets, convert targets to network output tensors """
        return self._build_targets_tensor(pred_cxywh, ground_truth, nH, nW, seen=seen, gt_weights=gt_weights)

    @profiler.profile
    def _build_targets_tensor(self, pred_cxywh, ground_truth, nH, nW, seen=0, gt_weights=None):
        """
        Compare prediction boxes and ground truths, convert ground truths to network output tensors

        Args:
            pred_cxywh (Tensor):   shape [B * A * W * H, 4] in normalized cxywh format
            ground_truth (Tensor): shape [B, max(gtannots), 4]

        Example:
            >>> from netharn.models.yolo2.light_yolo import Yolo
            >>> from netharn.models.yolo2.light_region_loss import RegionLoss
            >>> torch.random.manual_seed(0)
            >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
            >>> self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
            >>> Win, Hin = 96, 96
            >>> nW, nH = 3, 3
            >>> ground_truth = torch.FloatTensor([])
            >>> gt_weights = torch.FloatTensor([[-1, -1, -1], [1, 1, 0]])
            >>> #pred_cxywh = torch.rand(90, 4)
            >>> nB = len(gt_weights)
            >>> pred_cxywh = torch.rand(nB, len(anchors), nH, nW, 4).view(-1, 4)
            >>> seen = 0
            >>> self._build_targets_tensor(pred_cxywh, ground_truth, nH, nW, seen, gt_weights)

        Example:
            >>> from netharn.models.yolo2.light_region_loss import RegionLoss
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([[.75, .75], [1.0, .3], [.3, 1.0]])
            >>> self = RegionLoss(num_classes=2, anchors=anchors)
            >>> nW, nH = 2, 2
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
            >>> gt_weights = torch.FloatTensor([[-1, -1, -1], [1, 1, 0]])
            >>> nB = len(gt_weights)
            >>> pred_cxywh = torch.rand(nB, len(anchors), nH, nW, 4).view(-1, 4)
            >>> seen = 0
            >>> coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self._build_targets_tensor(pred_cxywh, ground_truth, nH, nW, seen, gt_weights)
        """
        gtempty = (ground_truth.numel() == 0)

        # Parameters
        nB = ground_truth.shape[0] if not gtempty else 0
        nT = ground_truth.shape[1] if not gtempty else 0
        nA = self.num_anchors

        if nB == 0:
            # torch does not preserve shapes when any dimension goes to 0
            # fix nB if there is no groundtruth
            nB = int(len(pred_cxywh) / (nA * nH * nW))
        else:
            assert nB == int(len(pred_cxywh) / (nA * nH * nW)), 'bad assumption'

        seen = seen + nB

        # Tensors
        device = self.get_device()
        conf_mask = torch.ones(nB, nA, 1, nH, nW, device=device)
        coord_mask = torch.zeros(nB, nA, 1, nH, nW, device=device)
        cls_mask = torch.zeros(nB, nA, 1, nH, nW, device=device).byte()
        tcoord = torch.zeros(nB, nA, 4, nH, nW, device=device)
        tconf = torch.zeros(nB, nA, 1, nH, nW, device=device)
        tcls = torch.zeros(nB, nA, 1, nH, nW, device=device)

        conf_mask = conf_mask * self.noobject_scale

        # if device is not None:
        #     conf_mask = conf_mask.to(device)
        #     coord_mask = coord_mask.to(device)
        #     cls_mask = cls_mask.to(device)
        #     tcoord = tcoord.to(device)
        #     tconf = tconf.to(device)
        #     tcls = tcls.to(device)

        if seen < 12800:
            coord_mask.fill_(1)
            # When the network is starting off, encourage background boxes to expand in size
            tcoord[:, :, 0, ...].fill_(0.5)
            tcoord[:, :, 1, ...].fill_(0.5)

        if not gtempty:
            # Put this back into a non-flat view
            pred_cxywh = pred_cxywh.view(nB, nA, nH, nW, 4)
            pred_boxes = util.Boxes(pred_cxywh, 'cxywh')

            gt_class = ground_truth[..., 0].data
            gt_boxes_norm = util.Boxes(ground_truth[..., 1:5], 'cxywh')
            gt_boxes = gt_boxes_norm.scale([nW, nH])
            # Construct "relative" versions of the true boxes, centered at 0
            rel_gt_boxes = gt_boxes.copy()
            rel_gt_boxes.data[..., 0:2] = 0
            # rel_gt_boxes = gt_boxes.translate(-gt_boxes.xy_center)

            # true boxes with a class of -1 are fillers, ignore them
            gt_isvalid = (gt_class >= 0)

            # Loop over ground_truths and construct tensors
            for bx in range(nB):
                # Get the actual groundtruth boxes for this batch item
                flags = gt_isvalid[bx]
                if not np.any(flags):
                    continue

                # Create gt anchor assignments
                batch_rel_gt_boxes = rel_gt_boxes[bx][flags]
                anchor_ious = self.rel_anchors_boxes.ious(batch_rel_gt_boxes, bias=0)
                best_ns = anchor_ious.max(dim=0)[1]

                # Setting confidence mask
                cur_pred_boxes = pred_boxes[bx]
                # pred_boxes.data.view(-1, 4)[bx * nAnchors:(bx + 1) * nAnchors]
                # cur_pred_boxes = pred_boxes[bx * nAnchors:(bx + 1) * nAnchors]
                cur_gt_boxes = gt_boxes[bx][flags]

                ious = cur_pred_boxes.ious(cur_gt_boxes, bias=0)

                cur_ious = ious.max(dim=-1)[0]
                conf_mask[bx].view(-1)[cur_ious.view(-1) > self.thresh] = 0

                for t in range(nT):
                    if not flags[t]:
                        break

                    if gt_weights is None:
                        weight = 1
                    else:
                        weight = gt_weights[bx][t]

                    gx, gy, gw, gh = gt_boxes[bx][t].to_cxywh().data
                    gi = min(nW - 1, max(0, int(gx)))
                    gj = min(nH - 1, max(0, int(gy)))

                    best_n = best_ns[t]
                    best_aw, best_ah = self.anchors[best_n]

                    gt_box_ = gt_boxes[bx][t]
                    pred_box_ = cur_pred_boxes[best_n, gj, gi]

                    iou = gt_box_[None, :].ious(pred_box_[None, :], bias=0)[0, 0]

                    if weight == 0:
                        # HACK: Only allow weight == 0 and weight == 1 for now
                        # TODO:
                        #    - [ ] Allow for continuous weights
                        #    - [ ] Allow for per-image background weight
                        conf_mask[bx, best_n, 0, gj, gi] = 0
                    else:
                        assert weight == 1, (
                            'can only have weight in {0, 1} for now')
                        coord_mask[bx, best_n, 0, gj, gi] = 1
                        cls_mask[bx, best_n, 0, gj, gi] = 1
                        conf_mask[bx, best_n, 0, gj, gi] = self.object_scale

                        tcoord[bx, best_n, 0, gj, gi] = gx - gi
                        tcoord[bx, best_n, 1, gj, gi] = gy - gj
                        tcoord[bx, best_n, 2, gj, gi] = math.log(gw / best_aw)
                        tcoord[bx, best_n, 3, gj, gi] = math.log(gh / best_ah)
                        tconf[bx, best_n, 0, gj, gi] = iou
                        tcls[bx, best_n, 0, gj, gi] = ground_truth[bx, t, 0]

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.models.yolo2.light_region_loss all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
