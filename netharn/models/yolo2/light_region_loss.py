#
#   Loss modules
#   Copyright EAVISE
#

import math
import warnings
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from lightnet.util import bbox_iou, bbox_multi_ious

__all__ = ['RegionLoss']


def _safelog(x):
    return math.log(max(x, np.finfo(np.float).eps))


class RegionLoss(torch.nn.modules.loss._Loss):
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

    Example:
        >>> from clab.models.yolo2.light_yolo import Yolo
        >>> from brambox.boxes.annotations import Annotation
        >>> torch.random.manual_seed(0)
        >>> def as_anno(class_id, x_center, y_center, w, h, Win, Hin):
        >>>     anno = Annotation()
        >>>     anno.class_id = class_id
        >>>     anno.x_top_left = (x_center - w / 2) * Win
        >>>     anno.y_top_left = (y_center - h / 2) * Hin
        >>>     anno.width, anno.height = w * Win, h * Hin
        >>>     return anno
        >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
        >>> anchors = {'num': network.num_anchors, 'values': network.anchors}
        >>> self = RegionLoss(num_classes=network.num_classes, anchors=anchors)
        >>> Win, Hin = 96, 96
        >>> # Annotation coordinates are specified wrt the input image size
        >>> target = [
        >>>     # boxes for batch item 1
        >>>     [as_anno(0, 0.50, 0.50, 1.00, 1.00, Win, Hin),
        >>>      as_anno(1, 0.32, 0.42, 0.22, 0.12, Win, Hin)],
        >>>     # boxes for batch item 2 (it has no objects!)
        >>>     []
        >>> ]
        >>> im_data = torch.randn(len(target), 3, Hin, Win)
        >>> output = network.forward(im_data)
        >>> loss = float(self(output, target))
        >>> print(f'loss = {loss:.2f}')
        loss = 20.18

    Example:
        >>> from clab.models.yolo2.light_yolo import Yolo
        >>> torch.random.manual_seed(0)
        >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
        >>> anchors = {'num': network.num_anchors, 'values': network.anchors}
        >>> self = RegionLoss(num_classes=network.num_classes, anchors=anchors)
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
        >>> loss = float(self(output, target))
        >>> print(f'loss = {loss:.2f}')
        loss = 20.18
    """
    def __init__(self, num_classes=None, anchors=None, coord_scale=1.0,
                 noobject_scale=1.0, object_scale=5.0, class_scale=1.0,
                 thresh=0.6):
        super().__init__()

        if num_classes is None:
            raise ValueError('Must specify num_classes if network is None')
        if anchors is None:
            raise ValueError('Must specify anchors if network is None')

        self.num_classes = num_classes
        self.anchors = anchors['values']
        self.num_anchors = anchors['num']
        self.anchor_step = len(self.anchors) // self.num_anchors
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

    def forward(self, output, target, seen=0):
        """ Compute Region loss.

        Args:
            output (torch.autograd.Variable): Output from the network
            target (brambox.boxes.annotations.Annotation or torch.Tensor): Brambox annotations or tensor containing the annotation targets (see :class:`lightnet.data.BramboxToTensor`)
            seen (int): if specified, overrides the `seen` attribute read from `self.net` (default None)

        Note:
            If target is a tensor, the shape should be [B, T, 5], where B is
            the batch size, T is the maximum number of boxes in an item, and
            the final dimension should correspond to [class_idx, center_x,
            center_y, width, height]. Items with fewer than T boxes should be
            padded with dummy boxes with class_idx=-1.
        """
        # Parameters
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cuda = output.is_cuda
        if isinstance(target, Variable):
            target = target.data

        # Get x,y,w,h,conf,cls
        output = output.view(nB, nA, -1, nH*nW)
        coord  = torch.zeros_like(output[:,:,:4])
        coord[:,:,:2]  = output[:,:,:2].sigmoid()   # tx,ty
        coord[:,:,2:4] = output[:,:,2:4]            # tw,th
        conf   = output[:,:,4].sigmoid()
        if nC > 1:
            cls = output[:,:,5:].contiguous().view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(-1, nC)

        # Create prediction boxes
        pred_boxes = torch.FloatTensor(nB*nA*nH*nW, 4)
        lin_x = torch.linspace(0, nW-1, nW).repeat(nH,1).view(nH*nW)
        lin_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().contiguous().view(nH*nW)
        anchor_w = torch.Tensor(self.anchors[::self.anchor_step]).view(nA, 1)
        anchor_h = torch.Tensor(self.anchors[1::self.anchor_step]).view(nA, 1)
        if cuda:
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes[:,0] = (coord[:,:,0].data + lin_x).view(-1)
        pred_boxes[:,1] = (coord[:,:,1].data + lin_y).view(-1)
        pred_boxes[:,2] = (coord[:,:,2].data.exp() * anchor_w).view(-1)
        pred_boxes[:,3] = (coord[:,:,3].data.exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()

        # Create predicted confs
        pred_confs = torch.FloatTensor(nB*nA*nH*nW)
        pred_confs = conf.data.view(-1).cpu()

        # Get target values
        coord_mask,conf_mask,cls_mask,tcoord,tconf,tcls = self.build_targets(pred_boxes,pred_confs,target,nH,nW, seen=seen)
        coord_mask = coord_mask.expand_as(tcoord)
        if nC > 1:
            tcls = tcls.view(-1)[cls_mask.view(-1)].long()
            cls_mask = cls_mask.view(-1, 1).repeat(1, nC)

        if cuda:
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            if nC > 1:
                tcls = tcls.cuda()
                cls_mask = cls_mask.cuda()

        tcoord = Variable(tcoord, requires_grad=False)
        tconf  = Variable(tconf, requires_grad=False)
        coord_mask = Variable(coord_mask, requires_grad=False)
        conf_mask  = Variable(conf_mask.sqrt(), requires_grad=False)
        if nC > 1:
            tcls  = Variable(tcls, requires_grad=False)
            cls_mask = Variable(cls_mask, requires_grad=False)
            cls      = cls[cls_mask].view(-1, nC)

        # Compute losses
        mse = nn.MSELoss(size_average=False)
        self.loss_coord = self.coord_scale * mse(coord*coord_mask, tcoord*coord_mask) / nB
        self.loss_conf = mse(conf*conf_mask, tconf*conf_mask) / nB
        if nC > 1:
            self.loss_cls = self.class_scale * 2 * nn.CrossEntropyLoss(size_average=False)(cls, tcls) / nB
            self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
        else:
            self.loss_cls = None
            self.loss_tot = self.loss_coord + self.loss_conf

        return self.loss_tot

    def build_targets(self, pred_boxes, pred_confs, ground_truth, nH, nW, seen=0):
        """ Compare prediction boxes and targets, convert targets to network output tensors """
        if torch.is_tensor(ground_truth):
            return self.__build_targets_tensor(pred_boxes, pred_confs, ground_truth, nH, nW, seen=seen)
        else:
            return self.__build_targets_brambox(pred_boxes, pred_confs, ground_truth, nH, nW, seen=seen)

    def __build_targets_tensor(self, pred_boxes, pred_confs, ground_truth, nH, nW, seen=0):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        nB = ground_truth.size(0)
        nT = ground_truth.size(1)
        nA = self.num_anchors
        nAnchors = nA*nH*nW
        nPixels  = nH*nW

        seen = seen + nB

        # Tensors
        conf_mask  = torch.ones(nB, nA, nH*nW) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, 1, nH*nW)
        cls_mask   = torch.zeros(nB, nA, nH*nW).byte()
        tcoord     = torch.zeros(nB, nA, 4, nH*nW)
        tconf      = torch.zeros(nB, nA, nH*nW)
        tcls       = torch.zeros(nB, nA, nH*nW)

        if seen < 12800:
            coord_mask.fill_(1)
            if self.anchor_step == 4:
                tcoord[:,:,0] = torch.Tensor(self.anchors[2::self.anchor_step]).view(1,nA,1,1).repeat(nB,1,1,nH*nW)
                tcoord[:,:,1] = torch.Tensor(self.anchors[3::self.anchor_step]).view(1,nA,1,1).repeat(nB,1,1,nH*nW)
            else:
                tcoord[:,:,0].fill_(0.5)
                tcoord[:,:,1].fill_(0.5)

        # Setting confidence mask
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors]
            cur_pred_confs = pred_confs[b*nAnchors:(b+1)*nAnchors]
            cur_ious = torch.zeros(nAnchors)
            for t in range(nT):
                if ground_truth[b][t][0] < 0:
                    break
                gx = ground_truth[b][t][1] * nW
                gy = ground_truth[b][t][2] * nH
                gw = ground_truth[b][t][3] * nW
                gh = ground_truth[b][t][4] * nH
                cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1)
                cur_ious = torch.max(cur_ious, bbox_multi_ious(cur_pred_boxes, cur_gt_boxes))
            conf_mask[b].view(-1)[cur_ious > self.thresh] = 0

        # Loop over ground_truths and construct tensors
        for b in range(nB):
            for t in range(nT):
                if ground_truth[b][t][0] < 0:
                    break
                best_iou = 0.0
                best_n = -1
                min_dist = 10000
                gx = ground_truth[b][t][1] * nW
                gy = ground_truth[b][t][2] * nH
                gw = ground_truth[b][t][3] * nW
                gh = ground_truth[b][t][4] * nH
                gi = min(nW-1, max(0, int(gx)))
                gj = min(nH-1, max(0, int(gy)))
                gt_box = [0, 0, gw, gh]
                for n in range(nA):
                    aw = self.anchors[self.anchor_step*n]
                    ah = self.anchors[self.anchor_step*n+1]
                    anchor_box = [0, 0, aw, ah]
                    iou  = bbox_iou(anchor_box, gt_box)
                    if self.anchor_step == 4:
                        ax = self.anchors[self.anchor_step*n+2]
                        ay = self.anchors[self.anchor_step*n+3]
                        dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                    elif self.anchor_step==4 and iou == best_iou and dist < min_dist:
                        best_iou = iou
                        best_n = n
                        min_dist = dist

                gt_box = [gx, gy, gw, gh]
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
                pred_conf = pred_confs[b*nAnchors+best_n*nPixels+gj*nW+gi]
                iou = bbox_iou(gt_box, pred_box)

                coord_mask[b][best_n][0][gj*nW+gi] = 1
                cls_mask[b][best_n][gj*nW+gi] = 1
                conf_mask[b][best_n][gj*nW+gi] = self.object_scale
                tcoord[b][best_n][0][gj*nW+gi] = gx - gi
                tcoord[b][best_n][1][gj*nW+gi] = gy - gj

                tcoord[b][best_n][2][gj*nW+gi] = _safelog(
                    gw/self.anchors[self.anchor_step*best_n])
                tcoord[b][best_n][3][gj*nW+gi] = _safelog(
                    gh/self.anchors[self.anchor_step*best_n+1])
                tconf[b][best_n][gj*nW+gi] = iou
                tcls[b][best_n][gj*nW+gi] = ground_truth[b][t][0]

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def __build_targets_brambox(self, pred_boxes, pred_confs, ground_truth, nH, nW, seen=0):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        nB = len(ground_truth)
        nA = self.num_anchors
        nAnchors = nA*nH*nW
        nPixels  = nH*nW

        seen = seen + nB

        # Tensors
        conf_mask  = torch.ones(nB, nA, nH*nW) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, 1, nH*nW)
        cls_mask   = torch.zeros(nB, nA, nH*nW).byte()
        tcoord     = torch.zeros(nB, nA, 4, nH*nW)
        tconf      = torch.zeros(nB, nA, nH*nW)
        tcls       = torch.zeros(nB, nA, nH*nW)

        if seen < 12800:
            coord_mask.fill_(1)
            if self.anchor_step == 4:
                tcoord[:,:,0] = torch.Tensor(self.anchors[2::self.anchor_step]).view(1,nA,1,1).repeat(nB,1,1,nH*nW)
                tcoord[:,:,1] = torch.Tensor(self.anchors[3::self.anchor_step]).view(1,nA,1,1).repeat(nB,1,1,nH*nW)
            else:
                tcoord[:,:,0].fill_(0.5)
                tcoord[:,:,1].fill_(0.5)

        # Setting confidence mask
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors]
            cur_pred_confs = pred_confs[b*nAnchors:(b+1)*nAnchors]
            cur_ious = torch.zeros(nAnchors)
            for anno in ground_truth[b]:
                gx = (anno.x_top_left + anno.width/2) / self.reduction
                gy = (anno.y_top_left + anno.height/2) / self.reduction
                gw = anno.width / self.reduction
                gh = anno.height / self.reduction
                cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).expand(nAnchors,-1)
                cur_ious = torch.max(cur_ious, bbox_multi_ious(cur_pred_boxes, cur_gt_boxes))
            conf_mask[b].view(-1)[cur_ious > self.thresh] = 0

        # Loop over ground_truths and construct tensors
        for b in range(nB):
            for anno in ground_truth[b]:
                best_iou = 0.0
                best_n = -1
                min_dist = 10000
                gx = (anno.x_top_left + anno.width/2) / self.reduction
                gy = (anno.y_top_left + anno.height/2) / self.reduction
                gw = anno.width / self.reduction
                gh = anno.height / self.reduction
                gi = min(nW-1, max(0, int(gx)))
                gj = min(nH-1, max(0, int(gy)))
                gt_box = [0, 0, gw, gh]
                for n in range(nA):
                    aw = self.anchors[self.anchor_step*n]
                    ah = self.anchors[self.anchor_step*n+1]
                    anchor_box = [0, 0, aw, ah]
                    iou  = bbox_iou(anchor_box, gt_box)
                    if self.anchor_step == 4:
                        ax = self.anchors[self.anchor_step*n+2]
                        ay = self.anchors[self.anchor_step*n+3]
                        dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                    elif self.anchor_step==4 and iou == best_iou and dist < min_dist:
                        best_iou = iou
                        best_n = n
                        min_dist = dist

                gt_box = [gx, gy, gw, gh]
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
                pred_conf = pred_confs[b*nAnchors+best_n*nPixels+gj*nW+gi]
                iou = bbox_iou(gt_box, pred_box)

                if anno.ignore:
                    conf_mask[b][best_n][gj*nW+gi] = 0
                else:
                    coord_mask[b][best_n][0][gj*nW+gi] = 1
                    cls_mask[b][best_n][gj*nW+gi] = 1
                    conf_mask[b][best_n][gj*nW+gi] = self.object_scale
                    tcoord[b][best_n][0][gj*nW+gi] = gx - gi
                    tcoord[b][best_n][1][gj*nW+gi] = gy - gj
                    tcoord[b][best_n][2][gj*nW+gi] = math.log(gw/self.anchors[self.anchor_step*best_n])
                    tcoord[b][best_n][3][gj*nW+gi] = math.log(gh/self.anchors[self.anchor_step*best_n+1])
                    tconf[b][best_n][gj*nW+gi] = iou
                    tcls[b][best_n][gj*nW+gi] = anno.class_id

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

if __name__ == '__main__':
    """
    CommandLine:
        python -m clab.models.yolo2.region_loss all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
