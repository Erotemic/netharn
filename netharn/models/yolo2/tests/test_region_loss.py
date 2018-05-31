import torch
import numpy as np  # NOQA
from netharn import util
from netharn.util import profiler
from netharn.models.yolo2 import light_region_loss


def test_perfect_score():
    from netharn.models.yolo2.light_region_loss import RegionLoss
    torch.random.manual_seed(0)
    anchors = np.array([[.75, .75], [1.0, .3], [.3, 1.0]])
    self = RegionLoss(num_classes=2, anchors=anchors)
    nW, nH = 1, 2
    # true boxes for each item in the batch
    # each box encodes class, center, width, and height
    # coordinates are normalized in the range 0 to 1
    # items in each batch are padded with dummy boxes with class_id=-1
    target = torch.FloatTensor([
        # boxes for batch item 1
        [[0, 0.50, 0.50, 1.00, 1.00],
         [1, 0.34, 0.32, 0.12, 0.32],
         [1, 0.32, 0.42, 0.22, 0.12]],
    ])
    gt_weights = torch.FloatTensor([[1, 1, 0]])
    nB = len(gt_weights)
    nO = len(anchors) * (5 + self.num_classes)

    # At each grid cell there are nA anchors, for each anchor we have 4
    # bbox predictions, an iou prediction and nC class predictions.

    output = torch.rand(nB, nO, nH, nW)

    output = output.view(nB, nA, 5 + self.num_classes, nH, nW)
    # Predictions at grid cell 0, 0 for each anchor
    output[0, :, :, 0, 0]

    # TODO: construct output such that the resulting boxes
    # will matching the groundtruth perfectly. Ensure that
    # loss is zero in this case. Then slowly perterb boxes
    # and ensure loss decreases smootly.


    pred_cxywh = torch.FloatTensor(nB * nA * nH * nW, 4)
    lin_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).view(nH * nW)
    lin_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().contiguous().view(nH * nW)
    anchor_w = self.anchors[:, 0].contiguous().view(nA, 1)
    anchor_h = self.anchors[:, 1].contiguous().view(nA, 1)


    seen = 0
    _tup = self._build_targets_tensor(pred_cxywh, ground_truth, nH, nW, seen,
                                      gt_weights)
    coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = _tup


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
    self = light_region_loss.RegionLoss(num_classes=network.num_classes,
                                        anchors=network.anchors)
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
    python ~/code/netharn/netharn/models/yolo2/tests/test_region_loss.py profile_loss_speed --profile

    Benchmark:
        #>>> profile_loss_speed()
    """
    from netharn.models.yolo2.light_yolo import Yolo
    import lightnet.network
    import netharn as nh

    rng = util.ensure_rng(0)
    torch.random.manual_seed(0)
    network = Yolo(num_classes=2, conf_thresh=4e-2)

    self1 = light_region_loss.RegionLoss(
        num_classes=network.num_classes, anchors=network.anchors)
    self2 = lightnet.network.loss.RegionLoss(num_classes=network.num_classes,
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
        #>>> compare_loss_speed()
    """
    from netharn.models.yolo2.light_yolo import Yolo
    import lightnet.network
    import ubelt as ub
    torch.random.manual_seed(0)
    network = Yolo(num_classes=2, conf_thresh=4e-2)

    self1 = light_region_loss.RegionLoss(num_classes=network.num_classes,
                                         anchors=network.anchors)
    self2 = lightnet.network.loss.RegionLoss(num_classes=network.num_classes,
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

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/models/yolo2/tests/test_region_loss.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
