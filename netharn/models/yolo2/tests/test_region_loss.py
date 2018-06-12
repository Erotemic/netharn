import torch
import numpy as np  # NOQA
from netharn import util
from netharn.util import profiler
from netharn.models.yolo2 import light_region_loss


def check_loss_value():
    pass


def check_perfect_score():
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


"""


PJReddie's delta formulation:

    # NOTES:
        output coordinates -
           x and y are offsets from anchor centers in Wout,Hout space.
           w and h are in 01 coordinates


        truth.{x, y, w, h} - true box in 01 coordinates
        tx, ty, tw, th - true box in output coordinates

    # Transform output coordinates to bbox pred
    box get_region_box(float *x, float *biases, int n, int index, int i, int j,
                       int w, int h, int stride)
    {
        # bbox pred is in 0 - 1 coordinates
        box b;
        b.x = (i + x[index + 0*stride]) / w;
        b.y = (j + x[index + 1*stride]) / h;
        b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
        b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
        return b;
    }


    # VOC Config:
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg

    # WHEN SEEN < 128000 CASE
    if(*(net.seen) < 12800){
        box truth = {0};
        truth.x = (i + .5)/l.w;
        truth.y = (j + .5)/l.h;
        truth.w = l.biases[2*n]/l.w;    # l.biases are anchor boxes
        truth.h = l.biases[2*n+1]/l.h;
        delta_region_box(
            truth, l.output, l.biases, n, box_index, i, j, l.w, l.h,
            delta=l.delta, scale=.01, stride=l.w*l.h);
    }

    # COORDINATE LOSS
    # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L254
    # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L86
    # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L293
    float iou = delta_region_box(
        truth, l.output, l.biases,
        n=best_n, index=box_index, i=i, j=j, w=l.w, h=l.h, delta=l.delta,
        scale=l.coord_scale * (2 - truth.w*truth.h), stride=l.w*l.h);
    {
        # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L86

        # CONVERT THE TRUTH BOX INTO OUTPUT COORD SPACE
        float tx = (truth.x * l.w - i);
        float ty = (truth.y * l.h - j);
        float tw = log(truth.w * l.w / biases[2*n]);
        float th = log(truth.h * l.h / biases[2*n + 1]);

        delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
        delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
        delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
        delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    }


    # CLASSIFICATION LOSS
    # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L112
    # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L314
    delta_region_class(
        output=l.output, delta=l.delta, index=class_index, class=class,
        classes=l.classes, hier=l.softmax_tree, scale=l.class_scale,
        stride=l.w*l.h, avg_cat=&avg_cat, tag=!l.softmax);


    # OBJECTNESS LOSS: FOREGROUND
    # THIS IS THE DEFAULT IOU LOSS FOR UNMATCHED OBJECTS
    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
    if(l.background) {
        # BG is 0
        l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
    }
    if (best_iou > l.thresh) {
        l.delta[obj_index] = 0;
    }


    # OBJECTNESS LOSS: BACKGROUND
    l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
    # rescore is 1 for VOC
    if (l.rescore) {
        l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
    }
    # background defaults to 0
    if(l.background){
        l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
    }


    IOU computation equivalent to:
        * default all conf_mask to noobject_scale
        * default all tconf to 0
        * for any predictions with a best iou over a threshold (0.6)
            - set its mask to 0 (equivalent to setting its delta to 0)
        * for any matching pred / true positions:
            - switch conf_mask to object_scale
            - switch tconf to (iou if rescore else 1)

Summary:
    Coordinate Loss:
        loss = coord_scale * (m * t - m * p) ** 2

        * When seen < N, YOLO sets the scale to 0.01
            - In our formulation, set m=sqrt(0.01 / coord_scale)
"""


def sympy_check_coordinate_loss():
    import sympy
    # s = coord_scale
    # m = coord_mask
    # p = coords
    # t = tcoords
    # tw, th = true width and height in normalized 01 coordinates
    s, m, p, t, tw, th = sympy.symbols('s, m, p, t, tw, th')

    # This is the general MSE loss forumlation for box coordinates We will
    # always use this at the end of our computation (so we can efficiently
    # broadcast the tensor). s, t, and p are always fixed constants.
    # The main questsion is: how do we set m to match PJR's results?
    loss_box = .5 * s * (m * t - m * p) ** 2

    # Our formulation produces this negative derivative
    our_grad = -1 * sympy.diff(loss_box, p)
    print('our_grad = {!r}'.format(our_grad))

    # PJReddie hard codes this negative derivative
    pjr_grad = s * (t - p)

    # Check how they differ
    # (see what the mask must be assuming the scale is the same)
    eq = sympy.Eq(pjr_grad, our_grad)
    sympy.solve(eq)

    # However, the scale is not the same in all cases PJReddie will change it
    # Depending on certain cases.

    # Coordinate Loss:
    #     loss = coord_scale * (m * t - m * p) ** 2

    #
    # BACKGROUND LOW SEEN CASE:
    # --------------
    #   * When seen < N, YOLO sets the scale to 0.01
    #       - In our formulation, set m=sqrt(0.01 / coord_scale)
    #
    #  * NOTE this loss is only applied to background predictions,
    #       it is overridden if a a true object is assigned to the box
    #       this is only a default case.
    bgseen_pjr_grad = pjr_grad.subs({s: 0.01})
    eq = sympy.Eq(bgseen_pjr_grad, our_grad)
    bgseen_m_soln = sympy.solve(eq, m)[1]
    print('when seen < N, set m = {}'.format(bgseen_m_soln))
    # Check that this is what we expect
    bgseen_m_expected = sympy.sqrt(0.01 / s)
    assert (bgseen_m_soln - bgseen_m_expected) == 0

    #
    # BACKGROUND LOW SEEN CASE:
    #     * when seen is high, the bbox loss in the background is just 0, we
    #     can achive this by simply setting m = 0
    assert our_grad.subs({'m': 0}) == 0

    #
    # FORGROUND NORMAL CASE
    # -----------
    # Whenever a pred box is assigned to a true object it gets this coordinate
    # loss

    # In the normal case pjr sets
    # true box width and height
    norm_pjr_grad = pjr_grad.subs({s: s * (2 - tw * th)})
    eq = sympy.Eq(norm_pjr_grad, our_grad)
    fgnorm_m_soln = sympy.solve(eq, m)[1]
    print('fgnorm_m_soln = {!r}'.format(fgnorm_m_soln))
    # Check that this is what we expect
    fgnorm_m_expected = sympy.sqrt(2.0 - th * tw)
    assert (fgnorm_m_soln - fgnorm_m_expected) == 0

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/models/yolo2/tests/test_region_loss.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
