import numpy as np
import torch
import ubelt as ub


def bboxes_iou_light(boxes1, boxes2):
    import itertools as it
    results = []
    for box1, box2 in it.product(boxes1, boxes2):
        result = bbox_iou_light(box1, box2)
        results.append(result)
    return results


def bbox_iou_light(box1, box2):
    """ Compute IOU between 2 bounding boxes
        Box format: [xc, yc, w, h]

    from netharn import util
    boxes1_ = util.Boxes(boxes1, 'tlbr').to_cxywh().data
    boxes2_ = util.Boxes(boxes2, 'tlbr').to_cxywh().data
    bboxes_iou_light(boxes1_, boxes2_)

    ious_c = bbox_ious_c(boxes1, boxes2, bias=0)

    """
    mx = min(box1[0]-box1[2]/2, box2[0]-box2[2]/2)
    Mx = max(box1[0]+box1[2]/2, box2[0]+box2[2]/2)
    my = min(box1[1]-box1[3]/2, box2[1]-box2[3]/2)
    My = max(box1[1]+box1[3]/2, box2[1]+box2[3]/2)
    w1 = box1[2]
    h1 = box1[3]
    w2 = box2[2]
    h2 = box2[3]

    uw = Mx - mx
    uh = My - my
    iw = w1 + w2 - uw
    ih = h1 + h2 - uh
    if iw <= 0 or ih <= 0:
        return 0

    area1 = w1 * h1
    area2 = w2 * h2
    iarea = iw * ih
    uarea = area1 + area2 - iarea
    return iarea / uarea


def box_ious_py3(boxes1, boxes2, bias=1):
    N = boxes1.shape[0]
    K = boxes2.shape[0]

    # Preallocate output
    intersec = np.zeros((N, K), dtype=np.float32)

    inter_areas3 = np.zeros((N, K), dtype=np.float32)
    union_areas3 = np.zeros((N, K), dtype=np.float32)
    iws3 = np.zeros((N, K), dtype=np.float32)
    ihs3 = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        qbox_area = (
            (boxes2[k, 2] - boxes2[k, 0] + bias) *
            (boxes2[k, 3] - boxes2[k, 1] + bias)
        )
        for n in range(N):
            iw = (
                min(boxes1[n, 2], boxes2[k, 2]) -
                max(boxes1[n, 0], boxes2[k, 0]) + bias
            )
            iw = max(iw, 0)

            # if iw > 0:
            ih = (
                min(boxes1[n, 3], boxes2[k, 3]) -
                max(boxes1[n, 1], boxes2[k, 1]) + bias
            )
            ih = max(ih, 0)
            # if ih > 0:
            box_area = (
                (boxes1[n, 2] - boxes1[n, 0] + bias) *
                (boxes1[n, 3] - boxes1[n, 1] + bias)
            )
            inter_area = iw * ih
            union_area = (qbox_area + box_area - inter_area)

            ihs3[n, k] = ih
            iws3[n, k] = iw
            inter_areas3[n, k] = inter_area
            union_areas3[n, k] = union_area

            intersec[n, k] = inter_area / union_area
    return intersec


def box_ious_py2(boxes1, boxes2, bias=1):
    """
    Implementation using 2d index based filtering.
    It turns out that this is slower than the dense version.

    for a 5x7:
        %timeit box_ious_py2(boxes1, boxes2)
        %timeit box_ious_py(boxes1, boxes2)
        101 µs ± 1.47 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        42.5 µs ± 298 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    for a 45x7:
        boxes1 = Boxes(random_boxes(100, scale=100.0).numpy(), 'tlbr').data
        boxes2 = Boxes(random_boxes(80, scale=100.0).numpy(), 'tlbr').data
        %timeit box_ious_py2(boxes1, boxes2)
        %timeit box_ious_py(boxes1, boxes2)
        %timeit bbox_ious_c(boxes1, boxes2)

        116 µs ± 962 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        49.2 µs ± 824 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

        from netharn import util
        _ = util.profile_onthefly(box_ious_py2)(boxes1, boxes2)
        _ = util.profile_onthefly(box_ious_py)(boxes1, boxes2)

    Benchmark:
        from netharn import util
        boxes1 = Boxes(random_boxes(45, scale=100.0).numpy(), 'tlbr').data
        boxes2 = Boxes(random_boxes(7, scale=100.0).numpy(), 'tlbr').data

        import ubelt as ub
        for timer in ub.Timerit(100, bestof=10, label='c'):
            bbox_ious_c(boxes1, boxes2)

        # from netharn.util.cython_boxes import bbox_ious_c_par
        # import ubelt as ub
        # for timer in ub.Timerit(100, bestof=10, label='c'):
        #     bbox_ious_c_par(boxes1, boxes2)

        for timer in ub.Timerit(100, bestof=10, label='py1'):
            box_ious_py1(boxes1, boxes2)

        for timer in ub.Timerit(100, bestof=10, label='py2'):
            box_ious_py2(boxes1, boxes2)

        for timer in ub.Timerit(100, bestof=10, label='py3'):
            box_ious_py3(boxes1, boxes2)

        boxes1_ = util.Boxes(boxes1, 'tlbr').to_cxywh().data
        boxes2_ = util.Boxes(boxes2, 'tlbr').to_cxywh().data

        for timer in ub.Timerit(100, bestof=10, label='py3'):
            bboxes_iou_light(boxes1_, boxes2_)
    """
    N = len(boxes1)
    K = len(boxes2)

    ax1, ay1, ax2, ay2 = (boxes1.T)
    bx1, by1, bx2, by2 = (boxes2.T)
    aw, ah = (ax2 - ax1 + bias), (ay2 - ay1 + bias)
    bw, bh = (bx2 - bx1 + bias), (by2 - by1 + bias)

    areas1 = aw * ah
    areas2 = bw * bh

    # Create all pairs of boxes
    ns = np.repeat(np.arange(N), K, axis=0)
    ks = np.repeat(np.arange(K)[None, :], N, axis=0).ravel()

    ex_ax1 = np.repeat(ax1, K, axis=0)
    ex_ay1 = np.repeat(ay1, K, axis=0)
    ex_ax2 = np.repeat(ax2, K, axis=0)
    ex_ay2 = np.repeat(ay2, K, axis=0)

    ex_bx1 = np.repeat(bx1[None, :], N, axis=0).ravel()
    ex_by1 = np.repeat(by1[None, :], N, axis=0).ravel()
    ex_bx2 = np.repeat(bx2[None, :], N, axis=0).ravel()
    ex_by2 = np.repeat(by2[None, :], N, axis=0).ravel()

    x_maxs = np.minimum(ex_ax2, ex_bx2)
    x_mins = np.maximum(ex_ax1, ex_bx1)

    iws = (x_maxs - x_mins + bias)

    # Remove pairs of boxes that don't intersect in the x dimension
    flags = iws > 0
    ex_ay1 = ex_ay1.compress(flags, axis=0)
    ex_ay2 = ex_ay2.compress(flags, axis=0)
    ex_by1 = ex_by1.compress(flags, axis=0)
    ex_by2 = ex_by2.compress(flags, axis=0)
    ns = ns.compress(flags, axis=0)
    ks = ks.compress(flags, axis=0)
    iws = iws.compress(flags, axis=0)

    y_maxs = np.minimum(ex_ay2, ex_by2)
    y_mins = np.maximum(ex_ay1, ex_by1)

    ihs = (y_maxs - y_mins + bias)

    # Remove pairs of boxes that don't intersect in the x dimension
    flags = ihs > 0
    ns = ns.compress(flags, axis=0)
    ks = ks.compress(flags, axis=0)
    iws = iws.compress(flags, axis=0)
    ihs = ihs.compress(flags, axis=0)

    areas_sum = areas1[ns] + areas2[ks]

    inter_areas = iws * ihs
    union_areas = (areas_sum - inter_areas)
    expanded_ious = inter_areas / union_areas

    ious = np.zeros((N, K), dtype=np.float32)
    ious[ns, ks] = expanded_ious
    return ious
