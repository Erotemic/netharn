

def torch_nms(bboxes, scores, classes=None, thresh=.5):
    """
    Non maximum suppression implemented with pytorch tensors

    Args:
        bboxes (Tensor): Bounding boxes of one image in the format (x1, y1, x2, y2)
        scores (Tensor): Scores of each box
        classes (Tensor, optional): the classes of each box. If specified nms is applied to each class separately.
        thresh (float): iou threshold

    Returns:
        ByteTensor: keep: boolean array indicating which boxes were not pruned.

    Example:
        >>> import torch
        >>> import numpy as np
        >>> bboxes = torch.FloatTensor(np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>> ], dtype=np.float32))
        >>> scores = torch.FloatTensor(np.array([.1, .5, .9, .1]))
        >>> classes = torch.FloatTensor(np.array([0, 0, 0, 0]))
        >>> thresh = .5
        >>> keep = nms(bboxes, scores, classes, thresh)
        >>> bboxes[keep]
    """
    if bboxes.numel() == 0:
        return []

    # Sort coordinates by descending score
    scores, order = scores.sort(0, descending=True)

    x1, y1, x2, y2 = bboxes[order].split(1, 1)

    # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
    dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp_(min=0)
    dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp_(min=0)

    # Compute iou
    intersections = dx * dy
    areas = (x2 - x1) * (y2 - y1)
    unions = (areas + areas.t()) - intersections
    ious = intersections / unions

    # Filter based on iou (and class)
    conflicting = (ious > thresh).triu(1)

    if classes is not None:
        same_class = (classes.unsqueeze(0) == classes.unsqueeze(1))
        conflicting = (conflicting & same_class)

    ordered_keep = (conflicting.sum(0) == 0)  # Unlike numpy, pytorch cannot perform any() along a certain axis
    keep = ordered_keep.new(*ordered_keep.size())
    keep.scatter_(0, order, ordered_keep)  # Unsort, so keep is aligned with input boxes
    return keep

    aaa = torch.LongTensor(np.arange(len(boxes))).reshape(-1, 1)

    sorted(aaa[order][keep1[:, None].expand_as(aaa)].cpu().numpy().ravel()) == sorted(aaa[keep].cpu().numpy().ravel())

    bboxes[keep]
    # keep1 = (conflicting.sum(0) == 0)    # Unlike numpy, pytorch cannot perform any() along a certain axis
    # bboxes[order][keep1[:, None].expand_as(bboxes)].view(-1, 4).contiguous()


def _benchmark():
    import ubelt
    import torch
    import numpy as np
    import netharn as nh
    from netharn.util.nms.torch_nms import torch_nms
    from netharn.util import non_max_supression
    import ubelt as ub
    import itertools as it

    N = 100
    bestof = 10

    ydata = ub.ddict(list)
    xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, 2000]

    rng = nh.util.ensure_rng(0)

    thresh = 0.5

    for num in xdata:

        outputs = {}

        # Build random test boxes and scores
        boxes = nh.util.Boxes.random(num, scale=10.0, rng=rng, format='tlbr', tensor=True).data
        scores = torch.Tensor(rng.rand(len(boxes)))

        t1 = ubelt.Timerit(N, bestof=bestof, label='torch(cpu)')
        for timer in t1:
            with timer:
                keep = torch_nms(boxes, scores, thresh=thresh)
        ydata[t1.label].append(t1.min())
        outputs[t1.label] = np.where(keep.cpu().numpy())[0]

        if torch.cuda.is_available():
            # Move boxes to the GPU
            gpu_boxes = boxes.cuda()
            gpu_scores = scores.cuda()

            t1 = ubelt.Timerit(N, bestof=bestof, label='torch(gpu)')
            for timer in t1:
                with timer:
                    keep = torch_nms(gpu_boxes, gpu_scores, thresh=thresh)
                    torch.cuda.synchronize()
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = np.where(keep.cpu().numpy())[0]

        # Move boxes to numpy
        np_boxes = boxes.cpu().numpy()
        np_scores = scores.cpu().numpy()

        t1 = ubelt.Timerit(N, bestof=bestof, label='numpy(cpu)')
        for timer in t1:
            with timer:
                keep = non_max_supression(np_boxes, np_scores, thresh=thresh, impl='py')
        ydata[t1.label].append(t1.min())
        outputs[t1.label] = sorted(keep)

        t1 = ubelt.Timerit(N, bestof=bestof, label='cython(cpu)')
        for timer in t1:
            with timer:
                keep = non_max_supression(np_boxes, np_scores, thresh=thresh, impl='cpu')
        ydata[t1.label].append(t1.min())
        outputs[t1.label] = sorted(keep)

        if torch.cuda.is_available():
            t1 = ubelt.Timerit(N, bestof=bestof, label='cython(gpu)')
            for timer in t1:
                with timer:
                    keep = non_max_supression(np_boxes, np_scores, thresh=thresh, impl='gpu')
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = sorted(keep)

        # Check that all kept boxes do not have more than `threshold` ious
        for key, idxs in outputs.items():
            ious = nh.util.box_ious(np_boxes[idxs], np_boxes[idxs])
            max_iou = (np.tril(ious) - np.eye(len(ious))).max()
            if max_iou > thresh:
                print('{} produced a bad result with max_iou={}'.format(key, max_iou))

        # Check result consistency:
        print('Result consistency:')
        for k1, k2 in it.combinations(outputs.keys(), 2):
            idxs1 = set(outputs[k1])
            idxs2 = set(outputs[k2])
            jaccard = len(idxs1 & idxs2) / len(idxs1 | idxs2)
            print('{}, {}: {}'.format(k1, k2, jaccard))

    nh.util.mplutil.qtensure()
    nh.util.mplutil.multi_plot(xdata, ydata, xlabel='num boxes', ylabel='seconds')
