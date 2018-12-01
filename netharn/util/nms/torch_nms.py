import torch
import numpy as np


def torch_nms(tlbr, scores, classes=None, thresh=.5, bias=0, fast=False):
    """
    Non maximum suppression implemented with pytorch tensors

    CURRENTLY NOT WORKING

    Args:
        tlbr (Tensor): Bounding boxes of one image in the format (tlbr)
        scores (Tensor): Scores of each box
        classes (Tensor, optional): the classes of each box. If specified nms is applied to each class separately.
        thresh (float): iou threshold

    Returns:
        ByteTensor: keep: boolean array indicating which boxes were not pruned.

    Example:
        >>> # DISABLE_DOCTEST
        >>> import torch
        >>> import numpy as np
        >>> tlbr = torch.FloatTensor(np.array([
        >>>     [0, 0, 100, 100],
        >>>     [100, 100, 10, 10],
        >>>     [10, 10, 100, 100],
        >>>     [50, 50, 100, 100],
        >>>     [100, 100, 130, 130],
        >>>     [100, 100, 130, 130],
        >>>     [100, 100, 130, 130],
        >>> ], dtype=np.float32))
        >>> scores = torch.FloatTensor(np.array([.1, .5, .9, .1, .3, .5, .4]))
        >>> classes = torch.FloatTensor(np.array([0, 0, 0, 0, 0, 0]))
        >>> thresh = .5
        >>> keep = torch_nms(tlbr, scores, classes, thresh)
        >>> bboxes[keep]

    Example:
        >>> # DISABLE_DOCTEST
        >>> import torch
        >>> import numpy as np
        >>> # Test to check that conflicts are correctly resolved
        >>> tlbr = torch.FloatTensor(np.array([
        >>>     [100, 100, 150, 101],
        >>>     [120, 100, 180, 101],
        >>>     [150, 100, 200, 101],
        >>> ], dtype=np.float32))
        >>> scores = torch.FloatTensor(np.linspace(.8, .9, len(tlbr)))
        >>> classes = None
        >>> thresh = .3
        >>> keep = torch_nms(tlbr, scores, classes, thresh, fast=False)
        >>> bboxes[keep]
    """
    if tlbr.numel() == 0:
        return []

    # Sort coordinates by descending score
    ordered_scores, order = scores.sort(0, descending=True)

    from netharn import util
    boxes = util.Boxes(tlbr[order], 'tlbr')
    ious = boxes.ious(boxes, bias=bias)

    # if False:
    #     x1, y1, x2, y2 = tlbr[order].split(1, 1)

    #     # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
    #     dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp_(min=0)
    #     dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp_(min=0)

    #     # Compute iou
    #     intersections = dx * dy
    #     areas = (x2 - x1) * (y2 - y1)
    #     unions = (areas + areas.t()) - intersections
    #     ious = intersections / unions

    # Filter based on iou (and class)
    # NOTE: We are using following convention:
    #     * suppress if overlap > thresh
    #     * consider if overlap <= thresh
    # This convention has the property that when thresh=0, we dont just
    # remove everything.
    conflicting = (ious > thresh).triu(1)

    if classes is not None:
        ordered_classes = classes[order]
        same_class = (ordered_classes.unsqueeze(0) == ordered_classes.unsqueeze(1))
        conflicting = (conflicting & same_class)
    # Now we have a 2D matrix where conflicting[i, j] indicates if box[i]
    # conflicts with box[j]. For each box[i] we want to only keep the first
    # one that does not conflict with any other box[j].

    # Find out how many conflicts each ordered box has with other boxes that
    # have higher scores than it does. In other words...
    # n_conflicts[i] is the number of conflicts box[i] has with other boxes
    # that have a **higher score** than box[i] does. We will definately
    # keep any box where n_conflicts is 0, but we need to postprocess because
    # we might actually keep some boxes currently marked as conflicted.
    n_conflicts = conflicting.sum(0).byte()

    if not fast:
        # It is not enought to simply use all places where there are no
        # conflicts. Say we have boxes A, B, and C, where A conflicts with B,
        # B conflicts with C but A does not conflict with C. The fact that we
        # use A should mean that C is not longer conflicted.

        if True:
            # Marginally faster. best=618.2 us
            ordered_keep = np.zeros(len(conflicting), dtype=np.uint8)
            supress = np.zeros(len(conflicting), dtype=np.bool)
            for i, row in enumerate(conflicting.cpu().numpy() > 0):
                if not supress[i]:
                    ordered_keep[i] = 1
                    supress[row] = 1
            ordered_keep = torch.ByteTensor(ordered_keep).to(tlbr.device)
        else:
            # Marginally slower: best=1.382 ms,
            n_conflicts_post = n_conflicts.cpu()
            conflicting = conflicting.cpu()

            keep_len = len(n_conflicts_post) - 1
            for i in range(1, keep_len):
                if n_conflicts_post[i] > 0:
                    n_conflicts_post -= conflicting[i]

            n_conflicts = n_conflicts_post.to(n_conflicts.device)
            ordered_keep = (n_conflicts == 0)
    else:
        # Now we can simply keep any box that has no conflicts.
        ordered_keep = (n_conflicts == 0)

    # Unsort, so keep is aligned with input boxes
    keep = ordered_keep.new(*ordered_keep.size())
    keep.scatter_(0, order, ordered_keep)
    return keep


def test_class_torch():
    import numpy as np
    import torch
    import netharn as nh
    import ubelt as ub
    # from netharn.util.nms.torch_nms import torch_nms
    # from netharn.util import non_max_supression

    thresh = .5

    num = 500
    rng = nh.util.ensure_rng(0)
    cpu_boxes = nh.util.Boxes.random(num, scale=400.0, rng=rng, format='tlbr', tensor=True)
    cpu_tlbr = cpu_boxes.to_tlbr().data
    # cpu_scores = torch.Tensor(rng.rand(len(cpu_tlbr)))
    # make all scores unique to ensure comparability
    cpu_scores = torch.Tensor(np.linspace(0, 1, len(cpu_tlbr)))
    cpu_cls = torch.LongTensor(rng.randint(0, 10, len(cpu_tlbr)))

    tlbr = cpu_boxes.to_tlbr().data.to('cuda')
    scores = cpu_scores.to('cuda')
    classes = cpu_cls.to('cuda')

    keep1 = []
    for idxs in ub.group_items(range(len(classes)), classes.cpu().numpy()).values():
        # cls_tlbr = tlbr.take(idxs, axis=0)
        # cls_scores = scores.take(idxs, axis=0)
        cls_tlbr = tlbr[idxs]
        cls_scores = scores[idxs]
        cls_keep = torch_nms(cls_tlbr, cls_scores, thresh=thresh, bias=0)
        keep1.extend(list(ub.compress(idxs, cls_keep.cpu().numpy())))
    keep1 = sorted(keep1)

    keep_ = torch_nms(tlbr, scores, classes=classes, thresh=thresh, bias=0)
    keep2 = np.where(keep_.cpu().numpy())[0].tolist()

    keep3 = nh.util.non_max_supression(tlbr.cpu().numpy(),
                                       scores.cpu().numpy(),
                                       classes=classes.cpu().numpy(),
                                       thresh=thresh, bias=0, impl='gpu')

    print(len(keep1))
    print(len(keep2))
    print(len(keep3))

    print(set(keep1) - set(keep2))
    print(set(keep2) - set(keep1))


def _benchmark():
    """
    python -m netharn.util.nms.torch_nms _benchmark --show

    SeeAlso:
        PJR Darknet NonMax supression
        https://github.com/pjreddie/darknet/blob/master/src/box.c

        Lightnet NMS
        https://gitlab.com/EAVISE/lightnet/blob/master/lightnet/data/transform/_postprocess.py#L116

    """
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
    # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, 2000]

    # max number of boxes yolo will spit out at a time
    max_boxes = 19 * 19 * 5

    xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 1500, max_boxes]
    # xdata = [10, 20, 40, 80, 100, 200, 300, 400, 500]
    xdata = [10, 100, 500]

    rng = nh.util.ensure_rng(0)

    thresh = 0.5

    for num in xdata:
        print('\n\n---- number of boxes = {} ----\n'.format(num))

        outputs = {}

        # Build random test boxes and scores
        cpu_boxes = nh.util.Boxes.random(num, scale=10.0, rng=rng, format='tlbr', tensor=True)
        cpu_tlbr = cpu_boxes.to_tlbr().data
        # cpu_scores = torch.Tensor(rng.rand(len(cpu_tlbr)))
        # make all scores unique to ensure comparability
        cpu_scores = torch.Tensor(np.linspace(0, 1, len(cpu_tlbr)))
        cpu_cls = torch.LongTensor(rng.randint(0, 10, len(cpu_tlbr)))

        # Format boxes in lightnet format
        cpu_ln_boxes = torch.cat([cpu_boxes.to_cxywh().data, cpu_scores[:, None], cpu_cls.float()[:, None]], dim=-1)

        # Move boxes to numpy
        np_tlbr = cpu_tlbr.numpy()
        np_scores = cpu_scores.numpy()
        np_cls = cpu_cls.numpy()  # NOQA

        gpu = torch.device('cuda', 0)

        measure_gpu = torch.cuda.is_available()
        measure_cpu = False or not torch.cuda.is_available()

        def _ln_output_to_keep(ln_output, ln_boxes):
            keep = []
            for row in ln_output:
                # Find the index that we kept
                idxs = np.where(np.all(np.isclose(ln_boxes, row), axis=1))[0]
                assert len(idxs) == 1
                keep.append(idxs[0])
            assert np.all(np.isclose(ln_boxes[keep], ln_output))
            return keep

        if measure_gpu:
            # Move boxes to the GPU
            gpu_tlbr = cpu_tlbr.to(gpu)
            gpu_scores = cpu_scores.to(gpu)
            gpu_cls = cpu_cls.to(gpu)  # NOQA
            gpu_ln_boxes = cpu_ln_boxes.to(gpu)

            t1 = ub.Timerit(N, bestof=bestof, label='torch(gpu)')
            for timer in t1:
                with timer:
                    keep = torch_nms(gpu_tlbr, gpu_scores, thresh=thresh)
                    torch.cuda.synchronize()
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = np.where(keep.cpu().numpy())[0]

            t1 = ub.Timerit(N, bestof=bestof, label='cython(gpu)')
            for timer in t1:
                with timer:
                    keep = non_max_supression(np_tlbr, np_scores, thresh=thresh, impl='gpu')
                    torch.cuda.synchronize()
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = sorted(keep)

            from lightnet.data.transform._postprocess import NonMaxSupression
            t1 = ub.Timerit(N, bestof=bestof, label='lightnet-slow(gpu)')
            for timer in t1:
                with timer:
                    ln_output = NonMaxSupression._nms(gpu_ln_boxes, nms_thresh=thresh, class_nms=False, fast=False)
                    torch.cuda.synchronize()
            # convert lightnet NMS output to keep for consistency
            keep = _ln_output_to_keep(ln_output, gpu_ln_boxes)
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = sorted(keep)

            if False:
                t1 = ub.Timerit(N, bestof=bestof, label='lightnet-fast(gpu)')
                for timer in t1:
                    with timer:
                        ln_output = NonMaxSupression._nms(gpu_ln_boxes, nms_thresh=thresh, class_nms=False, fast=True)
                        torch.cuda.synchronize()
                # convert lightnet NMS output to keep for consistency
                keep = _ln_output_to_keep(ln_output, gpu_ln_boxes)
                ydata[t1.label].append(t1.min())
                outputs[t1.label] = sorted(keep)

        if measure_cpu:
            t1 = ub.Timerit(N, bestof=bestof, label='torch(cpu)')
            for timer in t1:
                with timer:
                    keep = torch_nms(cpu_tlbr, cpu_scores, thresh=thresh)
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = np.where(keep.cpu().numpy())[0]

        if True:
            t1 = ub.Timerit(N, bestof=bestof, label='cython(cpu)')
            for timer in t1:
                with timer:
                    keep = non_max_supression(np_tlbr, np_scores, thresh=thresh, impl='cpu')
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = sorted(keep)

            t1 = ub.Timerit(N, bestof=bestof, label='numpy(cpu)')
            for timer in t1:
                with timer:
                    keep = non_max_supression(np_tlbr, np_scores, thresh=thresh, impl='py')
            ydata[t1.label].append(t1.min())
            outputs[t1.label] = sorted(keep)

        # Check that all kept boxes do not have more than `threshold` ious
        for key, idxs in outputs.items():
            ious = nh.util.box_ious(np_tlbr[idxs], np_tlbr[idxs])
            max_iou = (np.tril(ious) - np.eye(len(ious))).max()
            if max_iou > thresh:
                print('{} produced a bad result with max_iou={}'.format(key, max_iou))

        # Check result consistency:
        print('\nResult stats:')
        for key in sorted(outputs.keys()):
            print('    * {:<20}: num={}'.format(key, len(outputs[key])))

        print('\nResult overlaps (method1, method2: jaccard):')
        datas = []
        for k1, k2 in it.combinations(sorted(outputs.keys()), 2):
            idxs1 = set(outputs[k1])
            idxs2 = set(outputs[k2])
            jaccard = len(idxs1 & idxs2) / len(idxs1 | idxs2)
            datas.append((k1, k2, jaccard))
        datas = sorted(datas, key=lambda x: -x[2])
        for k1, k2, jaccard in datas:
            print('    * {:<20}, {:<20}: {:0.4f}'.format(k1, k2, jaccard))

    nh.util.mplutil.autompl()
    nh.util.mplutil.multi_plot(xdata, ydata, xlabel='num boxes', ylabel='seconds')
    nh.util.show_if_requested()


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.nms.torch_nms all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
