import numpy as np
import ubelt as ub
from netharn.metrics.detections import _ave_precision
from netharn.metrics.detections import detection_confusions


def _multiclass_ap(y):
    """ computes pr like lightnet from netharn confusions """
    y = y.sort_values('score', ascending=False)

    num_annotations = y[y.true >= 0].weight.sum()
    positives = []

    for rx, row in y.iterrows():
        if row.pred >= 0:
            if row.true == row.pred:
                positives.append((row.score, True))
            else:
                positives.append((row.score, False))

    # sort matches by confidence from high to low
    positives = sorted(positives, key=lambda d: (d[0], -d[1]), reverse=True)

    tps = []
    fps = []
    tp_counter = 0
    fp_counter = 0

    # all matches in dataset
    for pos in positives:
        if pos[1]:
            tp_counter += 1
        else:
            fp_counter += 1
        tps.append(tp_counter)
        fps.append(fp_counter)

    precision = []
    recall = []
    for tp, fp in zip(tps, fps):
        recall.append(tp / num_annotations)
        precision.append(tp / np.maximum(fp + tp, np.finfo(np.float64).eps))

    import scipy
    num_of_samples = 100

    if len(precision) > 1 and len(recall) > 1:
        p = np.array(precision)
        r = np.array(recall)
        p_start = p[np.argmin(r)]
        samples = np.arange(0., 1., 1.0 / num_of_samples)
        interpolated = scipy.interpolate.interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
        avg = sum(interpolated) / len(interpolated)
    elif len(precision) > 0 and len(recall) > 0:
        # 1 point on PR: AP is box between (0,0) and (p,r)
        avg = precision[0] * recall[0]
    else:
        avg = float('nan')

    return np.array(precision), np.array(recall), avg


def voc_eval(lines, recs, classname, ovthresh=0.5, method=False, bias=1):
    import copy
    # imagenames = ([x.strip().split(' ')[0] for x in lines])
    imagenames = ([x[0] for x in lines])
    # BUGFIX: the original code did not cast this to a set
    imagenames = set(imagenames)
    recs2 = copy.deepcopy(recs)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs2[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    splitlines = lines
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([x[1] for x in splitlines])
    BB = np.array([[z for z in x[2:]] for x in splitlines])

    # splitlines = [x.strip().split(' ') for x in lines]
    # confidence = np.array([float(x[1]) for x in splitlines])
    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)  #
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + bias, 0.)
            ih = np.maximum(iymax - iymin + bias, 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + bias) * (bb[3] - bb[1] + bias) +
                   (BBGT[:, 2] - BBGT[:, 0] + bias) *
                   (BBGT[:, 3] - BBGT[:, 1] + bias) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    def voc_ap(rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    eav_ap = voc_ap(rec, prec, use_07_metric=method ==  'voc2007')
    ap = _ave_precision(rec, prec, method)

    assert ap == eav_ap
    return rec, prec, ap


def _devcheck_voc_consistency2():
    """
    # CHECK FOR ISSUES WITH MY MAP COMPUTATION

    TODO:
        Check how cocoeval works
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    """
    import pandas as pd
    from netharn.metrics.detections import DetectionMetrics
    xdata = []
    ydatas = ub.ddict(list)

    dmets = []

    for box_noise in np.linspace(0, 8, 20):
        dmet = DetectionMetrics.demo(
            nimgs=20,
            nboxes=(0, 20),
            nclasses=3,
            rng=0,
            # anchors=np.array([[.5, .5], [.3, .3], [.1, .3], [.2, .1]]),
            box_noise=box_noise,
            # n_fp=0 if box_noise == 0 else (0, 3),
            # n_fn=0 if box_noise == 0 else (0, 3),
            # cls_noise=0 if box_noise == 0 else .3,
        )
        dmets.append(dmet)

        nh_scores = dmet.score_netharn(bias=0)
        voc_scores = dmet.score_voc(bias=0)
        coco_scores = dmet.score_coco()
        nh_map = nh_scores['mAP']
        voc_map = voc_scores['mAP']
        coco_map = coco_scores['mAP']
        print('nh_map = {!r}'.format(nh_map))
        print('voc_map = {!r}'.format(voc_map))
        print('coco_map = {!r}'.format(coco_map))

        xdata.append(box_noise)
        ydatas['voc'].append(voc_map)
        ydatas['netharn'].append(nh_map)
        ydatas['coco'].append(coco_map)

    ydf = pd.DataFrame(ydatas)
    print(ydf)

    import kwplot
    kwplot.autompl()
    kwplot.multi_plot(xdata=xdata, ydata=ydatas, fnum=1, doclf=True)

    if False:
        dmet_ = dmets[-1]
        dmet_ = dmets[0]
        print('true = ' + ub.repr2(dmet_.true.dataset, nl=2, precision=2))
        print('pred = ' + ub.repr2(dmet_.pred.dataset, nl=2, precision=2))

        dmet = DetectionMetrics()
        for gid in range(0, 5):
            print('----')
            print('gid = {!r}'.format(gid))
            dmet.true = dmet_.true.subset([gid])
            dmet.pred = dmet_.pred.subset([gid])

            nh_scores = dmet.score_netharn(bias=0)
            voc_scores = dmet.score_voc(bias=0)
            coco_scores = dmet.score_coco()
            nh_map = nh_scores['mAP']
            voc_map = voc_scores['mAP']
            coco_map = coco_scores['mAP']
            print('nh_map = {!r}'.format(nh_map))
            print('voc_map = {!r}'.format(voc_map))
            print('coco_map = {!r}'.format(coco_map))

            print('true = ' + ub.repr2(dmet.true.dataset, nl=2))
            print('pred = ' + ub.repr2(dmet.pred.dataset, nl=2))


def _devcheck_voc_consistency():
    """
    # CHECK FOR ISSUES WITH MY MAP COMPUTATION

    TODO:
        Check how cocoeval works
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    """
    import pandas as pd
    import netharn as nh
    # method = 'voc2012'
    method = 'voc2007'

    bias = 0
    bias = 0

    # classes = [0, 1, 2]
    classes = [0]

    classname = 0
    # nimgs = 5
    # nboxes = 2
    nimgs = 5
    nboxes = 5
    nbad = 1

    bg_weight=1.0
    ovthresh=0.5
    bg_cls=-1

    xdata = []
    ydatas = ub.ddict(list)
    for noise in np.linspace(0, 5, 10):
        recs = {}
        lines = []
        confusions = []
        rng = np.random.RandomState(0)

        detmetrics = DetectionMetrics()

        true_coco = nh.data.coco_api.CocoDataset()
        pred_coco = nh.data.coco_api.CocoDataset()
        cid = true_coco.add_category('cat1')
        cid = pred_coco.add_category('cat1')
        for imgname in range(nimgs):

            # Create voc style data
            imgname = str(imgname)
            import kwimage
            true_boxes = kwimage.Boxes.random(num=nboxes, scale=100., rng=rng, format='cxywh')
            pred_boxes = true_boxes.copy()
            pred_boxes.data = pred_boxes.data.astype(np.float) + (rng.rand() * noise)
            if nbad:
                pred_boxes.data = np.vstack([
                    pred_boxes.data,
                    kwimage.Boxes.random(num=nbad, scale=100., rng=rng, format='cxywh').data])

            true_cxs = rng.choice(classes, size=len(true_boxes))
            pred_cxs = true_cxs.copy()

            change = rng.rand(len(true_cxs)) < (noise / 5)
            pred_cxs_swap = rng.choice(classes, size=len(pred_cxs))
            pred_cxs[change] = pred_cxs_swap[change]
            if nbad:
                pred_cxs = np.hstack([pred_cxs, rng.choice(classes, size=nbad)])

            np.array([0] * len(true_boxes))
            pred_cxs = np.array([0] * len(pred_boxes))

            recs[imgname] = []
            for bbox in true_boxes.to_tlbr().data:
                recs[imgname].append({
                    'bbox': bbox,
                    'difficult': False,
                    'name': classname
                })

            for bbox, score in zip(pred_boxes.to_tlbr().data, np.arange(len(pred_boxes))):
                lines.append([imgname, score] + list(bbox))
                # lines.append('{} {} {} {} {} {}'.format(imgname, score, *bbox))

            # Create MS-COCO style data
            gid = true_coco.add_image(imgname)
            gid = pred_coco.add_image(imgname)
            for bbox in true_boxes.to_xywh():
                true_coco.add_annotation(gid, cid, bbox=bbox, iscrowd=False,
                                         ignore=0, area=bbox.area[0])
            for bbox, score in zip(pred_boxes.to_xywh(), np.arange(len(pred_boxes))):
                pred_coco.add_annotation(gid, cid, bbox=bbox, iscrowd=False,
                                         ignore=0, score=score,
                                         area=bbox.area[0])

            # Create netharn style confusion data
            true_weights = np.array([1] * len(true_boxes))
            pred_scores = np.arange(len(pred_boxes))

            y = pd.DataFrame(detection_confusions(true_boxes, true_cxs,
                                                  true_weights, pred_boxes,
                                                  pred_scores, pred_cxs,
                                                  bg_weight=1.0, ovthresh=0.5,
                                                  bg_cls=-1, bias=bias))
            y['gx'] = int(imgname)
            y = (y)
            confusions.append(y)

        from pycocotools import cocoeval as coco_score
        cocoGt = true_coco._aspycoco()
        cocoDt = pred_coco._aspycoco()

        evaler = coco_score.COCOeval(cocoGt, cocoDt, iouType='bbox')
        evaler.evaluate()
        evaler.accumulate()
        evaler.summarize()
        coco_ap = evaler.stats[1]

        y = pd.concat(confusions)

        mine_ap = score_detection_assignment(y, method=method)['ap']
        voc_rec, voc_prec, voc_ap = voc_eval(lines, recs, classname,
                                             ovthresh=0.5, method=method,
                                             bias=bias)
        eav_prec, eav_rec, eav_ap1 = _multiclass_ap(y)

        eav_ap2 = _ave_precision(eav_rec, eav_prec, method=method)
        voc_ap2 = _ave_precision(voc_rec, voc_prec, method=method)

        eav_ap = eav_ap2

        print('noise = {!r}'.format(noise))
        print('mine_ap = {!r}'.format(mine_ap.values.mean()))
        print('voc_ap = {!r}'.format(voc_ap))
        print('eav_ap = {!r}'.format(eav_ap))
        print('---')
        xdata.append(noise)
        ydatas['voc'].append(voc_ap)
        ydatas['eav'].append(eav_ap)
        ydatas['netharn'].append(mine_ap.values.mean())
        ydatas['coco'].append(coco_ap)

    ydf = pd.DataFrame(ydatas)
    print(ydf)

    import kwplot
    kwplot.autompl()
    kwplot.multi_plot(xdata=xdata, ydata=ydatas, fnum=1, doclf=True)
