# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os  # NOQA
from six.moves import cPickle  # NOQA
import numpy as np


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


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


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             bias=1):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # if not os.path.isdir(cachedir):
    #     os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip().split(' ')[0] for x in lines]

    # not os.path.isfile(cachefile):
    # load annots
    import ubelt as ub
    cacher = ub.Cacher('voc_cachefile', cfgstr=ub.hash_data(imagenames))
    recs = cacher.tryload()
    if recs is None:
        recs = {}
        for i, imagename in enumerate(ub.ProgIter(imagenames, desc='reading')):
            recs[imagename] = parse_rec(annopath.format(imagename))
        cacher.save(recs)
        # save
        # print('Saving cached annotations to {:s}'.format(cachefile))
        # with open(cachefile, 'w') as f:
        #     cPickle.dump(recs, f)
    # else:
    #     # load
    #     with open(cachefile, 'r') as f:
    #         recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

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
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def evaluate_model():
    """
    Try to evaulate the model using hte exact same VOC scoring metric

    import ubelt as ub
    import sys
    sys.path.append(ub.truepath('~/code/netharn/netharn/examples/tests'))
    from test_yolo import *

    REMOTE=xxx
    rsync -avPR $REMOTE:work/voc_yolo2/fit/nice/pjr_run/torch_snapshots/_epoch_00000314.pt ~/work/voc_yolo2/fit/nice/pjr_run/torch_snapshots/_epoch_00000314.pt


    mkdir -p $HOME/work/voc_yolo2/fit/nice/fixed_lrs/torch_snapshots
    rsync -avPR $REMOTE:work/voc_yolo2/fit/nice/fixed_lrs/torch_snapshots/./_epoch_00000314.pt $HOME/work/voc_yolo2/fit/nice/fixed_lrs/torch_snapshots/.
    """

    from os.path import join
    from netharn.examples.yolo_voc import YoloVOCDataset, light_yolo
    import ubelt as ub
    import netharn as nh
    # train_dpath = ub.truepath('~/work/voc_yolo2/fit/nice/pjr_run')
    # anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
    #                       (9.42, 5.11), (16.62, 10.52)], dtype=np.float)

    train_dpath = ub.truepath('~/work/voc_yolo2/fit/nice/fixed_lrs')
    anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                        (5.05587, 8.09892), (9.47112, 4.84053),
                        (11.2364, 10.0071)])

    snapshot_fpath = join(train_dpath, 'torch_snapshots', '_epoch_00000314.pt')

    model = light_yolo.Yolo(**{
        'num_classes': 20,
        'anchors': anchors,
        'conf_thresh': 0.001,
        'nms_thresh': 0.5
    })

    dataset = YoloVOCDataset(years=[2007], split='test')
    loader = dataset.make_loader(batch_size=16, num_workers=4, shuffle=False,
                                 pin_memory=True)

    xpu = nh.XPU.cast('auto')
    model = xpu.mount(model)

    snapshot = xpu.load(snapshot_fpath)
    model.load_state_dict(snapshot['model_state_dict'])

    all_postout = []
    all_labels = []

    for raw_batch in ub.ProgIter(loader, desc='predict'):
        batch_inputs, batch_labels = raw_batch
        inputs = xpu.move(batch_inputs)

        # Run data through the model
        labels = {k: xpu.move(d) for k, d in batch_labels.items()}
        outputs = model(inputs)

        # Postprocess outputs into box predictions in 01 space
        inp_size = np.array(inputs.shape[-2:][::-1])
        assert np.array_equal(inp_size, (416, 416))

        # Hack while I fix the call
        post = model.module.postprocess
        boxes = post._get_boxes(outputs.data, box_mode=2)
        boxes = [post._nms(box) for box in boxes]
        postout = [post._clip_boxes(box) for box in boxes]

        all_postout.append(postout)
        all_labels.append(batch_labels)

    # References:
    #    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py

    def asnumpy(tensor):
        return tensor.data.cpu().numpy()

    def unpack_truth(targets, gt_weights, bx):
        target = asnumpy(targets[bx]).reshape(-1, 5)
        true_cxywh = target[:, 1:5]
        true_cxs = target[:, 0]
        true_weight = asnumpy(gt_weights[bx])

        # Remove padded truth
        flags = true_cxs != -1
        true_cxywh = true_cxywh[flags]
        true_cxs = true_cxs[flags]
        true_weight = true_weight[flags]

        true_boxes = nh.util.Boxes(true_cxywh, 'cxywh')
        return true_boxes, true_cxs, true_weight

    def unpack_pred(postout, bx):
        # Unpack postprocessed predictions
        postitem = asnumpy(postout[bx])
        sboxes = postitem.reshape(-1, 6)
        pred_cxywh = sboxes[:, 0:4]
        pred_scores = sboxes[:, 4]
        pred_cxs = sboxes[:, 5].astype(np.int)

        pred_boxes = nh.util.Boxes(pred_cxywh, 'cxywh')
        return pred_boxes, pred_cxs, pred_scores

    class_to_dets = ub.ddict(list)

    for postout, labels in ub.ProgIter(list(zip(all_postout, all_labels))):
        targets = labels['targets']
        gt_weights = labels['gt_weights']
        orig_sizes = labels['orig_sizes']
        indices = labels['indices']

        bsize = len(targets)
        for bx in range(bsize):
            orig_size = asnumpy(orig_sizes[bx])
            gx = int(asnumpy(indices[bx]))

            true_boxes_norm, true_cxs, true_weight = unpack_truth(targets, gt_weights, bx)
            pred_boxes_norm, pred_cxs, pred_scores = unpack_pred(postout, bx)

            # Undo letterbox to move back to original input shapes
            letterbox = dataset.letterbox
            pred_boxes = letterbox._boxes_letterbox_invert(pred_boxes_norm.scale(inp_size), orig_size, inp_size)

            orig_w, orig_h = orig_size
            pred_boxes = pred_boxes.clip(0, 0, orig_w, orig_h)
            flags = (pred_boxes.area > 0).ravel()

            pred_scores = pred_scores[flags]
            pred_cxs = pred_cxs[flags]
            pred_boxes = pred_boxes.compress(flags)

            # --- DUMP TO VOC FORMAT ---
            for box, score, cx in zip(pred_boxes, pred_scores, pred_cxs):
                from os.path import basename, splitext
                gpath = dataset.gpaths[gx]
                imgid = splitext(basename(gpath))[0]
                class_to_dets[cx].append(list(ub.flatten([[imgid], [score], box.to_tlbr().data])))

    detpath = join(dataset.devkit_dpath, 'results', 'VOC2007', 'Main', 'comp3_det_test_{}.txt')
    for cx, dets in class_to_dets.items():
        lbl = dataset.label_names[cx]
        fpath = detpath.format(lbl)
        with open(fpath, 'w') as file:
            text = '\n'.join([' '.join(list(map(str, det))) for det in dets])
            file.write(text)

    class_aps = {}
    class_curve = {}

    import sys
    sys.path.append('/home/joncrall/code/netharn/netharn/examples/tests')
    from voc_eval_orig import voc_eval

    annopath = join(dataset.devkit_dpath, 'VOC2007', 'Annotations', '{}.xml')
    for classname in dataset.label_names:
        cachedir = None
        imagesetfile = join(dataset.devkit_dpath, 'VOC2007', 'ImageSets', 'Main', '{}_test.txt').format(classname)

        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname,
                                 cachedir, ovthresh=0.5, use_07_metric=False,
                                 bias=1)
        class_aps[classname] = ap
        class_curve[classname] = (rec, prec)

    mAP = np.mean(list(class_aps.values()))
    print('mAP = {!r}'.format(mAP))
    # I'm gettin 0.694 !? WHY? Too Low, should be in the .76ish range
    # Now I'm getting
    """
    Netharn:
        mAP = 0.7014818238794197
        'aeroplane'   : 0.71467,
        'bicycle'     : 0.79683,
        'bird'        : 0.72047,
        'boat'        : 0.58781,
        'bottle'      : 0.41089,
        'bus'         : 0.77103,
        'car'         : 0.78624,
        'cat'         : 0.86605,
        'chair'       : 0.47570,
        'cow'         : 0.72119,
        'diningtable' : 0.72532,
        'dog'         : 0.84651,
        'horse'       : 0.83273,
        'motorbike'   : 0.77517,
        'person'      : 0.72981,
        'pottedplant' : 0.38090,
        'sheep'       : 0.67960,
        'sofa'        : 0.67424,
        'train'       : 0.84316,
        'tvmonitor'   : 0.69121
    """


def evaluate_lightnet_model():
    """
    Try to evaulate the model using the exact same VOC scoring metric

    import ubelt as ub
    import sys
    sys.path.append(ub.truepath('~/code/netharn/netharn/examples/tests'))
    from test_yolo import *
    """
    import os  # NOQA
    import netharn as nh
    import ubelt as ub
    import lightnet as ln
    import torch
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))

    xpu = nh.XPU.cast('auto')

    ln_weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/final.pt')
    ln_weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')

    from netharn.models.yolo2 import light_yolo
    ln_weights_fpath = nh.models.yolo2.light_yolo.demo_voc_weights('darknet')

    # Lightnet model, postprocess, and lightnet weights
    ln_model = ln.models.Yolo(ln_test.CLASSES, ln_weights_fpath,
                              ln_test.CONF_THRESH, ln_test.NMS_THRESH)
    ln_model = xpu.move(ln_model)
    ln_model.eval()

    TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
    os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
    ln_dset = ln_test.CustomDataset(TESTFILE, ln_model)
    ln_loader = torch.utils.data.DataLoader(
        ln_dset, batch_size=8, shuffle=False, drop_last=False, num_workers=0,
        pin_memory=True, collate_fn=ln.data.list_collate,
    )

    # ----------------------
    # Postprocessing to transform yolo outputs into detections
    # Basic difference here is the implementation of NMS
    ln_postprocess = ln.data.transform.util.Compose(ln_model.postprocess.copy())

    # ----------------------
    # Define helper functions to deal with bramboxes
    detection_to_brambox = ln.data.transform.TensorToBrambox(ln_test.NETWORK_SIZE,
                                                             ln_test.LABELS)
    # hack so forward call behaves like it does in test
    ln_model.postprocess.append(detection_to_brambox)

    # ----------------------
    def img_to_box(ln_loader, boxes, offset):
        gname_lut = ln_loader.dataset.keys
        return {gname_lut[offset + k]: v for k, v in enumerate(boxes)}

    with torch.no_grad():
        anno = {}
        ln_det = {}

        moving_ave = nh.util.util_averages.CumMovingAve()

        prog = ub.ProgIter(ln_loader, desc='')
        for bx, ln_raw_batch in enumerate(prog):
            ln_raw_inputs, ln_bramboxes = ln_raw_batch

            # Convert brambox into components understood by netharn
            ln_inputs = xpu.move(ln_raw_inputs)

            ln_model.loss.seen = 1000000
            ln_outputs = ln_model._forward(ln_inputs)

            ln_loss_bram = ln_model.loss(ln_outputs, ln_bramboxes)
            moving_ave.update(ub.odict([
                ('loss_bram', float(ln_loss_bram.sum())),
            ]))

            # Display progress information
            average_losses = moving_ave.average()
            description = ub.repr2(average_losses, nl=0, precision=2, si=True)
            prog.set_description(description, refresh=False)

            # nh_outputs and ln_outputs should be the same, so no need to
            # differentiate between them here.
            ln_postout = ln_postprocess(ln_outputs.clone())

            ln_brambox_postout = detection_to_brambox([x.clone() for x in ln_postout])

            # out, loss = ln_model.forward(ln_inputs, ln_bramboxes)

            # Record data scored by brambox
            offset = len(anno)
            anno.update(img_to_box(ln_loader, ln_bramboxes, offset))
            ln_det.update(img_to_box(ln_loader, ln_brambox_postout, offset))

    import brambox.boxes as bbb
    # Compute mAP using brambox / lightnet
    pr = bbb.pr(ln_det, anno)

    ln_mAP = round(bbb.ap(*pr) * 100, 2)
    print('\nBrambox multiclass AP = {!r}'.format(ln_mAP))

    devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit/')

    # Convert bramboxes to VOC style data for eval
    class_to_dets = ub.ddict(list)
    for gpath, bb_dets in ub.ProgIter(list(ln_det.items())):

        from PIL import Image
        from os.path import join
        pil_image = Image.open(join(devkit_dpath, gpath) + '.jpg')
        img_size = pil_image.size
        bb_dets = ln.data.transform.ReverseLetterbox.apply([bb_dets], ln_test.NETWORK_SIZE, img_size)[0]
        for det in bb_dets:
            from os.path import basename, splitext
            imgid = splitext(basename(gpath))[0]
            score = det.confidence
            tlwh = [
                max(det.x_top_left, 0),
                max(det.y_top_left, 0),
                min(det.x_top_left + det.width, img_size[0]),
                min(det.y_top_left + det.height, img_size[1])]
            # See: /home/joncrall/data/VOC/VOCdevkit/devkit_doc.pdf
            # Each line is formatted as:
            # <image identifier> <confidence> <left> <top> <right> <bottom>
            class_to_dets[det.class_label].append(
                list(ub.flatten([[imgid], [score], tlwh])))

    # Calculate Original VOC measure
    from os.path import join
    results_path = join(devkit_dpath, 'results', 'VOC2007', 'Main')
    detpath = join(results_path, 'comp3_det_test_{}.txt')
    for lbl, dets in class_to_dets.items():
        fpath = detpath.format(lbl)
        with open(fpath, 'w') as file:
            text = '\n'.join([' '.join(list(map(str, det))) for det in dets])
            file.write(text)

    import sys
    sys.path.append('/home/joncrall/code/netharn/netharn/examples/tests')
    from voc_eval_orig import voc_eval

    use_07_metric = False
    bias = 1.0
    # for bias in [0.0, 1.0]:
    # for use_07_metric in [True, False]:
    class_aps = {}
    class_curve = {}
    annopath = join(devkit_dpath, 'VOC2007', 'Annotations', '{}.xml')
    for classname in ub.ProgIter(list(class_to_dets.keys())):
        cachedir = None
        imagesetfile = join(devkit_dpath, 'VOC2007', 'ImageSets', 'Main', '{}_test.txt').format(classname)

        rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname,
                                 cachedir, ovthresh=0.5, use_07_metric=use_07_metric, bias=bias)
        class_aps[classname] = ap
        class_curve[classname] = (rec, prec)

    mAP = np.mean(list(class_aps.values()))
    print('Official* bias={} {} VOC mAP = {!r}'.format(bias, '2007' if use_07_metric else '2012', mAP))
    # I get 0.71091 without 07 metric
    # I get 0.73164 without 07 metric

    """
    Lightnet:
        'aeroplane'   : 0.7738,
        'bicycle'     : 0.8326,
        'bird'        : 0.7505,
        'boat'        : 0.6202,
        'bottle'      : 0.4614,
        'bus'         : 0.8078,
        'car'         : 0.8052,
        'cat'         : 0.8857,
        'chair'       : 0.5385,
        'cow'         : 0.7768,
        'diningtable' : 0.7556,
        'dog'         : 0.8545,
        'horse'       : 0.8401,
        'motorbike'   : 0.8144,
        'person'      : 0.7608,
        'pottedplant' : 0.4640,
        'sheep'       : 0.7398,
        'sofa'        : 0.7334,
        'train'       : 0.8697,
        'tvmonitor'   : 0.7533,
    """

    """
    Official* bias=1.0 2012 VOC mAP = 0.7670408107201542
    Darknet:
        'aeroplane'   : 0.7597,
        'bicycle'     : 0.8412,
        'bird'        : 0.7705,
        'boat'        : 0.6360,
        'bottle'      : 0.4902,
        'bus'         : 0.8164,
        'car'         : 0.8444,
        'cat'         : 0.8926,
        'chair'       : 0.5902,
        'cow'         : 0.8184,
        'diningtable' : 0.7728,
        'dog'         : 0.8612,
        'horse'       : 0.8759,
        'motorbike'   : 0.8467,
        'person'      : 0.7855,
        'pottedplant' : 0.5117,
        'sheep'       : 0.7889,
        'sofa'        : 0.7584,
        'train'       : 0.8997,
        'tvmonitor'   : 0.7806,
    """

    """
                       DARKNET    LIGHTNET    NETHARN
        'aeroplane'   : 0.7597,    0.7738,    0.71467
        'bicycle'     : 0.8412,    0.8326,    0.79683
        'bird'        : 0.7705,    0.7505,    0.72047
        'boat'        : 0.6360,    0.6202,    0.58781
        'bottle'      : 0.4902,    0.4614,    0.41089
        'bus'         : 0.8164,    0.8078,    0.77103
        'car'         : 0.8444,    0.8052,    0.78624
        'cat'         : 0.8926,    0.8857,    0.86605
        'chair'       : 0.5902,    0.5385,    0.47570
        'cow'         : 0.8184,    0.7768,    0.72119
        'diningtable' : 0.7728,    0.7556,    0.72532
        'dog'         : 0.8612,    0.8545,    0.84651
        'horse'       : 0.8759,    0.8401,    0.83273
        'motorbike'   : 0.8467,    0.8144,    0.77517
        'person'      : 0.7855,    0.7608,    0.72981
        'pottedplant' : 0.5117,    0.4640,    0.38090
        'sheep'       : 0.7889,    0.7398,    0.67960
        'sofa'        : 0.7584,    0.7334,    0.67424
        'train'       : 0.8997,    0.8697,    0.84316
        'tvmonitor'   : 0.7806,    0.7533,    0.69121
        ---------------------------------------------
              mAP        0.767      0.731      0.7015
    """
