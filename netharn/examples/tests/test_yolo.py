import netharn as nh
import numpy as np
import pandas as pd
import ubelt as ub
import torch
# import sys
# from os.path import exists

# sys.path.append(ub.truepath('~/code/netharn/netharn/examples'))  # NOQA
# import mkinit
# exec(dynamic_init('yolo_voc'))
from yolo_voc import setup_harness, light_yolo

"""
mkinit /code/netharn/netharn/examples/example_test.py --dry
"""

# <AUTOGEN_INIT>
# </AUTOGEN_INIT>


def compare_ap_impl(**kw):
    """
    Compare computation of AP between netharn and brambox

    xdata = []
    ydatas = ub.ddict(list)
    for x in np.arange(0, 20):
        xdata.append(x)
        for k, v in compare_ap_impl(n_bad=x).items():
            ydatas[k].append(v)

    xdata = []
    ydatas = ub.ddict(list)
    for x in np.linspace(0.001, 0.1, 10):
        xdata.append(x)
        kw = {'good_perb': x}
        for k, v in compare_ap_impl(**kw).items():
            ydatas[k].append(v)

    nh.util.qtensure()
    nh.util.multi_plot(xdata, ydatas, fnum=1, doclf=True, ymax=100, ymin=0)

    compare_ap_impl(good_perb=0.1)
    """
    import netharn as nh
    rng = np.random.RandomState(0)

    NETWORK_SIZE = (1000, 1000)

    LABELS = list(map(str, range(20)))

    params = {
        'n_true': 10,
        'n_bad': 10,
        'n_missing': 1,
        'good_perb': 0.03,
    }
    params.update(**kw)

    true_boxes = nh.util.Boxes.random(params['n_true'], scale=1.0, format='cxywh', rng=rng)
    true_labels = rng.randint(0, len(LABELS), len(true_boxes.data))[:, None]

    # Perterb the truth a bit to make detections
    perb = nh.util.Boxes.random(len(true_boxes.data), scale=params['good_perb'], format='cxywh', rng=rng)

    good_det_boxes = nh.util.Boxes(true_boxes.to_cxywh().data + perb.data, 'cxywh')
    bad_det_boxes = nh.util.Boxes.random(params['n_bad'], scale=1.0, format='cxywh', rng=rng)

    good_det_boxes = good_det_boxes[params['n_missing']:]

    good_conf = np.clip(rng.randn(len(good_det_boxes.data)) / 3 + .7, 0, 1)[:, None]
    bad_conf = np.clip(rng.randn(len(bad_det_boxes.data)) / 3 + .3, 0, 1)[:, None]
    bad_labels = rng.randint(0, len(LABELS), len(bad_det_boxes.data))[:, None]

    good_pred = np.hstack([good_det_boxes.data, good_conf, true_labels[params['n_missing']:]])
    bad_pred = np.hstack([bad_det_boxes.data, bad_conf, bad_labels])

    truth = np.hstack([true_labels, true_boxes.data])
    pred = np.vstack([good_pred, bad_pred])

    def bb_map(truth, pred):
        import lightnet as ln
        import torch
        detection_to_brambox = ln.data.transform.TensorToBrambox(NETWORK_SIZE, LABELS)
        def as_annos(truth, Win=1, Hin=1):
            """
            Construct an BramBox annotation using the basic YOLO box format
            """
            from brambox.boxes.annotations import Annotation
            for true in truth:
                anno = Annotation()
                anno.class_id = true[0]
                anno.class_label = LABELS[int(true[0])]
                x_center, y_center, w, h = true[1:5]
                anno.x_top_left = (x_center - w / 2) * Win
                anno.y_top_left = (y_center - h / 2) * Hin
                anno.width, anno.height = w * Win, h * Hin
                anno.ignore = False
                yield anno

        true_bram = list(as_annos(truth, *NETWORK_SIZE))
        pred_bram = detection_to_brambox([torch.Tensor(pred)])[0]

        ln_det = {'a': pred_bram}
        anno = {'a': true_bram}

        import brambox.boxes as bbb
        detections = ln_det
        ground_truth = anno
        overlap_threshold = 0.5
        bb_precision, bb_recall = bbb.pr(detections, ground_truth, overlap_threshold)
        bb_map = round(bbb.ap(bb_precision, bb_recall) * 100, 2)
        return bb_map

    def nh_map(truth, pred, **nhkw):
        target = truth.reshape(-1, 5)
        true_cxywh   = target[:, 1:5]
        true_cxs     = target[:, 0]
        true_weight  = np.ones(len(target))

        inp_size = NETWORK_SIZE

        # Remove padded truth
        flags = true_cxs != -1
        true_cxywh  = true_cxywh[flags]
        true_cxs    = true_cxs[flags]
        true_weight = true_weight[flags]

        # how much do we care about the background in this image?
        bg_weight = 1.0

        # Unpack postprocessed predictions
        sboxes = pred.reshape(-1, 6)
        pred_cxywh = sboxes[:, 0:4]
        pred_scores = sboxes[:, 4]
        pred_cxs = sboxes[:, 5].astype(np.int)

        true_tlbr = nh.util.Boxes(true_cxywh, 'cxywh').to_tlbr()
        pred_tlbr = nh.util.Boxes(pred_cxywh, 'cxywh').to_tlbr()

        true_tlbr = true_tlbr.scale(inp_size)
        pred_tlbr = pred_tlbr.scale(inp_size)

        true_boxes = true_tlbr.data
        pred_boxes = pred_tlbr.data

        ovthresh = 0.5

        y = nh.metrics.detection_confusions(
            true_boxes=true_boxes,
            true_cxs=true_cxs,
            true_weights=true_weight,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_cxs=pred_cxs,
            bg_weight=bg_weight,
            bg_cls=-1,
            ovthresh=ovthresh,
            **nhkw,
        )
        y = pd.DataFrame(y)

        num_classes = len(LABELS)
        cls_labels = list(range(num_classes))
        aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=False)
        aps = aps.rename(dict(zip(cls_labels, LABELS)), axis=0)
        mean_ap = np.nanmean(aps['ap'])

        nh_precision, nh_recall, avg = nh.metrics.detections._multiclass_ap(y)
        map1 = round(mean_ap * 100, 2)
        map2 = round(avg * 100, 2)
        return map1, map2

    # print('bb_precision = {}'.format(ub.repr2(bb_precision, precision=2, nl=0)))
    # print('nh_precision = {}'.format(ub.repr2(nh_precision, precision=2, nl=0)))

    # print('bb_recall    = {}'.format(ub.repr2(bb_recall, precision=2, nl=0)))
    # print('nh_recall    = {}'.format(ub.repr2(nh_recall, precision=2, nl=0)))

    nhkw = {'bias': 0}  # NOQA

    data = {
        'bb': bb_map(truth, pred),
        'nh_m1': nh_map(truth, pred, bias=0.0)[0],
        'nh_m2': nh_map(truth, pred, bias=0.0)[1],
    }
    print(data)
    return data


def _compare_map():
    """
    Pascal 2007 + 2012 trainval has 16551 images
    Pascal 2007 test has 4952 images

    In Lightnet:
        One batch is 64 images, so one epoch is 16551 / 64 = 259 iterations.
        The LR says step at iteration 250, so thats just about one batch.  No
        special handling needed.

    Most recent training run gave:
        2018-06-03 00:57:31,830 : log_value(test epoch L_bbox, 0.4200094618143574, 160
        2018-06-03 00:57:31,830 : log_value(test epoch L_iou, 1.6416475874762382, 160
        2018-06-03 00:57:31,830 : log_value(test epoch L_cls, 1.3163336199137472, 160

        LightNet Results:
            TEST 30000 mAP:74.18% Loss:3.16839 (Coord:0.38 Conf:1.61 Cls:1.17)

        My Results:
            # Worse losses (due to different image loading)
            loss: 5.00 {coord: 0.69, conf: 2.05, cls: 2.26}
            mAP = 0.6227
            The MAP is quite a bit worse... Why is that?
    """
    import netharn as nh
    import ubelt as ub
    import sys
    from os.path import exists  # NOQA
    sys.path.append(ub.truepath('~/code/netharn/netharn/examples'))  # NOQA
    from yolo_voc import setup_harness, light_yolo  # NOQA
    import shutil
    import lightnet as ln
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))

    my_weights_fpath = ub.truepath('~/remote/namek/work/voc_yolo2/fit/nice/dynamic/torch_snapshots/_epoch_00000080.pt')
    my_weights_fpath = ub.truepath('~/remote/namek/work/voc_yolo2/fit/nice/dynamic/torch_snapshots/_epoch_00000160.pt')
    my_weights_fpath = ub.truepath('~/remote/namek/work/voc_yolo2/fit/nice/dynamic/torch_snapshots/_epoch_00000040.pt')
    ln_weights_fpath = ub.truepath('~/remote/namek/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')
    ln_weights_fpath = ub.truepath('~/remote/namek/code/lightnet/examples/yolo-voc/backup/weights_45000.pt')
    ln_weights_fpath = ub.truepath('~/remote/namek/code/lightnet/examples/yolo-voc/backup/final.pt')
    assert exists(my_weights_fpath)
    assert exists(ln_weights_fpath)

    ln_weights_fpath_ = ub.truepath('~/tmp/ln_weights.pt')
    my_weights_fpath_ = ub.truepath('~/tmp/my_weights.pt')

    # Move the weights to the local computer
    stamp = nh.util.CacheStamp(
        'ln_weights.stamp', product=ln_weights_fpath_, cfgstr='ln',
        dpath=ub.truepath('~/tmp'))
    if stamp.expired():
        shutil.copy2(ln_weights_fpath, ln_weights_fpath_)
        stamp.renew()

    stamp = nh.util.CacheStamp(
        'nh_weights.stamp', product=my_weights_fpath_, cfgstr='nh',
        dpath=ub.truepath('~/tmp'))
    if stamp.expired():
        shutil.copy2(my_weights_fpath, my_weights_fpath_)
        stamp.renew()

    ########
    # Create instances of netharn and lightnet YOLOv2 model
    ########

    # Netharn model, postprocess, and lightnet weights
    nh_harn = setup_harness(bsize=2)
    xpu = nh_harn.hyper.xpu = nh.XPU.cast('auto')
    nh_harn.initialize()
    my_model = nh_harn.model
    my_model.load_state_dict(nh_harn.xpu.load(my_weights_fpath)['model_state_dict'])

    # Netharn model, postprocess, and lightnet weights
    ln_harn = setup_harness(bsize=2)
    ln_harn.initialize()
    ln_weights = {'module.' + k: v for k, v in ln_harn.xpu.load(ln_weights_fpath)['weights'].items()}
    ln_harn.model.load_state_dict(ln_weights)

    # Lightnet model, postprocess, and lightnet weights
    ln_model_with_ln_weights = ln.models.Yolo(ln_test.CLASSES,
                                              ln_weights_fpath,
                                              ln_test.CONF_THRESH,
                                              ln_test.NMS_THRESH)
    ln_model_with_ln_weights = xpu.move(ln_model_with_ln_weights)

    # Lightnet model, postprocess, and netharn weights
    import copy
    ln_model_with_nh_weights = copy.deepcopy(ln_model_with_ln_weights)
    nh_weights = nh_harn.xpu.load(my_weights_fpath)['model_state_dict']
    nh_weights = {k.replace('module.', ''): v for k, v in nh_weights.items()}
    ln_model_with_nh_weights.load_state_dict(nh_weights)

    num = None
    num = 50

    # Compute brambox-style mAP on ln_model with LN and NH weights
    ln_mAP1 = _ln_data_ln_map(ln_model_with_ln_weights, xpu, num=num)
    nh_mAP1 = _ln_data_ln_map(ln_model_with_nh_weights, xpu, num=num)

    # Compute netharn-style mAP on ln_model with LN and NH weights
    ln_mAP2 = _ln_data_nh_map(ln_model_with_ln_weights, xpu, nh_harn, num=num)
    nh_mAP2 = _ln_data_nh_map(ln_model_with_nh_weights, xpu, nh_harn, num=num)

    print('\n')
    print('ln_mAP1 on ln_model with LN_weights = {!r}'.format(ln_mAP1))
    print('nh_mAP1 on ln_model with NH_weights = {!r}'.format(nh_mAP1))

    print('nh_mAP2 on ln_model with LN_weights = {:.4f}'.format(ln_mAP2))
    print('nh_mAP2 on ln_model with NH_weights = {:.4f}'.format(nh_mAP2))

    nh_mAP3, nh_aps = _nh_data_nh_map(nh_harn, num=num)
    ln_mAP3, ln_aps = _nh_data_nh_map(ln_harn, num=num)

    print('\n')
    print('nh_mAP3 on nh_model with LN_weights = {:.4f}'.format(nh_mAP3))
    print('ln_mAP3 on nh_model with NH_weights = {:.4f}'.format(ln_mAP3))

    # Method data       mAP  aero bike bird boat bottle bus  car  cat  chair cow  table dog  horse mbike person plant sheep sofa train tv
    # YOLOv2 544 07++12 73.4 86.3 82.0 74.8 59.2 51.8   79.8 76.5 90.6 52.1  78.2 58.5  89.3 82.5  83.4  81.3   49.1  77.2  62.4 83.8 68.7

    """
    BramBox MAP:
        ln_mAP1 on ln_model with LN_weights = 66.27
        ln_mAP1 on ln_model with NH_weights = 62.2

        Shows about a 4 point difference in mAP, which is not too bad considering
        that the lightnet data loader is slightly different than the netharn data
        loader. I think letterboxes are computed differently.

    NetHarn MAP:
        My map computation seems systematically off

        nh_mAP2 on ln_model with LN_weights = 0.5437
        nh_mAP2 on ln_model with NH_weights = 0.4991

    NetHarn Model And AP:
        nh_mAP3 on nh_model with LN_weights = 0.6890
        ln_mAP3 on nh_model with NH_weights = 0.7296

        It looks like the score from EAVISE was in AP not mAP.
        I'm still getting a difference in 4 score points, but this gap is
        more reasonable. The AP computation is probably correct, given that it
        produces the same results on dummy data.

        The difference now may be the learning rates.
        Perhaps doing the warmup run will actually give me desired results.
    """


def _ln_data_ln_map(ln_model, xpu, num=None):
    """
    Compute the results on the ln test set using a ln model.
    Weights can either be lightnet or netharn
    """
    import os
    import lightnet as ln
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))

    TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
    os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
    ln_dset = ln_test.CustomDataset(TESTFILE, ln_model)
    ln_loader = torch.utils.data.DataLoader(
        ln_dset, batch_size=2, shuffle=False, drop_last=False, num_workers=0,
        pin_memory=True, collate_fn=ln.data.list_collate,
    )

    # ----------------------
    # Postprocessing to transform yolo outputs into detections
    # Basic difference here is the implementation of NMS
    ln_postprocess = ln_model.postprocess

    # ----------------------
    # Define helper functions to deal with bramboxes
    detection_to_brambox = ln.data.transform.TensorToBrambox(ln_test.NETWORK_SIZE,
                                                             ln_test.LABELS)

    # ----------------------
    def img_to_box(boxes, offset):
        gname_lut = ln_loader.dataset.keys
        return {gname_lut[offset + k]: v for k, v in enumerate(boxes)}

    with torch.no_grad():
        anno = {}
        ln_det = {}

        moving_ave = nh.util.util_averages.CumMovingAve()

        prog = ub.ProgIter(ln_loader, desc='')
        for bx, ln_batch in enumerate(prog):
            ln_inputs, ln_bramboxes = ln_batch

            # Convert brambox into components understood by netharn
            ln_inputs = xpu.variable(ln_inputs)

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

            # Record data scored by brambox
            offset = len(anno)
            anno.update(img_to_box(ln_bramboxes, offset))
            ln_det.update(img_to_box(ln_brambox_postout, offset))

            if num is not None and bx >= num:
                break

    import brambox.boxes as bbb
    # Compute mAP using brambox / lightnet
    ln_mAP = round(bbb.ap(*bbb.pr(ln_det, anno)) * 100, 2)
    print('\nln_mAP = {!r}'.format(ln_mAP))

    return ln_mAP


def _ln_data_nh_map(ln_model, xpu, harn, num=None):
    """
    Uses ln data, but nh map computation
    """
    import os
    import lightnet as ln
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))

    TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
    os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
    ln_dset = ln_test.CustomDataset(TESTFILE, ln_model)
    ln_loader = torch.utils.data.DataLoader(
        ln_dset, batch_size=2, shuffle=False, drop_last=False, num_workers=0,
        pin_memory=True, collate_fn=ln.data.list_collate,
    )

    # ----------------------
    # Postprocessing to transform yolo outputs into detections
    # Basic difference here is the implementation of NMS
    ln_postprocess = ln_model.postprocess

    # ----------------------
    with torch.no_grad():
        ln_results = []

        moving_ave = nh.util.util_averages.CumMovingAve()

        prog = ub.ProgIter(ln_loader, desc='')
        for bx, ln_batch in enumerate(prog):
            ln_inputs, ln_bramboxes = ln_batch

            # Convert brambox into components understood by netharn
            ln_inputs = xpu.variable(ln_inputs)

            inp_size = tuple(ln_inputs.shape[-2:][::-1])
            ln_labels = brambox_to_labels(ln_bramboxes, inp_size, ln_test.LABELS)

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

            ln_results.append((ln_postout, ln_labels, inp_size))

            if num is not None and bx >= num:
                break

    batch_confusions = []
    kw = dict(bias=0, PREFER_WEIGHTED_TRUTH=False)
    for ln_postout, ln_labels, inp_size in ln_results:
        for y in harn._measure_confusion(ln_postout, ln_labels, inp_size, **kw):
            batch_confusions.append(y)

    y = pd.concat([pd.DataFrame(c) for c in batch_confusions])

    num_classes = len(ln_test.LABELS)
    cls_labels = list(range(num_classes))

    precision, recall, mean_ap = nh.metrics.detections._multiclass_ap(y)

    aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=True)
    aps = aps.rename(dict(zip(cls_labels, ln_test.LABELS)), axis=0)
    # mean_ap = np.nanmean(aps['ap'])

    return mean_ap


def _nh_data_nh_map(harn, num=10):
    with torch.no_grad():
        postprocess = harn.model.module.postprocess
        # postprocess.conf_thresh = 0.001
        # postprocess.nms_thresh = 0.5
        batch_confusions = []
        moving_ave = nh.util.util_averages.CumMovingAve()
        loader = harn.loaders['test']
        prog = ub.ProgIter(iter(loader), desc='')
        for bx, batch in enumerate(prog):
            inputs, labels = harn.prepare_batch(batch)
            inp_size = np.array(inputs.shape[-2:][::-1])
            outputs = harn.model(inputs)

            loss = harn.criterion(outputs, labels['targets'],
                                  gt_weights=labels['gt_weights'],
                                  seen=1000000000)
            moving_ave.update(ub.odict([
                ('loss', float(loss.sum())),
                ('coord', harn.criterion.loss_coord),
                ('conf', harn.criterion.loss_conf),
                ('cls', harn.criterion.loss_cls),
            ]))

            average_losses = moving_ave.average()
            desc = ub.repr2(average_losses, nl=0, precision=2, si=True)
            prog.set_description(desc, refresh=False)

            postout = postprocess(outputs)
            for y in harn._measure_confusion(postout, labels, inp_size):
                batch_confusions.append(y)

            # batch_output.append((outputs.cpu().data.numpy().copy(), inp_size))
            # batch_labels.append([x.cpu().data.numpy().copy() for x in labels])
            if num is not None and bx >= num:
                break

        average_losses = moving_ave.average()
        print('average_losses {}'.format(ub.repr2(average_losses)))

    if False:
        from netharn.util import mplutil
        mplutil.qtensure()  # xdoc: +SKIP
        harn.visualize_prediction(batch, outputs, postout, thresh=.1)

    y = pd.concat([pd.DataFrame(c) for c in batch_confusions])
    precision, recall, ap = nh.metrics.detections._multiclass_ap(y)

    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))
    num_classes = len(ln_test.LABELS)
    cls_labels = list(range(num_classes))

    aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=True)
    aps = aps.rename(dict(zip(cls_labels, ln_test.LABELS)), axis=0)
    # return ap
    return ap, aps


def brambox_to_labels(ln_bramboxes, inp_size, LABELS):
    """ convert brambox to netharn style labels """
    import lightnet as ln
    max_anno = max(map(len, ln_bramboxes))
    ln_targets = [
        ln.data.transform.BramboxToTensor.apply(
            annos, inp_size, max_anno=max_anno, class_label_map=LABELS)
        for annos in ln_bramboxes]
    ln_targets = torch.stack(ln_targets)

    gt_weights = -np.ones((len(ln_bramboxes), max_anno), dtype=np.float32)
    for i, annos in enumerate(ln_bramboxes):
        weights = 1.0 - np.array([anno.ignore for anno in annos], dtype=np.float32)
        gt_weights[i, 0:len(annos)] = weights
    gt_weights = torch.Tensor(gt_weights)

    bg_weights = torch.FloatTensor(np.ones(len(ln_targets)))
    indices = None
    orig_sizes = None
    ln_labels = {
        'targets': ln_targets,
        'gt_weights': gt_weights,
        'orig_sizes': orig_sizes,
        'indices': indices,
        'bg_weights': bg_weights,
    }
    return ln_labels


def _run_quick_test():
    harn = setup_harness(bsize=2)
    harn.hyper.xpu = nh.XPU(0)
    harn.initialize()

    if 0:
        # Load up pretrained VOC weights
        weights_fpath = light_yolo.demo_voc_weights()
        state_dict = harn.xpu.load(weights_fpath)['weights']
        harn.model.module.load_state_dict(state_dict)
    else:
        weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')
        state_dict = harn.xpu.load(weights_fpath)['weights']
        harn.model.module.load_state_dict(state_dict)

    harn.model.eval()

    with torch.no_grad():
        postprocess = harn.model.module.postprocess
        # postprocess.conf_thresh = 0.001
        # postprocess.nms_thresh = 0.5
        batch_confusions = []
        moving_ave = nh.util.util_averages.CumMovingAve()
        loader = harn.loaders['test']
        prog = ub.ProgIter(iter(loader), desc='')
        for batch in prog:
            inputs, labels = harn.prepare_batch(batch)
            inp_size = np.array(inputs.shape[-2:][::-1])
            outputs = harn.model(inputs)

            target, gt_weights, orig_sizes, indices, bg_weights = labels
            loss = harn.criterion(outputs, target, gt_weights=gt_weights,
                                  seen=1000000000)
            moving_ave.update(ub.odict([
                ('loss', float(loss.sum())),
                ('coord', harn.criterion.loss_coord),
                ('conf', harn.criterion.loss_conf),
                ('cls', harn.criterion.loss_cls),
            ]))

            average_losses = moving_ave.average()
            desc = ub.repr2(average_losses, nl=0, precision=2, si=True)
            prog.set_description(desc, refresh=False)

            postout = postprocess(outputs)
            for y in harn._measure_confusion(postout, labels, inp_size):
                batch_confusions.append(y)

            # batch_output.append((outputs.cpu().data.numpy().copy(), inp_size))
            # batch_labels.append([x.cpu().data.numpy().copy() for x in labels])

        average_losses = moving_ave.average()
        print('average_losses {}'.format(ub.repr2(average_losses)))

    # batch_confusions = []
    # for (outputs, inp_size), labels in ub.ProgIter(zip(batch_output, batch_labels), total=len(batch_labels)):
    #     labels = [torch.Tensor(x) for x in labels]
    #     outputs = torch.Tensor(outputs)
    #     postout = postprocess(outputs)
    #     for y in harn._measure_confusion(postout, labels, inp_size):
    #         batch_confusions.append(y)

    if False:
        from netharn.util import mplutil
        mplutil.qtensure()  # xdoc: +SKIP
        harn.visualize_prediction(batch, outputs, postout, thresh=.1)

    y = pd.concat([pd.DataFrame(c) for c in batch_confusions])
    # TODO: write out a few visualizations
    num_classes = len(loader.dataset.label_names)
    cls_labels = list(range(num_classes))

    aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=True)
    aps = aps.rename(dict(zip(cls_labels, loader.dataset.label_names)), axis=0)
    mean_ap = np.nanmean(aps['ap'])
    max_ap = np.nanmax(aps['ap'])
    print(aps)
    print('mean_ap = {!r}'.format(mean_ap))
    print('max_ap = {!r}'.format(max_ap))

    aps = nh.metrics.ave_precisions(y[y.score > .01], cls_labels, use_07_metric=True)
    aps = aps.rename(dict(zip(cls_labels, loader.dataset.label_names)), axis=0)
    mean_ap = np.nanmean(aps['ap'])
    max_ap = np.nanmax(aps['ap'])
    print(aps)
    print('mean_ap = {!r}'.format(mean_ap))
    print('max_ap = {!r}'.format(max_ap))

    # import sklearn.metrics
    # sklearn.metrics.accuracy_score(y.true, y.pred)
    # sklearn.metrics.precision_score(y.true, y.pred, average='weighted')


def compare_loss():
    harn = setup_harness(bsize=2)
    harn.hyper.xpu = nh.XPU(0)
    harn.initialize()

    weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')

    # Load weights into a lightnet model
    import lightnet as ln
    import os
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))
    ln_net = ln.models.Yolo(ln_test.CLASSES, weights_fpath,
                            ln_test.CONF_THRESH, ln_test.NMS_THRESH)
    ln_net = harn.xpu.move(ln_net)
    os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
    TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
    ln_dset = ln_test.CustomDataset(TESTFILE, ln_net)

    # Load weights into a netharn model
    state_dict = harn.xpu.load(weights_fpath)['weights']
    harn.model.module.load_state_dict(state_dict)

    ln_img, ln_label = ln_dset[0]
    nh_img, nh_label = harn.datasets['test'][0]
    nh_targets = nh_label[0][None, :]
    ln_targets = [ln_label]

    # Test model forward is the same for my image
    ln_outputs = ln_net._forward(harn.xpu.move(nh_img[None, :]))
    # nh_outputs = harn.model(harn.xpu.move(nh_img[None, :]))

    seen = ln_net.loss.seen = 99999999
    ln_loss = ln_net.loss(ln_outputs, nh_targets)
    nh_loss = harn.criterion(ln_outputs, nh_targets, seen=seen)
    print('nh_loss = {!r}'.format(nh_loss))
    print('ln_loss = {!r}'.format(ln_loss))

    ln_brambox_loss = ln_net.loss(ln_outputs, ln_targets)
    print('ln_brambox_loss = {!r}'.format(ln_brambox_loss))

    inp_size = tuple(nh_img.shape[-2:][::-1])

    ln_tf_target = []
    for anno in ln_targets[0]:
        anno.class_label = anno.class_id
        tf = ln.data.transform.BramboxToTensor._tf_anno(anno, inp_size, None)
        ln_tf_target.append(tf)

    ln_boxes = nh.util.Boxes(np.array(ln_tf_target)[:, 1:], 'cxywh').scale(inp_size)
    nh_boxes = nh.util.Boxes(np.array(nh_targets[0])[:, 1:], 'cxywh').scale(inp_size)

    nh.util.imshow(ln_img.numpy(), colorspace='rgb', fnum=1)
    nh.util.draw_boxes(ln_boxes, color='blue')
    nh.util.draw_boxes(nh_boxes, color='red')


def _test_with_lnstyle_data():
    """
    Uses pretrained lightnet weights, and the lightnet data loader.

    Uses my critrion and net implementations.
    (already verified to produce the same outputs)

    Checks to see if my loss and map calcluations are the same as lightnet
    CommandLine:
        python ~/code/netharn/netharn/examples/yolo_voc.py _test_with_lnstyle_data

    Using LighNet Trained Weights:

        LightNet Results:
            TEST 30000 mAP:74.18% Loss:3.16839 (Coord:0.38 Conf:1.61 Cls:1.17)

        My Results:
            # Worse losses (due to different image loading)
            loss: 5.00 {coord: 0.69, conf: 2.05, cls: 2.26}
            mAP = 0.6227
            The MAP is quite a bit worse... Why is that?

    USING THE SAME WEIGHTS! I must be doing something wrong.

        Results using the same Data Loader:
            {cls: 2.22, conf: 2.05, coord: 0.65, loss: 4.91}

            # Checking with extra info
            {loss_bram: 3.17, loss_ten1: 4.91, loss_ten2: 4.91}

        OH!, Is is just that BramBox has an ignore function?
            - [X] Add ignore / weight to tensor version to see if its the same
            YUP! {loss_bram: 3.17, loss_ten1: 4.91, loss_ten2: 4.91, nh_unweighted: 4.92, nh_weighted: 3.16}

    TO CHECK:
        - [ ] why is the loss different?
            - [X] network input size is 416 in both
            - [x] networks output the same data given the same input
            - [x] loss outputs the same data given the same input (they do if seen is the same)
            - [x] CONCLUSION: The loss is not actually different


        - [ ] why is the mAP different?
            - [x] does brambox compute AP differently?
                ... YES
            CONCLUSION: several factors are at play
                * brambox has a different AP computation
                * netharn and lightnet non-max supressions are different
                * The NMS seems to have the most significant impact

    # """
    import brambox.boxes as bbb
    CHECK_SANITY = False

    # Path to weights that we will be testing
    # (These were trained using the lightnet harness and achived a good mAP)
    weights_fpath = ub.truepath('~/code/lightnet/examples/yolo-voc/backup/weights_30000.pt')

    # Load weights into a netharn model
    harn = setup_harness(bsize=2)
    harn.hyper.xpu = nh.XPU(0)
    harn.initialize()
    state_dict = harn.xpu.load(weights_fpath)['weights']
    harn.model.module.load_state_dict(state_dict)
    harn.model.eval()
    nh_net = harn.model.module

    # Load weights into a lightnet model
    import os
    import lightnet as ln
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))
    ln_net = ln.models.Yolo(ln_test.CLASSES, weights_fpath,
                            ln_test.CONF_THRESH, ln_test.NMS_THRESH)
    ln_net = harn.xpu.move(ln_net)
    ln_net.eval()

    # Sanity check, the weights should be the same
    if CHECK_SANITY:
        state1 = nh_net.state_dict()
        state2 = ln_net.state_dict()
        assert state1.keys() == state2.keys()
        for k in state1.keys():
            v1 = state1[k]
            v2 = state2[k]
            assert np.all(v1 == v2)

    # Create a lightnet dataset loader
    TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
    os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
    ln_dset = ln_test.CustomDataset(TESTFILE, ln_net)
    ln_loader = torch.utils.data.DataLoader(
        ln_dset, batch_size=2, shuffle=False, drop_last=False, num_workers=0,
        pin_memory=True, collate_fn=ln.data.list_collate,
    )

    # Create a netharn dataset loader
    nh_loader = harn.loaders['test']

    # ----------------------
    # Postprocessing to transform yolo outputs into detections
    # Basic difference here is the implementation of NMS
    ln_postprocess = ln_net.postprocess
    nh_postprocess = harn.model.module.postprocess

    # ----------------------
    # Define helper functions to deal with bramboxes
    LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
              'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']
    NETWORK_SIZE = (416, 416)
    detection_to_brambox = ln.data.TensorToBrambox(NETWORK_SIZE, LABELS)

    # ----------------------
    CHECK_LOSS = False

    with torch.no_grad():

        # Track netharn and lightnet results that will be scored
        ln_batch_confusions = []
        nh_batch_confusions = []
        nh_batch_confusions0 = []

        ln_results = []
        nh_results = []
        nh_results0 = []

        anno = {}
        ln_det = {}
        nh_det = {}
        nh_det0 = {}

        moving_ave = nh.util.util_averages.CumMovingAve()

        coco_truth = []
        # ln_coco_detections = []
        nh_coco_detections0 = []

        prog = ub.ProgIter(zip(ln_loader, nh_loader), desc='')
        for bx, (ln_batch, nh_batch) in enumerate(prog):
            ln_inputs, ln_bramboxes = ln_batch
            inp_size = tuple(ln_inputs.shape[-2:][::-1])
            nh_inputs, nh_labels = nh_batch

            nh_targets = nh_labels['targets']
            nh_gt_weights = nh_labels['gt_weights']

            # Convert brambox into components understood by netharn
            ln_labels = brambox_to_labels(ln_bramboxes, inp_size, ln_test.LABELS)
            ln_inputs = harn.xpu.variable(ln_inputs)
            ln_targets = harn.xpu.variable(ln_labels['targets'])
            ln_gt_weights = harn.xpu.variable(ln_labels['gt_weights'])  # NOQA

            ln_net.loss.seen = 1000000
            ln_outputs = ln_net._forward(ln_inputs)

            if CHECK_SANITY:
                nh_outputs = harn.model(ln_inputs)
                assert np.all(nh_outputs == ln_outputs)

            ln_loss_bram = ln_net.loss(ln_outputs, ln_bramboxes)
            moving_ave.update(ub.odict([
                ('loss_bram', float(ln_loss_bram.sum())),
            ]))

            if CHECK_LOSS:
                seen = ln_net.loss.seen
                ln_loss_ten1 = harn.criterion(ln_outputs, ln_targets,
                                              seen=seen)
                ln_loss_ten2 = ln_net.loss(ln_outputs, ln_targets)

                nh_weighted = harn.criterion(ln_outputs, nh_targets,
                                             gt_weights=nh_gt_weights,
                                             seen=seen)
                nh_unweighted = harn.criterion(ln_outputs, nh_targets,
                                               seen=seen)

                moving_ave.update(ub.odict([
                    ('loss_ten1', float(ln_loss_ten1.sum())),
                    ('loss_ten2', float(ln_loss_ten2.sum())),
                    ('nh_weighted', float(nh_weighted.sum())),
                    ('nh_unweighted', float(nh_unweighted.sum())),
                    # ('coord', harn.criterion.loss_coord),
                    # ('conf', harn.criterion.loss_conf),
                    # ('cls', harn.criterion.loss_cls),
                ]))
            # Display progress information
            average_losses = moving_ave.average()
            description = ub.repr2(average_losses, nl=0, precision=2, si=True)
            prog.set_description(description, refresh=False)

            # nh_outputs and ln_outputs should be the same, so no need to
            # differentiate between them here.
            ln_postout = ln_postprocess(ln_outputs.clone())
            nh_postout = nh_postprocess(ln_outputs.clone())

            # Should use the original NMS strategy
            nh_postout0 = nh_postprocess(ln_outputs.clone(), mode=0)

            ln_brambox_postout = detection_to_brambox([x.clone() for x in ln_postout])
            nh_brambox_postout = detection_to_brambox([x.clone() for x in nh_postout])
            nh_brambox_postout0 = detection_to_brambox([x.clone() for x in nh_postout0])

            # Record data scored by brambox
            offset = len(anno)
            def img_to_box(boxes, offset):
                gname_lut = ln_loader.dataset.keys
                return {gname_lut[offset + k]: v for k, v in enumerate(boxes)}
            anno.update(img_to_box(ln_bramboxes, offset))
            ln_det.update(img_to_box(ln_brambox_postout, offset))
            nh_det.update(img_to_box(nh_brambox_postout, offset))
            nh_det0.update(img_to_box(nh_brambox_postout0, offset))

            # Record data scored by netharn
            ln_results.append((ln_postout, ln_labels, inp_size))
            nh_results.append((nh_postout, nh_labels, inp_size))
            nh_results0.append((nh_postout0, nh_labels, inp_size))

            # preds, truths = harn._postout_to_coco(ln_postout, ln_labels, inp_size)
            # ln_coco_detections.append(preds)

            preds, truths = harn._postout_to_coco(nh_postout0, nh_labels, inp_size)
            nh_coco_detections0.append(preds)
            coco_truth.append(truths)

            # kw = dict(bias=0)
            # for y in harn._measure_confusion(ln_postout, ln_labels, inp_size, **kw):
            #     ln_batch_confusions.append(y)

            # for y in harn._measure_confusion(nh_postout, nh_labels, inp_size, **kw):
            #     nh_batch_confusions.append(y)

            # for y in harn._measure_confusion(nh_postout0, nh_labels, inp_size, **kw):
            #     nh_batch_confusions0.append(y)

            if bx > 50:
                break

    # Compute mAP using brambox / lightnet
    ln_mAP = round(bbb.ap(*bbb.pr(ln_det, anno)) * 100, 2)
    nh_mAP = round(bbb.ap(*bbb.pr(nh_det, anno)) * 100, 2)
    nh_mAP0 = round(bbb.ap(*bbb.pr(nh_det0, anno)) * 100, 2)
    print('\n----')
    print('ln_mAP = {!r}'.format(ln_mAP))
    print('nh_mAP = {!r}'.format(nh_mAP))
    print('nh_mAP0 = {!r}'.format(nh_mAP0))

    # num_classes = len(LABELS)
    # cls_labels = list(range(num_classes))

    # # Compute mAP using netharn
    # if False:
    #     is_tp = (y.true == y.pred) & (y.pred >= 0)
    #     is_fp = (y.true != y.pred) & (y.pred >= 0)
    #     rest = ~(is_fp | is_tp)

    #     y.true[is_tp] = 1
    #     y.pred[is_tp] = 1

    #     y.true[is_fp] = 0
    #     y.pred[is_fp] = 1

    #     y.true[rest] = 0
    #     y.pred[rest] = 0

    #     import sklearn
    #     import sklearn.metrics
    #     sklearn.metrics.average_precision_score(y.true, y.score, 'weighted',
    #                                             y.weight)

    for bias in [0, 1]:
        ln_batch_confusions = []
        nh_batch_confusions = []
        nh_batch_confusions0 = []
        print('\n\n======\n\nbias = {!r}'.format(bias))
        kw = dict(bias=bias, PREFER_WEIGHTED_TRUTH=False)
        for ln_postout, ln_labels, inp_size in ln_results:
            for y in harn._measure_confusion(ln_postout, ln_labels, inp_size, **kw):
                ln_batch_confusions.append(y)

        for nh_postout, nh_labels, inp_size in nh_results:
            for y in harn._measure_confusion(nh_postout, nh_labels, inp_size, **kw):
                nh_batch_confusions.append(y)

        for nh_postout0, nh_labels, inp_size in nh_results0:
            for y in harn._measure_confusion(nh_postout0, nh_labels, inp_size, **kw):
                nh_batch_confusions0.append(y)

        confusions = {
            'lh': ln_batch_confusions,
            # 'nh': nh_batch_confusions,
            'nh0': nh_batch_confusions0,
        }

        for lbl, batch_confusions in confusions.items():
            print('----')
            print('\nlbl = {!r}'.format(lbl))
            y = pd.concat([pd.DataFrame(c) for c in batch_confusions])

            # aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=True)
            # aps = aps.rename(dict(zip(cls_labels, LABELS)), axis=0)
            # mean_ap = np.nanmean(aps['ap'])
            # print('mean_ap_07 = {:.2f}'.format(mean_ap * 100))

            # aps = nh.metrics.ave_precisions(y, cls_labels, use_07_metric=False)
            # aps = aps.rename(dict(zip(cls_labels, LABELS)), axis=0)
            # mean_ap = np.nanmean(aps['ap'])
            # print('mean_ap_12 = {:.2f}'.format(mean_ap * 100))

            # Try the other way
            from netharn.metrics.detections import _multiclass_ap
            prec, recall, ap2 = _multiclass_ap(y)
            print('ap2 = {!r}'.format(round(ap2 * 100, 2)))

            # max_ap = np.nanmax(aps['ap'])
            # print(aps)
            # print('max_ap = {!r}'.format(max_ap))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/example_test.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
