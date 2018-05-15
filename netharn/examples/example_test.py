import netharn as nh
import numpy as np
import pandas as pd
import ubelt as ub
import torch
import sys

sys.path.append(ub.truepath('~/code/netharn/netharn/examples'))  # NOQA
# import mkinit
# exec(mkinit.dynamic_init('yolo_voc'))
from yolo_voc import setup_harness, light_yolo

"""
mkinit /code/netharn/netharn/examples/example_test.py --dry
"""

# <AUTOGEN_INIT>
# </AUTOGEN_INIT>


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
    nh_outputs = harn.model(harn.xpu.move(nh_img[None, :]))

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
        tf = ln.data.preprocess.BramboxToTensor._tf_anno(anno, inp_size, None)
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

    def brambox_to_labels(ln_bramboxes, inp_size):
        """ convert brambox to netharn style labels """
        max_anno = max(map(len, ln_bramboxes))
        ln_targets = [
            ln.data.preprocess.BramboxToTensor.apply(
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
        ln_labels = ln_targets, gt_weights, orig_sizes, indices, bg_weights
        return ln_labels

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
        ln_coco_detections = []
        nh_coco_detections0 = []

        prog = ub.ProgIter(zip(ln_loader, nh_loader), desc='')
        for bx, (ln_batch, nh_batch) in enumerate(prog):
            ln_inputs, ln_bramboxes = ln_batch
            inp_size = tuple(ln_inputs.shape[-2:][::-1])
            nh_inputs, nh_labels = nh_batch

            nh_targets = nh_labels[0]
            nh_gt_weights = nh_labels[1]

            # Convert brambox into components understood by netharn
            ln_labels = brambox_to_labels(ln_bramboxes, inp_size)
            ln_inputs = harn.xpu.variable(ln_inputs)
            ln_targets = harn.xpu.variable(ln_labels[0])
            ln_gt_weights = harn.xpu.variable(ln_labels[1])  # NOQA

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

            bg_weights = torch.FloatTensor(np.ones(len(ln_targets)))
            indices = None
            orig_sizes = None

            # Record data scored by netharn

            ln_labels = ln_targets, ln_gt_weights, orig_sizes, indices, bg_weights

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

    num_classes = len(LABELS)
    cls_labels = list(range(num_classes))

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
            from netharn.metrics.detections import _confusion_pr_ap
            prec, recall, ap2 = _confusion_pr_ap(y)
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
