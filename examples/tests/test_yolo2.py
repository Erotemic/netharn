

def check_inconsistency():
    import netharn as nh
    import numpy as np
    import torch
    import ubelt as ub
    from netharn.models.yolo2 import light_yolo
    from netharn.models.yolo2 import light_region_loss

    yolo_voc = ub.import_module_from_path(ub.truepath('~/code/netharn/examples/yolo_voc.py'))
    xpu = nh.XPU.cast('argv')

    nice = ub.argval('--nice', default='Yolo2Baseline')
    batch_size = 8
    bstep = 8
    workers = 0
    decay = 0.0005
    lr = 0.001
    ovthresh = 0.5
    simulated_bsize = bstep * batch_size

    # We will divide the learning rate by the simulated batch size
    datasets = {
        # 'train': yolo_voc.YoloVOCDataset(years=[2007, 2012], split='trainval'),
        'test': yolo_voc.YoloVOCDataset(years=[2007], split='test'),
    }
    loaders = {
        key: dset.make_loader(batch_size=batch_size, num_workers=workers,
                              shuffle=(key == 'train'), pin_memory=True,
                              resize_rate=10 * bstep, drop_last=True)
        for key, dset in datasets.items()
    }

    if workers > 0:
        import cv2
        cv2.setNumThreads(0)

    assert simulated_bsize == 64, 'must be 64'

    lr_step_points = {
        0:   0,  # Hack to see performance before any learning
        1:   0,
        2:   lr * 1.0 / simulated_bsize,
        3:   lr * 1.0 / simulated_bsize,
    }
    max_epoch = 3

    # Anchors
    anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                        (5.05587, 8.09892), (9.47112, 4.84053),
                        (11.2364, 10.0071)])

    hyper = nh.HyperParams(**{
        'nice': nice,
        'workdir': ub.truepath('~/work/devcheck_yolo'),
        'datasets': datasets,
        'xpu': xpu,

        # a single dict is applied to all datset loaders
        'loaders': loaders,

        'model': (light_yolo.Yolo, {
            # 'num_classes': datasets['train'].num_classes,
            'num_classes': 20,
            'anchors': anchors,
            # 'conf_thresh': 0.001,
            'conf_thresh': 0.1,  # make training a bit faster
            # nms_thresh=0.5 to reproduce original yolo
            # nms_thresh=0.4 to reproduce lightnet
            'nms_thresh': 0.5 if not ub.argflag('--eav') else 0.4
        }),

        'criterion': (light_region_loss.RegionLoss, {
            # 'num_classes': datasets['train'].num_classes,
            'num_classes': 20,
            'anchors': anchors,
            'object_scale': 5.0,
            'noobject_scale': 1.0,
            'class_scale': 1.0,
            'coord_scale': 1.0,
            'thresh': 0.6,  # iou_thresh
        }),

        'initializer': (nh.initializers.Pretrained, {
            # 'fpath': light_yolo.initial_imagenet_weights(),
            'fpath': light_yolo.demo_voc_weights(),
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': lr_step_points[0],
            'momentum': 0.9,
            'dampening': 0,
            # multiplying by batch size was one of those unpublished details
            'weight_decay': decay * simulated_bsize,
        }),

        'scheduler': (nh.schedulers.core.YOLOScheduler, {
            'points': lr_step_points,
            'interpolate': True,
            'burn_in': 1,
            # 'dset_size': len(datasets['train']),  # when drop_last=False
            'dset_size': len(datasets['test']),  # when drop_last=False
            'batch_size': batch_size,
        }),

        'monitor': (nh.Monitor, {
            'minimize': ['loss'],
            'maximize': ['mAP'],
            'patience': max_epoch,
            'max_epoch': max_epoch,
        }),
        # 'augment': datasets['train'].augmenter,
        'dynamics': {'batch_step': bstep},
        'other': {
            'nice': nice,
            'ovthresh': ovthresh,
        },
    })
    print('max_epoch = {!r}'.format(max_epoch))
    harn = yolo_voc.YoloHarn(hyper=hyper)
    harn.config['use_tqdm'] = False
    harn.intervals['log_iter_train'] = None
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None

    harn.initialize()
    harn.run()


def compare_both_with_ln_weights():
    """

    * Use weights from lightnet
    * Compare the netharn dataloader and the lightnet dataloader.

    """
    import netharn as nh
    import numpy as np
    import ubelt as ub
    import os
    import copy
    import sys
    import torch
    from os.path import exists  # NOQA
    import lightnet as ln
    sys.path.append(ub.truepath('~/code/netharn/examples'))  # NOQA
    yolo_voc = ub.import_module_from_path(ub.truepath('~/code/netharn/examples/yolo_voc.py'))
    ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))

    ln_weights_fpath = yolo_voc.light_yolo.demo_voc_weights()

    # Netharn model, postprocess, and lightnet weights
    harn = yolo_voc.setup_yolo_harness(bsize=8)
    harn.initialize()
    nh_model_with_ln_weights = harn.model
    nh_model_with_ln_weights.load_state_dict({'module.' + k: v for k, v in harn.xpu.load(ln_weights_fpath)['weights'].items()})

    xpu = harn.xpu

    # Lightnet model, postprocess, and lightnet weights
    ln_model_with_ln_weights = ln.models.Yolo(ln_test.CLASSES,
                                              ln_weights_fpath,
                                              ln_test.CONF_THRESH,
                                              ln_test.NMS_THRESH)
    ln_model_with_ln_weights = xpu.move(ln_model_with_ln_weights)
    ln_model = ln_model_with_ln_weights

    def _nh_loop(harn):
        # Reset
        harn.current_tag = tag = 'test'

        dmet = harn.dmets[tag]
        dmet.pred.remove_all_annotations()
        dmet.true.remove_all_annotations()
        dmet.true._build_index()
        dmet.pred._build_index()

        moving_ave = nh.util.util_averages.CumMovingAve()
        loader = harn.loaders[tag]
        loader.num_workers = 4
        prog = ub.ProgIter(iter(loader), desc='')
        with torch.no_grad():
            for bx, batch in enumerate(prog):
                inputs, labels = harn.prepare_batch(batch)
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

                postout = harn.model.module.postprocess(outputs, nms_mode=2)

                inputs, labels = batch
                inp_size = np.array(inputs.shape[-2:][::-1])
                pred_anns = list(harn._postout_to_pred_ann(
                    inp_size, labels, postout,
                    _aidbase=len(dmet.pred.dataset['annotations']) + 1
                ))
                dmet.pred.add_annotations(pred_anns)

                true_anns = list(harn._labels_to_true_ann(
                    inp_size, labels,
                    _aidbase=len(dmet.true.dataset['annotations']) + 1
                ))
                dmet.true.add_annotations(true_anns)

            average_losses = moving_ave.average()
            print('average_losses {}'.format(ub.repr2(average_losses)))
        print('netharn voc_mAP = {}'.format(dmet.score_voc()['mAP']))
        print('netharn nh_mAP = {}'.format(dmet.score_netharn()['mAP']))
        # Reset
        dmet.pred.remove_all_annotations()
        dmet.true.remove_all_annotations()

    def _ln_loop(ln_model, xpu, harn):
        """
        Uses ln data, but nh map computation
        """
        import lightnet as ln
        ln_test = ub.import_module_from_path(ub.truepath('~/code/lightnet/examples/yolo-voc/test.py'))

        harn.current_tag = tag = 'test'

        # Keep track of NH metrics
        dmet = harn.dmets[tag]
        dmet.pred.remove_all_annotations()
        dmet.true.remove_all_annotations()
        dmet.true._build_index()
        dmet.pred._build_index()

        # Keep track of LN metrics
        anno = {}
        ln_det = {}
        resize_anno = {}
        resize_ln_det = {}

        ln_keys_to_gid = {}
        for gid, img in dmet.true.imgs.items():
            key = os.path.splitext(img['file_name'].split('VOCdevkit/')[1])[0]
            ln_keys_to_gid[key] = gid

        def brambox_to_labels(ln_loader, ln_bramboxes, inp_size, LABELS, offset=None):
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

            if offset is not None:
                # Hack to find image size, assume ordered iteration, which
                # might be true for the test set.
                orig_sizes = []
                indices = []
                for k in range(len(ln_bramboxes)):
                    key = ln_loader.dataset.keys[offset + k]
                    gid = ln_keys_to_gid[key]
                    orig_sizes += [(dmet.true.imgs[gid]['width'], dmet.true.imgs[gid]['height'])]
                    indices += [gid]
                indices = torch.FloatTensor(indices)
                orig_sizes = torch.FloatTensor(orig_sizes)
            else:
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

        def img_to_box(ln_loader, boxes, offset):
            gname_lut = ln_loader.dataset.keys
            return {gname_lut[offset + k]: v for k, v in enumerate(boxes)}

        TESTFILE = ub.truepath('~/code/lightnet/examples/yolo-voc/data/test.pkl')
        os.chdir(ub.truepath('~/code/lightnet/examples/yolo-voc/'))
        ln_dset = ln_test.CustomDataset(TESTFILE, ln_model)
        ln_loader = torch.utils.data.DataLoader(
            ln_dset, batch_size=8, shuffle=False, drop_last=False,
            num_workers=4, pin_memory=True, collate_fn=ln.data.list_collate,
        )
        detection_to_brambox = ln.data.transform.TensorToBrambox(ln_test.NETWORK_SIZE, ln_test.LABELS)

        ln.data.transform.ReverseLetterbox
        # ----------------------
        # Postprocessing to transform yolo outputs into detections
        # Basic difference here is the implementation of NMS
        ln_postprocess = ln_model.postprocess
        # ----------------------
        # ln_results = []
        moving_ave = nh.util.util_averages.CumMovingAve()
        prog = ub.ProgIter(ln_loader, desc='')
        with torch.no_grad():
            ln_loader.dataset.keys
            for bx, ln_batch in enumerate(prog):
                ln_inputs, ln_bramboxes = ln_batch

                # Convert brambox into components understood by netharn
                ln_inputs = xpu.variable(ln_inputs)

                inp_size = tuple(ln_inputs.shape[-2:][::-1])

                # hack image index (assume they are sequential)
                offset = len(anno)
                ln_labels = brambox_to_labels(ln_loader, ln_bramboxes,
                                              inp_size, ln_test.LABELS,
                                              offset=offset)

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
                # ln_results.append((ln_postout, ln_labels, inp_size))

                # Track NH stats
                pred_anns = list(harn._postout_to_pred_ann(
                    inp_size, ln_labels, ln_postout, undo_lb=False,
                    _aidbase=len(dmet.pred.dataset['annotations']) + 1))
                dmet.pred.add_annotations(pred_anns)

                true_anns = list(harn._labels_to_true_ann(
                    inp_size, ln_labels, undo_lb=False,
                    _aidbase=len(dmet.true.dataset['annotations']) + 1))
                dmet.true.add_annotations(true_anns)

                # Track LN stats
                ln_brambox_postout = detection_to_brambox([x.clone() for x in ln_postout])
                anno.update(img_to_box(ln_loader, ln_bramboxes, offset))
                ln_det.update(img_to_box(ln_loader, ln_brambox_postout, offset))

                # Also track bb-stats in original sizes
                ln_resize_annos = []
                for bb_anns, orig_size in zip(ln_bramboxes, ln_labels['orig_sizes']):
                    old_bb_anns = copy.deepcopy(bb_anns)
                    new_bb_anns = ln.data.transform.ReverseLetterbox.apply(
                        [old_bb_anns], ln_test.NETWORK_SIZE, orig_size)[0]
                    ln_resize_annos.append(new_bb_anns)

                ln_resize_dets = []
                for bb_dets, orig_size in zip(ln_brambox_postout, ln_labels['orig_sizes']):
                    old_bb_dets = copy.deepcopy(bb_dets)
                    new_bb_dets = ln.data.transform.ReverseLetterbox.apply(
                        [old_bb_dets], ln_test.NETWORK_SIZE, orig_size)[0]
                    ln_resize_dets.append(new_bb_dets)

                resize_anno.update(img_to_box(ln_loader, ln_resize_annos, offset))
                resize_ln_det.update(img_to_box(ln_loader, ln_resize_dets, offset))

        print('lightnet voc_mAP = {}'.format(dmet.score_voc()['mAP']))
        print('lightnet nh_mAP = {}'.format(dmet.score_netharn()['mAP']))

        # Compute mAP using brambox / lightnet
        import brambox.boxes as bbb
        ln_mAP = round(bbb.ap(*bbb.pr(ln_det, anno)) * 100, 2)
        print('ln_bb_mAP = {!r}'.format(ln_mAP))

        ln_resize_mAP = round(bbb.ap(*bbb.pr(resize_ln_det, resize_anno)) * 100, 2)
        print('ln_resize_bb_mAP = {!r}'.format(ln_resize_mAP))

        if False:
            # Check sizes
            for bb in resize_anno[ln_loader.dataset.keys[0]]:
                print(list(map(float, [bb.x_top_left, bb.y_top_left, bb.width, bb.height])))
            print(dmet.true.annots(gid=0).boxes)

            # Check sizes
            for bb in resize_ln_det[ln_loader.dataset.keys[0]][0:3]:
                print(list(map(float, [bb.x_top_left, bb.y_top_left, bb.width, bb.height])))
            print(dmet.pred.annots(gid=0).boxes[0:3])

    _nh_loop(harn)
    _ln_loop(ln_model, xpu, harn)
