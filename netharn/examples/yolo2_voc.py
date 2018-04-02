"""
Need to compile yolo first.

Currently setup is a bit hacked.

pip install cffi
python setup.py build_ext --inplace


Conda:
    conda create --name py36 python=3.6
    conda activate py36
    pip install pytest pip -U
    conda install -c pytorch pytorch

    pip install git+https://gitlab.com/EAVISE/lightnet.git
    pip install git+https://gitlab.com/EAVISE/brambox.git

pip install git+https://gitlab.com/EAVISE/brambox.git
git clone git@gitlab.com:Erotemic/lightnet.git:dev/mine
"""
from netharn import util
from netharn.util import profiler  # NOQA
import psutil
import torch
import cv2
import ubelt as ub
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
# from netharn.models.yolo2.utils import yolo_utils as yolo_utils
from netharn.models.yolo2 import multiscale_batch_sampler
from netharn.data import voc
import netharn as nh
# from netharn import monitor
# from netharn import initializers
# from netharn.lr_scheduler import ListedLR

# from lightnet.network.loss import RegionLoss
# from lightnet.models.network_yolo import Yolo
# from netharn.models.yolo2.light_yolo import Yolo
from netharn.models.yolo2 import light_region_loss
from netharn.models.yolo2 import light_yolo


class YoloVOCDataset(voc.VOCDataset):
    """
    Extends VOC localization dataset (which simply loads the images in VOC2008
    with minimal processing) for multiscale training.

    Example:
        >>> assert len(YoloVOCDataset(split='train', years=[2007])) == 2501
        >>> assert len(YoloVOCDataset(split='test', years=[2007])) == 4952
        >>> assert len(YoloVOCDataset(split='val', years=[2007])) == 2510
        >>> assert len(YoloVOCDataset(split='trainval', years=[2007])) == 5011

        >>> assert len(YoloVOCDataset(split='train', years=[2007, 2012])) == 8218
        >>> assert len(YoloVOCDataset(split='test', years=[2007, 2012])) == 4952
        >>> assert len(YoloVOCDataset(split='val', years=[2007, 2012])) == 8333

    Example:
        >>> self = YoloVOCDataset()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)
    """

    def __init__(self, devkit_dpath=None, split='train', years=[2007, 2012],
                 base_wh=[416, 416], scales=[-3, 6], factor=32):
        super(YoloVOCDataset, self).__init__(devkit_dpath, split=split,
                                             years=years)

        # From YOLO9000.pdf:
        # With the addition of anchor boxes we changed the resolution to
        # 416×416.  Since our model downsamples by a factor of 32, we pull
        # from the following multiples of 32: {320, 352, ..., 608}.
        self.factor = factor  # downsample factor of yolo grid

        self.base_wh = np.array(base_wh, dtype=np.int)

        assert np.all(self.base_wh % self.factor == 0)

        self.multi_scale_inp_size = np.array([
            self.base_wh + (self.factor * i) for i in range(*scales)])
        self.multi_scale_out_size = self.multi_scale_inp_size // self.factor

        self.anchors = np.asarray([(1.08, 1.19), (3.42, 4.41),
                                   (6.63, 11.38), (9.42, 5.11),
                                   (16.62, 10.52)],
                                  dtype=np.float)
        self.num_anchors = len(self.anchors)
        self.augmenter = None

        if split == 'train':
            # From YOLO-V1 paper:
            #     For data augmentation we introduce random scaling and
            #     translations of up to 20% of the original image size. We
            #     also randomly adjust the exposure and saturation of the image
            #     by up to a factor of 1.5 in the HSV color space.
            # YoloV2 seems to use the same augmentation as YoloV1
            augmentors = [
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5),
                iaa.Affine(
                    scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-15, 15),
                    shear=(-7, 7),
                    order=[0, 1, 3],
                    cval=(0, 255),
                    mode=ia.ALL,
                    # use any of scikit-image's warping modes (see 2nd image
                    # from the top for examples)
                    # Note: currently requires imgaug master version
                    backend='cv2',
                ),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # improve or worsen the contrast
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            ]
            self.augmenter = iaa.Sequential(augmentors)

    def _find_anchors(self):
        """

        Example:
            >>> self = YoloVOCDataset(split='train', years=[2007])
            >>> anchors = self._find_anchors()
            >>> print('anchors = {}'.format(ub.repr2(anchors, precision=2)))
            >>> # xdoctest: +REQUIRES(--show)
            >>> xy = -anchors / 2
            >>> wh = anchors
            >>> show_boxes = np.hstack([xy, wh])
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.draw_boxes(show_boxes, box_format='xywh')
            >>> from matplotlib import pyplot as plt
            >>> plt.gca().set_xlim(xy.min() - 1, wh.max() / 2 + 1)
            >>> plt.gca().set_ylim(xy.min() - 1, wh.max() / 2 + 1)
            >>> plt.gca().set_aspect('equal')
        """
        from PIL import Image
        from sklearn import cluster
        all_norm_wh = []
        for i in ub.ProgIter(range(len(self)), desc='find anchors'):
            annots = self._load_annotation(i)
            img_wh = np.array(Image.open(self.gpaths[i]).size)
            boxes = np.array(annots['boxes'])
            box_wh = boxes[:, 2:4] - boxes[:, 0:2]
            # normalize to 0-1
            norm_wh = box_wh / img_wh
            all_norm_wh.extend(norm_wh.tolist())
        # Re-normalize to the size of the grid
        all_wh = np.array(all_norm_wh) * self.base_wh[0] / self.factor
        algo = cluster.KMeans(
            n_clusters=5, n_init=20, max_iter=10000, tol=1e-6,
            algorithm='elkan', verbose=0)
        algo.fit(all_wh)
        anchors = algo.cluster_centers_
        return anchors

    def _load_sized_image(self, index, inp_size):
        # load the raw data from VOC

        cacher = ub.Cacher('voc_img', cfgstr=ub.repr2([index, inp_size]),
                           appname='clab')
        data = cacher.tryload()
        if data is None:
            image = self._load_image(index)
            orig_size = np.array(image.shape[0:2][::-1])
            factor = inp_size / orig_size
            # squish the image into network input coordinates
            interpolation = (cv2.INTER_AREA if factor.sum() <= 2 else
                             cv2.INTER_CUBIC)
            hwc255 = cv2.resize(image, tuple(inp_size),
                                interpolation=interpolation)
            data = hwc255, orig_size, factor
            cacher.save(data)

        hwc255, orig_size, factor = data
        return hwc255, orig_size, factor

    @ub.memoize_method
    def _load_item(self, index, inp_size):
        # load the raw data from VOC
        hwc255, orig_size, factor = self._load_sized_image(index, inp_size)
        # orig_size = np.array(image.shape[0:2][::-1])
        # factor = inp_size / orig_size
        # # squish the image into network input coordinates
        # interpolation = (cv2.INTER_AREA if factor.sum() <= 2 else
        #                  cv2.INTER_CUBIC)
        # hwc255 = cv2.resize(image, tuple(inp_size),
        #                     interpolation=interpolation)

        # VOC loads annotations in tlbr
        annot = self._load_annotation(index)
        tlbr_orig = util.Boxes(annot['boxes'].astype(np.float), 'tlbr')
        gt_classes = annot['gt_classes']
        # Weight samples so we dont care about difficult cases
        gt_weights = 1.0 - annot['gt_ishard'].astype(np.float)
        # squish the bounding box into network input coordinates
        tlbr = tlbr_orig.scale(factor).data

        return hwc255, orig_size, tlbr, gt_classes, gt_weights

    @profiler.profile
    def __getitem__(self, index):
        """
        Example:
            >>> from yolo_voc2 import *
            >>> self = YoloVOCDataset(split='train')
            >>> index = 1
            >>> chw01, label = self[index]
            >>> hwc01 = chw01.numpy().transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> boxes, class_idxs = label[0:2]
            >>> # xdoc: +REQUIRES(--show)
            >>> from netharn.util import mplutil
            >>> mplutil.figure(doclf=True, fnum=1)
            >>> mplutil.qtensure()  # xdoc: +SKIP
            >>> mplutil.imshow(hwc01, colorspace='rgb')
            >>> mplutil.draw_boxes(boxes.numpy(), box_format='tlbr')

        Ignore:
            >>> from yolo_voc2 import *
            >>> # Check that we can collate this data
            >>> self = YoloVOCDataset(split='train')
            >>> inbatch = [self[index] for index in range(0, 16)]
            >>> from netharn.data import collate
            >>> batch = collate.padded_collate(inbatch)
            >>> inputs, labels = batch
            >>> assert len(labels) == len(inbatch[0][1])
            >>> target, gt_weights, origsize, index = labels
            >>> assert list(target.shape) == [16, 6, 5]
            >>> assert list(gt_weights.shape) == [16, 6]
            >>> assert list(origsize.shape) == [16, 2]
            >>> assert list(index.shape) == [16, 1]
        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, size_index = index
            inp_size = self.multi_scale_inp_size[size_index]
        else:
            inp_size = self.base_size

        (hwc255, orig_size,
         tlbr, gt_classes, gt_weights) = self._load_item(index, inp_size)

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            seq_det = self.augmenter.to_deterministic()

            hwc255 = seq_det.augment_image(hwc255)

            bbs = ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1, y1, x2, y2)
                 for x1, y1, x2, y2 in tlbr], shape=hwc255.shape)
            bbs = seq_det.augment_bounding_boxes([bbs])[0]

            tlbr = np.array([[bb.x1, bb.y1, bb.x2, bb.y2]
                             for bb in bbs.bounding_boxes])
            tlbr = util.clip_boxes(tlbr, hwc255.shape[0:2])

        # Remove boxes that are too small
        # ONLY DO THIS FOR THE SMALL DEMO TASK
        tlbr = util.Boxes(tlbr, 'tlbr')
        flags = (tlbr.area > 10).ravel()
        tlbr = tlbr.data[flags]
        gt_classes = gt_classes[flags]
        gt_weights = gt_weights[flags]

        chw01 = torch.FloatTensor(hwc255.transpose(2, 0, 1) / 255)

        # Lightnet YOLO accepts truth tensors in the format:
        # [class_id, center_x, center_y, w, h]
        # where coordinates are noramlized between 0 and 1
        tlbr_inp = util.Boxes(tlbr, 'tlbr')
        cxywh_norm = tlbr_inp.asformat('cxywh').scale(1 / inp_size)

        import utool
        with utool.embed_on_exception_context:
            datas = [gt_classes[:, None], cxywh_norm.data]
            # [d.shape for d in datas]
            target = np.concatenate(datas, axis=-1)
            target = torch.FloatTensor(target)

        # Return index information in the label as well
        orig_size = torch.LongTensor(orig_size)
        index = torch.LongTensor([index])
        gt_weights = torch.FloatTensor(gt_weights)
        label = (target, gt_weights, orig_size, index)

        return chw01, label

    def _load_image(self, index):
        return super(YoloVOCDataset, self)._load_image(index)

    @ub.memoize_method
    def _load_annotation(self, index):
        return super(YoloVOCDataset, self)._load_annotation(index)


def make_loaders(datasets, batch_size=16, workers=0):
    """
    Example:
        >>> datasets = {'train': YoloVOCDataset(split='train'),
        >>>             'vali': YoloVOCDataset(split='val')}
        >>> torch.random.manual_seed(0)
        >>> loaders = make_loaders(datasets)
        >>> train_iter = iter(loaders['train'])
        >>> # training batches should have multiple shapes
        >>> shapes = set()
        >>> for batch in train_iter:
        >>>     shapes.add(batch[0].shape[-1])
        >>>     if len(shapes) > 1:
        >>>         break
        >>> assert len(shapes) > 1

        >>> vali_loader = iter(loaders['vali'])
        >>> vali_iter = iter(loaders['vali'])
        >>> # vali batches should have one shape
        >>> shapes = set()
        >>> for batch, _ in zip(vali_iter, [1, 2, 3, 4]):
        >>>     shapes.add(batch[0].shape[-1])
        >>> assert len(shapes) == 1
    """
    from netharn.data import collate
    import torch.utils.data as torch_data
    loaders = {}
    for key, dset in datasets.items():
        assert len(dset) > 0, 'must have some data'
        # use custom sampler that does multiscale training
        batch_sampler = multiscale_batch_sampler.MultiScaleBatchSampler(
            dset, batch_size=batch_size, shuffle=(key == 'train')
        )
        loader = torch_data.DataLoader(dset, batch_sampler=batch_sampler,
                                       collate_fn=collate.padded_collate,
                                       num_workers=workers)
        loader.batch_size = batch_size
        loaders[key] = loader
    return loaders


def ensure_ulimit():
    # NOTE: It is important to have a high enought ulimit for DataParallel
    try:
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if rlimit[0] <= 8192:
            resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
    except Exception:
        print('Unable to fix ulimit. Ensure manually')
        raise


def ensure_lightnet_initial_weights():
    import os
    weight_fpath = ub.grabdata(
        'https://pjreddie.com/media/files/darknet19_448.conv.23', appname='clab')
    torch_fpath = weight_fpath + '.pt'
    if not os.path.exists(torch_fpath):
        # hack to transform initial state
        model = light_yolo.Yolo(num_classes=1000)
        model.load_weights(weight_fpath)
        torch.save(model.state_dict(), torch_fpath)
    return torch_fpath


def setup_harness(workers=None):
    """
    CommandLine:
        python ~/code/netharn/examples/yolo_voc2.py setup_harness
        python ~/code/netharn/examples/yolo_voc2.py setup_harness --profile
        python ~/code/netharn/examples/yolo_voc2.py setup_harness --flamegraph

    Example:
        >>> harn = setup_harness(workers=0)
        >>> harn.initialize()
        >>> harn.dry = True
        >>> harn.run()
    """
    workdir = ub.truepath('~/work/VOC2007')
    devkit_dpath = ub.truepath('~/data/VOC/VOCdevkit')
    YoloVOCDataset.ensure_voc_data()

    if ub.argflag('--2007'):
        dsetkw = {'years': [2007]}
    elif ub.argflag('--2012'):
        dsetkw = {'years': [2007, 2012]}
    else:
        dsetkw = {'years': [2007]}

    data_choice = ub.argval('--data', 'normal')

    if ub.argflag('--small'):
        dsetkw['base_wh'] = np.array([7, 7]) * 32
        dsetkw['scales'] = [-1, 1]

    if data_choice == 'combined':
        datasets = {
            'test': YoloVOCDataset(devkit_dpath, split='test', **dsetkw),
            'train': YoloVOCDataset(devkit_dpath, split='trainval', **dsetkw),
        }
    elif data_choice == 'notest':
        datasets = {
            'train': YoloVOCDataset(devkit_dpath, split='train', **dsetkw),
            'vali': YoloVOCDataset(devkit_dpath, split='val', **dsetkw),
        }
    elif data_choice == 'normal':
        datasets = {
            'train': YoloVOCDataset(devkit_dpath, split='train', **dsetkw),
            'vali': YoloVOCDataset(devkit_dpath, split='val', **dsetkw),
            'test': YoloVOCDataset(devkit_dpath, split='test', **dsetkw),
        }
    else:
        raise KeyError(data_choice)

    nice = ub.argval('--nice', default=None)

    pretrained_fpath = light_yolo.initial_imagenet_weights()

    # NOTE: XPU implicitly supports DataParallel just pass --gpu=0,1,2,3
    xpu = nh.XPU.cast('argv')
    print('xpu = {!r}'.format(xpu))

    ensure_ulimit()

    postproc_params = dict(
        conf_thresh=0.001,
        # nms_thresh=0.5,
        nms_thresh=0.4,
        ovthresh=0.5,
    )

    max_epoch = 160

    lr_step_points = {
        0: 0.001,
        60: 0.0001,
        90: 0.00001,
    }

    # if ub.argflag('--warmup'):
    lr_step_points = {
        # warmup learning rate
        0:  0.0001,
        1:  0.0001,
        2:  0.0002,
        3:  0.0003,
        4:  0.0004,
        5:  0.0005,
        6:  0.0006,
        7:  0.0007,
        8:  0.0008,
        9:  0.0009,
        10: 0.0010,
        # cooldown learning rate
        60: 0.0001,
        90: 0.00001,
    }

    batch_size = int(ub.argval('--batch_size', default=16))
    n_cpus = psutil.cpu_count(logical=True)
    if workers is None:
        workers = int(ub.argval('--workers', default=int(n_cpus / 2)))

    print('Making loaders')
    loaders = make_loaders(datasets, batch_size=batch_size,
                           workers=workers if workers is not None else workers)

    # anchors = {'num': 5, 'values': list(ub.flatten(datasets['train'].anchors))}
    anchors = dict(num=5, values=[1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                                  8.09892, 9.47112, 4.84053, 11.2364, 10.0071])

    print('Making hyperparams')
    hyper = nh.HyperParams(

        # model=(darknet.Darknet19, {
        model=(light_yolo.Yolo, {
            'num_classes': datasets['train'].num_classes,
            'anchors': anchors,
            'conf_thresh': postproc_params['conf_thresh'],
            'nms_thresh': postproc_params['nms_thresh'],
        }),

        criterion=(light_region_loss.RegionLoss, {
            'num_classes': datasets['train'].num_classes,
            'anchors': anchors,
            # 'object_scale': 5.0,
            # 'noobject_scale': 1.0,
            # 'class_scale': 1.0,
            # 'coord_scale': 1.0,
            # 'thresh': 0.6,
        }),

        optimizer=(torch.optim.SGD, dict(
            lr=lr_step_points[0],
            momentum=0.9,
            weight_decay=0.0005
        )),

        initializer=(nh.initializers.Pretrained, {
            'fpath': pretrained_fpath,
        }),

        scheduler=(nh.schedulers.ListedLR, dict(
            step_points=lr_step_points
        )),

        other=ub.dict_union({
            'nice': str(nice),
            'batch_size': loaders['train'].batch_sampler.batch_size,
        }, postproc_params),

        nice=nice,
        monitor=(nh.Monitor, {'min_keys': ['loss'],
                              # max_keys=['global_acc', 'class_acc'],
                              'patience': max_epoch}),

        # centering=datasets['train'].centering,
        augment=datasets['train'].augmenter,
        xpu=xpu,
        loaders=loaders,
        workdir=workdir,
    )

    harn = nh.FitHarn(
        hyper=hyper,
        # xpu=xpu, loaders=loaders, max_iter=max_epoch,
        # workdir=workdir,
    )
    harn.postproc_params = postproc_params

    @harn.set_batch_runner
    @profiler.profile
    def batch_runner(harn, inputs, labels):
        """
        Custom function to compute the output of a batch and its loss.

        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/netharn/examples')
            >>> from yolo_voc2 import *
            >>> harn = setup_harness(workers=0)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'vali')
            >>> inputs, labels = batch
            >>> criterion = harn.criterion
            >>> weights_fpath = light_yolo.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn._custom_run_batch(harn, inputs, labels)
        """
        if harn.dry:
            shape = harn.model.module.output_shape_for(inputs[0].shape)
            outputs = torch.rand(*shape)
        else:
            outputs = harn.model.forward(*inputs)

        # darknet criterion needs to know the input image shape
        # inp_size = tuple(inputs[0].shape[-2:])
        target = labels[0]

        bsize = inputs[0].shape[0]

        n_items = len(harn.loaders['train'])
        bx = harn.bxs.get('train', 0)
        seen = harn.epoch * n_items + (bx * bsize)
        loss = harn.criterion(outputs, target, seen=seen)
        return outputs, loss

    @harn.add_batch_metric_hook
    @profiler.profile
    def custom_metrics(harn, output, labels):
        metrics_dict = ub.odict()
        criterion = harn.criterion
        metrics_dict['L_bbox'] = float(criterion.loss_coord.data.cpu().numpy())
        metrics_dict['L_iou'] = float(criterion.loss_conf.data.cpu().numpy())
        metrics_dict['L_cls'] = float(criterion.loss_cls.data.cpu().numpy())
        return metrics_dict

    # Set as a harness attribute instead of using a closure
    harn.batch_confusions = []

    @harn.add_iter_callback
    @profiler.profile
    def on_batch(harn, tag, loader, bx, inputs, labels, outputs, loss):
        """
        Custom hook to run on each batch (used to compute mAP on the fly)

        Example:
            >>> harn = setup_harness(workers=0)
            >>> harn.initialize()
            >>> batch = harn._demo_batch(0, 'vali')
            >>> inputs, labels = batch
            >>> criterion = harn.criterion
            >>> loader = harn.loaders['train']
            >>> weights_fpath = light_yolo.demo_weights()
            >>> state_dict = torch.load(weights_fpath)['weights']
            >>> harn.model.module.load_state_dict(state_dict)
            >>> outputs, loss = harn._custom_run_batch(harn, inputs, labels)
            >>> tag = 'train'
            >>> on_batch(harn, tag, loader, bx, inputs, labels, outputs, loss)

        Ignore:

            >>> target, gt_weights, batch_orig_sz, batch_index = labels
            >>> bx = 0
            >>> postout = harn.model.module.postprocess(outputs.clone())
            >>> item = postout[bx].cpu().numpy()
            >>> item = item[item[:, 4] > .6]
            >>> cxywh = util.Boxes(item[..., 0:4], 'cxywh')
            >>> orig_size = batch_orig_sz[bx].numpy().ravel()
            >>> tlbr = cxywh.scale(orig_size).asformat('tlbr').data


            >>> truth_bx = target[bx]
            >>> truth_bx = truth_bx[truth_bx[:, 0] != -1]
            >>> truth_tlbr = util.Boxes(truth_bx[..., 1:5].numpy(), 'cxywh').scale(orig_size).asformat('tlbr').data

            >>> chw = inputs[0][bx].numpy().transpose(1, 2, 0)
            >>> rgb255 = cv2.resize(chw * 255, tuple(orig_size))
            >>> mplutil.figure(fnum=1, doclf=True)
            >>> mplutil.imshow(rgb255, colorspace='rgb')
            >>> mplutil.draw_boxes(tlbr, 'tlbr')
            >>> mplutil.draw_boxes(truth_tlbr, 'tlbr', color='orange')
            >>> mplutil.show_if_requested()
        """
        # Accumulate relevant outputs to measure
        target, gt_weights, batch_orig_sz, batch_index = labels
        # inp_size = inputs[0].shape[-2:][::-1]

        conf_thresh = harn.postproc_params['conf_thresh']
        nms_thresh = harn.postproc_params['nms_thresh']
        ovthresh = harn.postproc_params['ovthresh']

        if outputs is None:
            return

        get_bboxes = harn.model.module.postprocess
        get_bboxes.conf_thresh = conf_thresh
        get_bboxes.nms_thresh = nms_thresh

        postout = harn.model.module.postprocess(outputs)

        batch_pred_boxes = []
        batch_pred_scores = []
        batch_pred_cls_inds = []
        for bx, item_ in enumerate(postout):
            item = item_.cpu().numpy()
            if len(item):
                cxywh = util.Boxes(item[..., 0:4], 'cxywh')
                orig_size = batch_orig_sz[bx].cpu().numpy().ravel()
                tlbr = cxywh.scale(orig_size).asformat('tlbr').data
                batch_pred_boxes.append(tlbr)
                batch_pred_scores.append(item[..., 4])
                batch_pred_cls_inds.append(item[..., 5])
            else:
                batch_pred_boxes.append(np.empty((0, 4)))
                batch_pred_scores.append(np.empty(0))
                batch_pred_cls_inds.append(np.empty(0))

        batch_true_cls_inds = target[..., 0]
        batch_true_boxes = target[..., 1:5]

        batch_img_inds = batch_index

        y_batch = []
        for bx, index in enumerate(batch_img_inds.data.cpu().numpy().ravel()):
            pred_boxes = batch_pred_boxes[bx]
            pred_scores = batch_pred_scores[bx]
            pred_cxs = batch_pred_cls_inds[bx]

            # Group groundtruth boxes by class
            true_boxes_ = batch_true_boxes[bx].data.cpu().numpy()
            true_cxs = batch_true_cls_inds[bx].data.cpu().numpy()
            true_weights = gt_weights[bx].data.cpu().numpy()

            # Unnormalize the true bboxes back to orig coords
            orig_size = batch_orig_sz[bx]
            if len(true_boxes_):
                true_boxes = util.Boxes(true_boxes_, 'cxywh').scale(
                    orig_size).asformat('tlbr').data
                true_boxes = np.hstack([true_boxes, true_weights[:, None]])
            else:
                true_boxes = true_boxes_.reshape(-1, 4)

            y = voc.EvaluateVOC.image_confusions(true_boxes, true_cxs,
                                                 pred_boxes, pred_scores,
                                                 pred_cxs, ovthresh=ovthresh)
            y['gx'] = index
            y_batch.append(y)

        harn.batch_confusions.extend(y_batch)

    @harn.add_epoch_callback
    @profiler.profile
    def on_epoch(harn, tag, loader):
        y = pd.concat(harn.batch_confusions)
        num_classes = len(loader.dataset.label_names)

        mean_ap, ap_list = voc.EvaluateVOC.compute_map(y, num_classes)

        harn.log_value(tag + ' epoch mAP', mean_ap, harn.epoch)
        max_ap = np.nanmax(ap_list)
        harn.log_value(tag + ' epoch max-AP', max_ap, harn.epoch)
        harn.batch_confusions.clear()

    return harn


def train():
    """
    # -------
    # LOCAL
    python ~/code/netharn/netharn/examples/yolo2_voc.py train --nice=test --workers=0 --gpu=0 --batch_size=32 --data=combined --2007 --warmup --small

    """
    harn = setup_harness()
    print('Making default xpu')
    print('harn.xpu = {!r}'.format(harn.xpu))
    with harn.xpu:
        print('Running')
        harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/netharn/examples
        python -m xdoctest ~/code/netharn/examples/yolo_voc2.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
