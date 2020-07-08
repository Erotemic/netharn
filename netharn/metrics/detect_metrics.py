"""
DEPRECATED

USE kwcoco.metrics instead!
"""
import numpy as np
import ubelt as ub
import networkx as nx
# from .assignment import _assign_confusion_vectors
from netharn.metrics.confusion_vectors import ConfusionVectors
from netharn.metrics.assignment import _assign_confusion_vectors


class DetectionMetrics(ub.NiceRepr):
    """
    Attributes:
        gid_to_true_dets (Dict): maps image ids to truth
        gid_to_pred_dets (Dict): maps image ids to predictions
        classes (CategoryTree): category coder

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=100, nboxes=(0, 3), n_fp=(0, 1), nclasses=8, score_noise=0.9, hacked=False)
        >>> print(dmet.score_netharn(bias=0, compat='mutex', prioritize='iou')['mAP'])
        ...
        >>> # NOTE: IN GENERAL NETHARN AND VOC ARE NOT THE SAME
        >>> print(dmet.score_voc(bias=0)['mAP'])
        0.8582...
        >>> #print(dmet.score_coco()['mAP'])
    """
    def __init__(dmet, classes=None):
        dmet.classes = classes
        dmet.gid_to_true_dets = {}
        dmet.gid_to_pred_dets = {}
        dmet._imgname_to_gid = {}

    def clear(dmet):
        dmet.gid_to_true_dets = {}
        dmet.gid_to_pred_dets = {}
        dmet._imgname_to_gid = {}

    def __nice__(dmet):
        info = {
            'n_true_imgs': len(dmet.gid_to_true_dets),
            'n_pred_imgs': len(dmet.gid_to_pred_dets),
            'n_true_anns': sum(map(len, dmet.gid_to_true_dets.values())),
            'n_pred_anns': sum(map(len, dmet.gid_to_pred_dets.values())),
            'classes': dmet.classes,
        }
        return ub.repr2(info)

    @classmethod
    def from_coco(DetectionMetrics, true_coco, pred_coco, gids=None, verbose=0):
        """
        Create detection metrics from two coco files representing the truth and
        predictions.

        Args:
            true_coco (ndsampler.CocoDataset):
            pred_coco (ndsampler.CocoDataset):

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> import ndsampler
            >>> true_coco = ndsampler.CocoDataset.demo('shapes')
            >>> pred_coco = true_coco
            >>> self = DetectionMetrics.from_coco(true_coco, pred_coco)
            >>> self.score_voc()
        """
        import kwimage

        classes = true_coco.object_categories()
        self = DetectionMetrics(classes)

        if gids is None:
            gids = sorted(set(true_coco.imgs.keys()) & set(pred_coco.imgs.keys()))

        def _coco_to_dets(coco_dset, desc=''):
            for gid in ub.ProgIter(gids, desc=desc, verbose=verbose):
                img = coco_dset.imgs[gid]
                gid = img['id']
                imgname = img['file_name']
                aids = coco_dset.gid_to_aids[gid]
                annots = [coco_dset.anns[aid] for aid in aids]
                dets = kwimage.Detections.from_coco_annots(
                    annots, dset=coco_dset, classes=classes)
                yield dets, imgname, gid

        for dets, imgname, gid in _coco_to_dets(true_coco, desc='add truth'):
            self.add_truth(dets, imgname, gid=gid)

        for dets, imgname, gid in _coco_to_dets(pred_coco, desc='add pred'):
            self.add_predictions(dets, imgname, gid=gid)

        return self

    def _register_imagename(dmet, imgname, gid=None):
        if gid is not None:
            if imgname is None:
                imgname = 'gid_{}'.format(str(gid))
            dmet._imgname_to_gid[imgname] = gid
        else:
            if imgname is None:
                raise ValueError('must specify imgname or gid')
            try:
                gid = dmet._imgname_to_gid[imgname]
            except KeyError:
                gid = len(dmet._imgname_to_gid) + 1
                dmet._imgname_to_gid[imgname] = gid
        return gid

    def add_predictions(dmet, pred_dets, imgname=None, gid=None):
        """
        Register/Add predicted detections for an image

        Args:
            pred_dets (Detections): predicted detections
            imgname (str): a unique string to identify the image
            gid (int, optional): the integer image id if known
        """
        gid = dmet._register_imagename(imgname, gid)
        dmet.gid_to_pred_dets[gid] = pred_dets

    def add_truth(dmet, true_dets, imgname=None, gid=None):
        """
        Register/Add groundtruth detections for an image

        Args:
            true_dets (Detections): groundtruth
            imgname (str): a unique string to identify the image
            gid (int, optional): the integer image id if known
        """
        gid = dmet._register_imagename(imgname, gid)
        dmet.gid_to_true_dets[gid] = true_dets

    def true_detections(dmet, gid):
        """ gets Detections representation for groundtruth in an image """
        return dmet.gid_to_true_dets[gid]

    def pred_detections(dmet, gid):
        """ gets Detections representation for predictions in an image """
        return dmet.gid_to_pred_dets[gid]

    def confusion_vectors(dmet, ovthresh=0.5, bias=0, gids=None, compat='all',
                          prioritize='iou', ignore_classes='ignore',
                          background_class=ub.NoParam, verbose='auto', workers=0):
        """
        Assigns predicted boxes to the true boxes so we can transform the
        detection problem into a classification problem for scoring.

        Args:

            ovthresh (float, default=0.5):
                bounding box overlap iou threshold required for assignment

            bias (float, default=0.0):
                for computing bounding box overlap, either 1 or 0

            gids (List[int], default=None):
                which subset of images ids to compute confusion metrics on. If
                not specified all images are used.

            compat (str, default='all'):
                can be ('ancestors' | 'mutex' | 'all').  determines which pred
                boxes are allowed to match which true boxes. If 'mutex', then
                pred boxes can only match true boxes of the same class. If
                'ancestors', then pred boxes can match true boxes that match or
                have a coarser label. If 'all', then any pred can match any
                true, regardless of its category label.

            prioritize (str, default='iou'):
                can be ('iou' | 'class' | 'correct') determines which box to
                assign to if mutiple true boxes overlap a predicted box.  if
                prioritize is iou, then the true box with maximum iou (above
                ovthresh) will be chosen.  If prioritize is class, then it will
                prefer matching a compatible class above a higher iou. If
                prioritize is correct, then ancestors of the true class are
                preferred over descendents of the true class, over unreleated
                classes.

            ignore_classes (set, default={'ignore'}):
                class names indicating ignore regions

            background_class (str, default=ub.NoParam):
                Name of the background class. If unspecified we try to
                determine it with heuristics. A value of None means there is no
                background class.

            verbose (int, default='auto'): verbosity flag. In auto mode,
                verbose=1 if len(gids) > 1000.

            workers (int, default=0):
                number of parallel assignment processes

        Ignore:
            globals().update(xdev.get_func_kwargs(dmet.confusion_vectors))
        """
        import kwarray
        y_accum = ub.ddict(list)

        TRACK_PROBS = True
        if TRACK_PROBS:
            prob_accum = []

        if gids is None:
            gids = sorted(dmet._imgname_to_gid.values())

        if verbose == 'auto':
            verbose = 1 if len(gids) > 10 else 0

        if background_class is ub.NoParam:
            # Try to autodetermine background class name,
            # otherwise fallback to None
            background_class = None
            if dmet.classes is not None:
                lower_classes = [c.lower() for c in dmet.classes]
                try:
                    idx = lower_classes.index('background')
                    background_class = dmet.classes[idx]
                    # TODO: if we know the background class name should we
                    # change bg_cidx in assignment?
                except ValueError:
                    pass

        from ndsampler.utils import util_futures
        workers = 0
        jobs = util_futures.JobPool(mode='process', max_workers=workers)

        for gid in ub.ProgIter(gids, desc='submit assign jobs',
                               verbose=verbose):
            true_dets = dmet.true_detections(gid)
            pred_dets = dmet.pred_detections(gid)
            job = jobs.submit(
                _assign_confusion_vectors, true_dets, pred_dets,
                bg_weight=1, ovthresh=ovthresh, bg_cidx=-1, bias=bias,
                classes=dmet.classes, compat=compat, prioritize=prioritize,
                ignore_classes=ignore_classes)
            job.gid = gid

        for job in ub.ProgIter(jobs.jobs, desc='assign detections',
                               verbose=verbose):
            y = job.result()
            gid = job.gid

            if TRACK_PROBS:
                # Keep track of per-class probs
                pred_dets = dmet.pred_detections(gid)
                try:
                    pred_probs = pred_dets.probs
                except KeyError:
                    TRACK_PROBS = False
                else:
                    pxs = np.array(y['pxs'], dtype=np.int)

                    # For unassigned truths, we need to create dummy probs
                    # where a background class has probability 1.
                    flags = pxs > -1
                    probs = np.zeros((len(pxs), pred_probs.shape[1]),
                                     dtype=np.float32)
                    if background_class is not None:
                        bg_idx = dmet.classes.index(background_class)
                        probs[:, bg_idx] = 1
                    probs[flags] = pred_probs[pxs[flags]]
                    prob_accum.append(probs)

            y['gid'] = [gid] * len(y['pred'])
            for k, v in y.items():
                y_accum[k].extend(v)

        # else:
        #     for gid in ub.ProgIter(gids, desc='assign detections', verbose=verbose):
        #         true_dets = dmet.true_detections(gid)
        #         pred_dets = dmet.pred_detections(gid)

        #         y = _assign_confusion_vectors(true_dets, pred_dets, bg_weight=1,
        #                                       ovthresh=ovthresh, bg_cidx=-1,
        #                                       bias=bias, classes=dmet.classes,
        #                                       compat=compat, prioritize=prioritize,
        #                                       ignore_classes=ignore_classes)

        #         if TRACK_PROBS:
        #             # Keep track of per-class probs
        #             try:
        #                 pred_probs = pred_dets.probs
        #             except KeyError:
        #                 TRACK_PROBS = False
        #             else:
        #                 pxs = np.array(y['pxs'], dtype=np.int)
        #                 flags = pxs > -1
        #                 probs = np.zeros((len(pxs), pred_probs.shape[1]),
        #                                  dtype=np.float32)
        #                 bg_idx = dmet.classes.node_to_idx['background']
        #                 probs[:, bg_idx] = 1
        #                 probs[flags] = pred_probs[pxs[flags]]
        #                 prob_accum.append(probs)

        #         y['gid'] = [gid] * len(y['pred'])
        #         for k, v in y.items():
        #             y_accum[k].extend(v)

        _data = {}
        for k, v in ub.ProgIter(list(y_accum.items()), desc='ndarray convert', verbose=verbose):
            # Try to use 32 bit types for large evaluation problems
            kw = dict()
            if k in {'iou', 'score', 'weight'}:
                kw['dtype'] = np.float32
            if k in {'pxs', 'txs', 'gid', 'pred', 'true', 'pred_raw'}:
                kw['dtype'] = np.int32
            try:
                _data[k] = np.asarray(v, **kw)
            except TypeError:
                _data[k] = np.asarray(v)

        # Avoid pandas when possible
        cfsn_data = kwarray.DataFrameArray(_data)

        if 0:
            import xdev
            nbytes = 0
            for k, v in _data.items():
                nbytes += v.size * v.dtype.itemsize
            print(xdev.byte_str(nbytes))

        if TRACK_PROBS:
            y_prob = np.vstack(prob_accum)
        else:
            y_prob = None
        cfsn_vecs = ConfusionVectors(cfsn_data, classes=dmet.classes,
                                     probs=y_prob)

        return cfsn_vecs

    def score_kwant(dmet, ovthresh=0.5):
        """
        Scores the detections using kwant
        """
        try:
            from kwil.misc import kwant
            if not kwant.is_available():
                raise ImportError
        except ImportError:
            raise RuntimeError('kwant is not available')

        from kwil import kw18
        gids = list(dmet.gid_to_true_dets.keys())
        true_kw18s = []
        pred_kw18s = []
        for gid in ub.ProgIter(gids, desc='convert to kw18'):
            true_dets = dmet.gid_to_true_dets[gid]
            pred_dets = dmet.gid_to_pred_dets[gid]

            if len(true_dets) == 0:
                print('foo')
            if len(pred_dets) == 0:
                # kwant breaks on 0 predictions, hack in a bad prediction
                import kwimage
                hack_ = kwimage.Detections.random(1)
                hack_.scores[:] = 0
                pred_dets = hack_

            true_kw18 = kw18.make_kw18_from_detections(true_dets,
                                                       frame_number=gid,
                                                       timestamp=gid)
            pred_kw18 = kw18.make_kw18_from_detections(pred_dets,
                                                       frame_number=gid,
                                                       timestamp=gid)
            true_kw18s.append(true_kw18)
            pred_kw18s.append(pred_kw18)

        true_kw18 = true_kw18s
        pred_kw18 = pred_kw18s

        roc_info = kwant.score_events(true_kw18s, pred_kw18s,
                                      ovthresh=ovthresh, prefiltered=True,
                                      verbose=3)

        fp = roc_info['fp'].values
        tp = roc_info['tp'].values

        ppv = tp / (tp + fp)
        ppv[np.isnan(ppv)] = 1

        tpr = roc_info['pd'].values
        fpr = fp / fp[0]
        import sklearn
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        from netharn.metrics.functional import _average_precision
        ap = _average_precision(tpr, ppv)

        roc_info['fpr'] = fpr
        roc_info['ppv'] = ppv

        info = {
            'roc_info': roc_info,
            'ap': ap,
            'roc_auc': roc_auc,
        }

        if False:
            import kwil
            kwil.autompl()
            kwil.multi_plot(roc_info['fa'], roc_info['pd'],
                            xlabel='fa (fp count)',
                            ylabel='pd (tpr)', fnum=1,
                            title='kwant roc_auc={:.4f}'.format(roc_auc))

            import kwil
            kwil.autompl()
            kwil.multi_plot(tpr, ppv,
                            xlabel='recall (fpr)',
                            ylabel='precision (tpr)',
                            fnum=2,
                            title='kwant ap={:.4f}'.format(ap))

        return info

    def score_netharn(dmet, ovthresh=0.5, bias=0, gids=None,
                      compat='all', prioritize='iou'):
        """ our scoring method """
        cfsn_vecs = dmet.confusion_vectors(ovthresh=ovthresh, bias=bias,
                                           gids=gids,
                                           compat=compat,
                                           prioritize=prioritize)

        # THE BINARIZE_PERITEM IS BROKEN
        # cfsn_peritem = cfsn_vecs.binarize_peritem()
        # peritem = cfsn_peritem.precision_recall()

        info = {}
        # info['peritem'] = peritem
        try:
            cfsn_perclass = cfsn_vecs.binarize_ovr(mode=1)
            perclass = cfsn_perclass.precision_recall()
        except Exception as ex:
            print('warning: ex = {!r}'.format(ex))
        else:
            info['perclass'] = perclass['perclass']
            info['mAP'] = perclass['mAP']
        return info

    def score_voc(dmet, ovthresh=0.5, bias=1, method='voc2012', gids=None,
                  ignore_classes='ignore'):
        """
        score using voc method

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=100, nboxes=(0, 3), n_fp=(0, 1), nclasses=8,
            >>>     score_noise=.5)
            >>> print(dmet.score_voc()['mAP'])
            0.9399...
        """
        # from . import voc_metrics
        from netharn.metrics.assignment import _filter_ignore_regions
        from netharn.metrics import voc_metrics
        if gids is None:
            gids = sorted(dmet._imgname_to_gid.values())
        # Convert true/pred detections into VOC format
        vmet = voc_metrics.VOC_Metrics(classes=dmet.classes)
        for gid in gids:
            true_dets = dmet.true_detections(gid)
            pred_dets = dmet.pred_detections(gid)

            if ignore_classes is not None:
                true_ignore_flags, pred_ignore_flags = _filter_ignore_regions(
                    true_dets, pred_dets, ovthresh=ovthresh,
                    ignore_classes=ignore_classes)
                true_dets = true_dets.compress(~true_ignore_flags)
                pred_dets = pred_dets.compress(~pred_ignore_flags)

            vmet.add_truth(true_dets, gid=gid)
            vmet.add_predictions(pred_dets, gid=gid)
        voc_scores = vmet.score(ovthresh, bias=bias, method=method)
        return voc_scores

    def _to_coco(dmet):
        """
        Convert to a coco representation of truth and predictions
        """
        import ndsampler
        true = ndsampler.CocoDataset()
        pred = ndsampler.CocoDataset()

        for node in dmet.classes:
            # cid = dmet.classes.graph.node[node]['id']
            cid = dmet.classes.index(node)
            supercategory = list(dmet.classes.graph.pred[node])
            if len(supercategory) == 0:
                supercategory = None
            else:
                assert len(supercategory) == 1
                supercategory = supercategory[0]
            true.add_category(node, id=cid, supercategory=supercategory)
            pred.add_category(node, id=cid, supercategory=supercategory)

        for imgname, gid in dmet._imgname_to_gid.items():
            true.add_image(imgname, id=gid)
            pred.add_image(imgname, id=gid)

        idx_to_id = {
            idx: dmet.classes.index(node)
            for idx, node in enumerate(dmet.classes.idx_to_node)
        }

        for gid, pred_dets in dmet.gid_to_pred_dets.items():
            pred_boxes = pred_dets.boxes
            if 'scores' in pred_dets.data:
                pred_scores = pred_dets.scores
            else:
                pred_scores = np.ones(len(pred_dets))
            pred_cids = list(ub.take(idx_to_id, pred_dets.class_idxs))
            pred_xywh = pred_boxes.to_xywh().data.tolist()
            for bbox, cid, score in zip(pred_xywh, pred_cids, pred_scores):
                pred.add_annotation(gid, cid, bbox=bbox, score=score)

        for gid, true_dets in dmet.gid_to_true_dets.items():
            true_boxes = true_dets.boxes
            if 'weights' in true_dets.data:
                true_weights = true_dets.weights
            else:
                true_weights = np.ones(len(true_boxes))
            true_cids = list(ub.take(idx_to_id, true_dets.class_idxs))
            true_xywh = true_boxes.to_xywh().data.tolist()
            for bbox, cid, weight in zip(true_xywh, true_cids, true_weights):
                true.add_annotation(gid, cid, bbox=bbox, weight=weight)

        return pred, true

    def score_coco(dmet, verbose=0):
        """
        score using ms-coco method

        Example:
            >>> # xdoctest: +REQUIRES(--pycocotools)
            >>> dmet = DetectionMetrics.demo(
            >>>     nimgs=100, nboxes=(0, 3), n_fp=(0, 1), nclasses=8)
            >>> print(dmet.score_coco()['mAP'])
            0.711016...
        """
        from pycocotools import coco
        from pycocotools import cocoeval
        # The original pycoco-api prints to much, supress it
        import netharn as nh

        pred, true = dmet._to_coco()

        quiet = verbose == 0
        with nh.util.SupressPrint(coco, cocoeval, enabled=quiet):
            cocoGt = true._aspycoco()
            cocoDt = pred._aspycoco()

            for ann in cocoGt.dataset['annotations']:
                w, h = ann['bbox'][-2:]
                ann['ignore'] = ann['weight'] < .5
                ann['area'] = w * h
                ann['iscrowd'] = False

            for ann in cocoDt.dataset['annotations']:
                w, h = ann['bbox'][-2:]
                ann['area'] = w * h

            evaler = cocoeval.COCOeval(cocoGt, cocoDt, iouType='bbox')
            evaler.evaluate()
            evaler.accumulate()
            evaler.summarize()
            coco_ap = evaler.stats[1]
            coco_scores = {
                'mAP': coco_ap,
                'evalar_stats': evaler.stats
            }
        return coco_scores

    @classmethod
    def demo(cls, **kwargs):
        """
        Creates random true boxes and predicted boxes that have some noisy
        offset from the truth.

        Kwargs:
            nclasses (int, default=1): number of foreground classes.
            nimgs (int, default=1): number of images in the coco datasts.
            nboxes (int, default=1): boxes per image.
            n_fp (int, default=0): number of false positives.
            n_fn (int, default=0): number of false negatives.
            box_noise (float, default=0): std of a normal distribution used to
                perterb both box location and box size.
            cls_noise (float, default=0): probability that a class label will
                change. Must be within 0 and 1.
            anchors (ndarray, default=None): used to create random boxes
            null_pred (bool, default=0):
                if True, predicted classes are returned as null, which means
                only localization scoring is suitable.
            with_probs (bool, default=1):
                if True, includes per-class probabilities with predictions

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> kwargs = {}
            >>> # Seed the RNG
            >>> kwargs['rng'] = 0
            >>> # Size parameters determine how big the data is
            >>> kwargs['nimgs'] = 5
            >>> kwargs['nboxes'] = 7
            >>> kwargs['nclasses'] = 11
            >>> # Noise parameters perterb predictions further from the truth
            >>> kwargs['n_fp'] = 3
            >>> kwargs['box_noise'] = 0.1
            >>> kwargs['cls_noise'] = 0.5
            >>> dmet = DetectionMetrics.demo(**kwargs)
            >>> print('dmet.classes = {}'.format(dmet.classes))
            dmet.classes = <CategoryTree(nNodes=12, maxDepth=3, maxBreadth=4...)>
            >>> # Can grab kwimage.Detection object for any image
            >>> print(dmet.true_detections(gid=0))
            <Detections(4)>
            >>> print(dmet.pred_detections(gid=0))
            <Detections(7)>

        Example:
            >>> # xdoctest: +REQUIRES(module:ndsampler)
            >>> # Test case with null predicted categories
            >>> dmet = DetectionMetrics.demo(nimgs=30, null_pred=1, nclasses=3,
            >>>                              nboxes=10, n_fp=10, box_noise=0.3,
            >>>                              with_probs=False)
            >>> dmet.gid_to_pred_dets[0].data
            >>> dmet.gid_to_true_dets[0].data
            >>> cfsn_vecs = dmet.confusion_vectors()
            >>> binvecs_ovr = cfsn_vecs.binarize_ovr()
            >>> binvecs_per = cfsn_vecs.binarize_peritem()
            >>> pr_per = binvecs_per.precision_recall()
            >>> pr_ovr = binvecs_ovr.precision_recall()
            >>> print('pr_per = {!r}'.format(pr_per))
            >>> print('pr_ovr = {!r}'.format(pr_ovr))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> pr_per.draw(fnum=1)
            >>> pr_ovr['perclass'].draw(fnum=2)
        """
        import kwimage
        import kwarray
        import ndsampler
        # Parse kwargs
        rng = kwarray.ensure_rng(kwargs.get('rng', 0))
        nclasses = kwargs.get('nclasses', 1)
        nimgs = kwargs.get('nimgs', 1)
        box_noise = kwargs.get('box_noise', 0)
        cls_noise = kwargs.get('cls_noise', 0)

        null_pred = kwargs.get('null_pred', False)
        with_probs = kwargs.get('with_probs', True)

        # specify an amount of overlap between true and false scores
        score_noise = kwargs.get('score_noise', 0.2)

        anchors = kwargs.get('anchors', None)
        scale = 100.0

        # Build random variables
        from kwarray import distributions
        DiscreteUniform = distributions.DiscreteUniform.seeded(rng=rng)
        def _parse_arg(key, default):
            value = kwargs.get(key, default)
            try:
                low, high = value
                return (low, high + 1)
            except Exception:
                return (0, value + 1)
        nboxes_RV = DiscreteUniform(*_parse_arg('nboxes', 1))
        n_fp_RV = DiscreteUniform(*_parse_arg('n_fp', 0))
        n_fn_RV = DiscreteUniform(*_parse_arg('n_fn', 0))

        box_noise_RV = distributions.Normal(0, box_noise, rng=rng)
        cls_noise_RV = distributions.Bernoulli(cls_noise, rng=rng)

        # the values of true and false scores starts off with no overlap and
        # the overlap increases as the score noise increases.
        def _interp(v1, v2, alpha):
            return v1 * alpha + (1 - alpha) * v2
        mid = 0.5
        # true_high = 2.0
        true_high = 1.0
        true_low   = _interp(0, mid, score_noise)
        false_high = _interp(true_high, mid - 1e-3, score_noise)
        true_mean  = _interp(0.5, .8, score_noise)
        false_mean = _interp(0.5, .2, score_noise)

        true_score_RV = distributions.TruncNormal(
            mean=true_mean, std=.5, low=true_low, high=true_high, rng=rng)
        false_score_RV = distributions.TruncNormal(
            mean=false_mean, std=.5, low=0, high=false_high, rng=rng)

        frgnd_cx_RV = distributions.DiscreteUniform(
            1, nclasses + 1, rng=rng)

        # Create the category hierarcy
        graph = nx.DiGraph()
        graph.add_node('background', id=0)
        for cid in range(1, nclasses + 1):
            # binary heap encoding of a tree
            cx = cid - 1
            parent_cx = (cx - 1) // 2
            node = 'cat_{}'.format(cid)
            graph.add_node(node, id=cid)
            if parent_cx > 0:
                supercategory = 'cat_{}'.format(parent_cx + 1)
                graph.add_edge(supercategory, node)
        classes = ndsampler.CategoryTree(graph)

        dmet = cls()
        dmet.classes = classes

        for gid in range(nimgs):

            # Sample random variables
            nboxes_ = nboxes_RV()
            n_fp_ = n_fp_RV()
            n_fn_ = n_fn_RV()

            imgname = 'img_{}'.format(gid)
            dmet._register_imagename(imgname, gid)

            # Generate random ground truth detections
            true_boxes = kwimage.Boxes.random(num=nboxes_, scale=scale,
                                              anchors=anchors, rng=rng,
                                              format='cxywh')
            # Prevent 0 sized boxes: increase w/h by 1
            true_boxes.data[..., 2:4] += 1
            true_cxs = frgnd_cx_RV(len(true_boxes))
            true_weights = np.ones(len(true_boxes), dtype=np.int32)

            # Initialize predicted detections as a copy of truth
            pred_boxes = true_boxes.copy()
            pred_cxs = true_cxs.copy()

            # Perterb box coordinates
            pred_boxes.data = np.abs(pred_boxes.data.astype(np.float) +
                                     box_noise_RV())

            # Perterb class predictions
            change = cls_noise_RV(len(pred_cxs))
            pred_cxs_swap = frgnd_cx_RV(len(pred_cxs))
            pred_cxs[change] = pred_cxs_swap[change]

            # Drop true positive boxes
            if n_fn_:
                pred_boxes.data = pred_boxes.data[n_fn_:]
                pred_cxs = pred_cxs[n_fn_:]

            # pred_scores = np.linspace(true_min, true_max, len(pred_boxes))[::-1]
            n_tp_ = len(pred_boxes)
            pred_scores = true_score_RV(n_tp_)

            # Add false positive boxes
            if n_fp_:
                false_boxes = kwimage.Boxes.random(num=n_fp_, scale=scale,
                                                   rng=rng, format='cxywh')
                false_cxs = frgnd_cx_RV(n_fp_)
                false_scores = false_score_RV(n_fp_)

                pred_boxes.data = np.vstack([pred_boxes.data, false_boxes.data])
                pred_cxs = np.hstack([pred_cxs, false_cxs])
                pred_scores = np.hstack([pred_scores, false_scores])

            # Transform the scores for the assigned class into a predicted
            # probability for each class. (Currently a bit hacky).
            class_probs = _demo_construct_probs(
                pred_cxs, pred_scores, classes, rng,
                hacked=kwargs.get('hacked', 0))

            true_dets = kwimage.Detections(boxes=true_boxes,
                                           class_idxs=true_cxs,
                                           weights=true_weights)

            pred_dets = kwimage.Detections(boxes=pred_boxes,
                                           class_idxs=pred_cxs,
                                           scores=pred_scores)

            # Hack in the probs
            if with_probs:
                pred_dets.data['probs'] = class_probs

            if null_pred:
                pred_dets.data['class_idxs'] = np.array(
                    [None] * len(pred_dets), dtype=object)

            dmet.add_truth(true_dets, imgname=imgname)
            dmet.add_predictions(pred_dets, imgname=imgname)

        return dmet


def _demo_construct_probs(pred_cxs, pred_scores, classes, rng, hacked=1):
    """
    Constructs random probabilities for demo data
    """
    # Setup probs such that the assigned class receives a probability
    # equal-(ish) to the assigned score.
    # Its a bit tricky to setup hierarchical probs such that we get the
    # scores in the right place. We punt and just make probs
    # conditional. The right thing to do would be to do this, and then
    # perterb ancestor categories such that the probability evenetually
    # converges on the right value at that specific classes depth.
    import torch

    # Ensure probs
    pred_scores2 = pred_scores.clip(0, 1.0)

    class_energy = rng.rand(len(pred_scores2), len(classes)).astype(np.float32)
    for p, x, s in zip(class_energy, pred_cxs, pred_scores2):
        p[x] = s

    if hacked:
        # HACK! All that nice work we did is too slow for doctests
        return class_energy

    class_energy = torch.Tensor(class_energy)
    cond_logprobs = classes.conditional_log_softmax(class_energy, dim=1)
    cond_probs = torch.exp(cond_logprobs).numpy()

    # I was having a difficult time getting this right, so an
    # inefficient per-item non-vectorized implementation it is.
    # Note: that this implementation takes 70% of the time in this function
    # and is a bottleneck for the doctests. A vectorized implementation would
    # be nice.
    idx_to_ancestor_idxs = classes.idx_to_ancestor_idxs()
    idx_to_groups = {idx: group for group in classes.idx_groups for idx in group}

    def set_conditional_score(row, cx, score, idx_to_groups):
        group_cxs = np.array(idx_to_groups[cx])
        flags = group_cxs == cx
        group_row = row[group_cxs]
        # Ensure that that heriarchical probs sum to 1
        current = group_row[~flags]
        other = current * (1 - score) / current.sum()
        other = np.nan_to_num(other)
        group_row[~flags] = other
        group_row[flags] = score
        row[group_cxs] = group_row

    for row, cx, score in zip(cond_probs, pred_cxs, pred_scores2):
        set_conditional_score(row, cx, score, idx_to_groups)
        for ancestor_cx in idx_to_ancestor_idxs[cx]:
            if ancestor_cx != cx:
                # Hack all parent probs to 1.0 so conditional probs
                # turn into real probs.
                set_conditional_score(row, ancestor_cx, 1.0, idx_to_groups)
                # TODO: could add a fudge factor here so the
                # conditional prob is higher than score, but parent
                # probs are less than 1.0

                # TODO: could also maximize entropy of descendant nodes
                # so classes.decision2 would stop at this node

    # For each level the conditional probs must sum to 1
    if cond_probs.size > 0:
        for idxs in classes.idx_groups:
            level = cond_probs[:, idxs]
            totals = level.sum(axis=1)
            assert level.shape[1] == 1 or np.allclose(totals, 1.0), str(level) + ' : ' + str(totals)

    cond_logprobs = torch.Tensor(cond_probs).log()
    class_probs = classes._apply_logprob_chain_rule(cond_logprobs, dim=1).exp().numpy()
    class_probs = class_probs.reshape(-1, len(classes))
    # print([p[x] for p, x in zip(class_probs, pred_cxs)])
    # print(pred_scores2)
    return class_probs


def eval_detections_cli(**kw):
    """
    CommandLine:
        xdoctest -m ~/code/netharn/netharn/metrics/detect_metrics.py eval_detections_cli
    """
    import scriptconfig as scfg
    import ndsampler

    class EvalDetectionCLI(scfg.Config):
        default = {
            'true': scfg.Path(None, help='true coco dataset'),
            'pred': scfg.Path(None, help='predicted coco dataset'),
            'out_dpath': scfg.Path('./out', help='output directory')
        }
        pass

    config = EvalDetectionCLI()
    cmdline = kw.pop('cmdline', True)
    config.load(kw, cmdline=cmdline)

    true_coco = ndsampler.CocoDataset(config['true'])
    pred_coco = ndsampler.CocoDataset(config['pred'])

    from netharn.metrics.detect_metrics import DetectionMetrics
    dmet = DetectionMetrics.from_coco(true_coco, pred_coco)

    voc_info = dmet.score_voc()

    cls_info = voc_info['perclass'][0]
    tp = cls_info['tp']
    fp = cls_info['fp']
    fn = cls_info['fn']

    tpr = cls_info['tpr']
    ppv = cls_info['ppv']
    fp = cls_info['fp']

    # Compute the MCC as TN->inf
    thresh = cls_info['thresholds']

    # https://erotemic.wordpress.com/2019/10/23/closed-form-of-the-mcc-when-tn-inf/
    mcc_lim = tp / (np.sqrt(fn + tp) * np.sqrt(fp + tp))
    f1 = 2 * (ppv * tpr) / (ppv + tpr)

    draw = False
    if draw:

        mcc_idx = mcc_lim.argmax()
        f1_idx = f1.argmax()

        import kwplot
        plt = kwplot.autoplt()

        kwplot.multi_plot(
            xdata=thresh,
            ydata=mcc_lim,
            xlabel='threshold',
            ylabel='mcc*',
            fnum=1, pnum=(1, 4, 1),
            title='MCC*',
            color=['blue'],
        )
        plt.plot(thresh[mcc_idx], mcc_lim[mcc_idx], 'r*', markersize=20)
        plt.plot(thresh[f1_idx], mcc_lim[f1_idx], 'k*', markersize=20)

        kwplot.multi_plot(
            xdata=fp,
            ydata=tpr,
            xlabel='fp (fa)',
            ylabel='tpr (pd)',
            fnum=1, pnum=(1, 4, 2),
            title='ROC',
            color=['blue'],
        )
        plt.plot(fp[mcc_idx], tpr[mcc_idx], 'r*', markersize=20)
        plt.plot(fp[f1_idx], tpr[f1_idx], 'k*', markersize=20)

        kwplot.multi_plot(
            xdata=tpr,
            ydata=ppv,
            xlabel='tpr (recall)',
            ylabel='ppv (precision)',
            fnum=1, pnum=(1, 4, 3),
            title='PR',
            color=['blue'],
        )
        plt.plot(tpr[mcc_idx], ppv[mcc_idx], 'r*', markersize=20)
        plt.plot(tpr[f1_idx], ppv[f1_idx], 'k*', markersize=20)

        kwplot.multi_plot(
            xdata=thresh,
            ydata=f1,
            xlabel='threshold',
            ylabel='f1',
            fnum=1, pnum=(1, 4, 4),
            title='F1',
            color=['blue'],
        )
        plt.plot(thresh[mcc_idx], f1[mcc_idx], 'r*', markersize=20)
        plt.plot(thresh[f1_idx], f1[f1_idx], 'k*', markersize=20)
