# -*- coding: utf-8 -*-
"""
This module can be used as both a script and an importable module.
Run `python ggr_matching.py --help` for more details.
See docstring in fit for more details on the importable module.


conda install opencv
conda install pytorch torchvision -c pytorch

TestMe:
    xdoctest ~/code/netharn/netharn/examples/ggr_matching.py all
"""

from os.path import join
import ubelt as ub
import numpy as np
import netharn as nh
import torch
import torchvision
import ndsampler
import itertools as it


def setup_harn(**kwargs):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/ggr_matching.py setup_harn

    Args:
        dbname (str): Name of IBEIS database to use
        nice (str): Custom tag for this run
        workdir (PathLike): path to dump all the intermedate results
        dim (int): Width and height of the network input
        batch_size (int): Base batch size. Number of examples in GPU at any time.
        bstep (int): Multiply by batch_size to simulate a larger batches.
        lr (float): Base learning rate
        decay (float): Weight decay (L2 regularization)
        workers (int): Number of parallel data loader workers
        xpu (str): Device to train on. Can be either `'cpu'`, `'gpu'`, a number
            indicating a GPU (e.g. `0`), or a list of numbers (e.g. `[0,1,2]`)
            indicating multiple GPUs
        triple (bool): if True uses triplet loss, otherwise contrastive loss
        margin (float): margin for loss criterion
        norm_desc (bool): if True normalizes the descriptors
        pretrained (PathLike): path to a compatible pretrained model

    Example:
        >>> harn = setup_harn(dbname='PZ_MTEST')
        >>> harn.initialize()
    """
    config = {}
    config['dim'] = int(kwargs.get('dim', 416))
    config['triple'] = kwargs.get('triple', False)
    config['norm_desc'] = kwargs.get('norm_desc', False)

    config['init'] = kwargs.get('init', 'kaiming_normal')
    config['pretrained'] = config.get('pretrained', ub.argval('--pretrained', default=None))

    config['scheduler'] = kwargs.get('scheduler', 'onecycle71')
    config['margin'] = float(kwargs.get('margin', 4.0))
    config['xpu'] = kwargs.get('xpu', 'argv')
    config['nice'] = kwargs.get('nice', 'untitled')
    config['workdir'] = kwargs.get('workdir', None)
    config['batch_size'] = int(kwargs.get('batch_size', 6))
    config['workers'] = int(kwargs.get('workers', 0))
    config['dbname'] = kwargs.get('dbname', 'ggr2')
    config['bstep'] = int(kwargs.get('bstep', 1))
    config['decay'] = float(kwargs.get('decay', 1e-5))
    config['lr'] = float(kwargs.get('lr', 0.019))

    nh.configure_hacks(config)

    workdir = nh.configure_workdir(config,
                                   workdir=join('~/work/siam-ibeis2', config['dbname']))

    dim = config['dim']
    triple = config['triple']
    if config['dbname'] == 'ggr2':
        print('Creating torch CocoDataset')
        train_dset = ndsampler.CocoDataset(
            data='/media/joncrall/raid/data/ggr2-coco/annotations/instances_train2018.json',
            img_root='/media/joncrall/raid/data/ggr2-coco/images/train2018',
        )
        train_dset.hashid = 'ggr2-coco-train2018'
        vali_dset = ndsampler.CocoDataset(
            data='/media/joncrall/raid/data/ggr2-coco/annotations/instances_val2018.json',
            img_root='/media/joncrall/raid/data/ggr2-coco/images/val2018',
        )
        vali_dset.hashid = 'ggr2-coco-val2018'

        print('Creating samplers')
        train_sampler = ndsampler.CocoSampler(train_dset, workdir=workdir)
        vali_sampler = ndsampler.CocoSampler(vali_dset, workdir=workdir)

        print('Creating torch Datasets')
        datasets = {
            'train': MatchingCocoDataset(train_sampler, train_dset, workdir,
                                         dim=dim, augment=True, triple=triple),
            'vali': MatchingCocoDataset(vali_sampler, vali_dset, workdir,
                                        dim=dim, triple=triple),
        }
    else:
        from ibeis_utils import randomized_ibeis_dset
        # TODO: triple
        datasets = randomized_ibeis_dset(config['dbname'], dim=dim)

    for k, v in datasets.items():
        print('* len({}) = {}'.format(k, len(v)))

    if config['triple']:
        criterion_ = (torch.nn.TripletMarginLoss, {
            'margin': config['margin'],
        })
    else:
        criterion_ = (nh.criterions.ContrastiveLoss, {
            'margin': config['margin'],
            'weight': None,
        })

    hyper = nh.HyperParams(**{
        'nice': config['nice'],
        'workdir': config['workdir'],
        'datasets': datasets,

        'model': (MatchingNetworkLP, {
            'p': 2,
            'input_shape': (1, 3, dim, dim),
            'norm_desc': config['norm_desc'],
        }),

        'criterion': criterion_,

        'loaders': nh.Loaders.coerce(datasets, config),
        'xpu': nh.XPU.cast(config['xpu']),

        'initializer': nh.Initializer.coerce(config),

        'optimizer': nh.Optimizer.coerce(config),
        'scheduler': nh.Scheduler.coerce(config, scheduler='onecycle70'),

        'monitor': nh.Monitor.coerce(
            config,
            minimize=['loss', 'pos_dist', 'brier'],
            maximize=['accuracy', 'neg_dist', 'mcc'],
            patience=40,
            max_epoch=100,
        ),
        'dynamics': nh.Dynamics.coerce(config),

        'other': {
            'n_classes': 2,
            # 'augment': datasets['train'].augmenter,
        },
    })
    harn = MatchingHarness(hyper=hyper)
    harn.config['prog_backend'] = 'progiter'
    harn.intervals['log_iter_train'] = 1
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None

    return harn


class MatchingHarness(nh.FitHarn):
    """
    Define how to process a batch, compute loss, and evaluate validation
    metrics.

    Example:
        >>> from ggr_matching import *
        >>> harn = setup_harn()
        >>> harn.initialize()
        >>> batch = harn._demo_batch(0, 'train')
        >>> batch = harn._demo_batch(0, 'vali')
    """

    def __init__(harn, *args, **kw):
        super(MatchingHarness, harn).__init__(*args, **kw)

    def after_initialize(harn, **kw):
        harn.confusion_vectors = []
        harn._has_preselected = False

    def before_epochs(harn):
        verbose = 0
        for tag, dset in harn.datasets.items():
            if isinstance(dset, torch.utils.data.Subset):
                dset = dset.dataset

            # Presample negatives before each epoch.
            if tag == 'train':
                if not harn._has_preselected:
                    harn.log('Presample {} dataset'.format(tag))
                # Randomly resample training negatives every epoch
                harn.datasets['train'].preselect(verbose=verbose)
            else:
                if not harn._has_preselected:
                    harn.log('Presample {} dataset'.format(tag))
                    dset.preselect(verbose=verbose)
        harn._has_preselected = True

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        if harn.datasets['train'].triple:
            batch = {
                'img1': harn.xpu.move(raw_batch['img1']),
                'img2': harn.xpu.move(raw_batch['img2']),
                'img3': harn.xpu.move(raw_batch['img3']),
            }
        else:
            batch = {
                'img1': harn.xpu.move(raw_batch['img1']),
                'img2': harn.xpu.move(raw_batch['img2']),
                'label': harn.xpu.move(raw_batch['label'])
            }
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader
        """
        if harn.datasets['train'].triple:
            inputs = [batch['img1'], batch['img2'], batch['img3']]
            outputs = harn.model(*inputs)
            d1, d2, d3 = outputs['dvecs']
            loss = harn.criterion(d1, d2, d3).sum()
        else:
            inputs = [batch['img1'], batch['img2']]
            label = batch['label']
            outputs = harn.model(*inputs)
            d1, d2 = outputs['dvecs']
            dist12 = harn.xpu.raw(harn.model).pdist(d1, d2)
            loss = harn.criterion(dist12, label).sum()
        return outputs, loss

    def _decode(harn, outputs):
        dvecs = [d for d in outputs['dvecs']]

        decoded = {
            'dvecs': [d.data.cpu().numpy() for d in dvecs],
        }
        if len(dvecs) > 1:
            dist12 = harn.xpu.raw(harn.model).pdist(dvecs[0], dvecs[1])
            decoded['dist12'] = dist12.data.cpu().numpy()
        if len(dvecs) > 2:
            dist13 = harn.xpu.raw(harn.model).pdist(dvecs[0], dvecs[2])
            decoded['dist13'] = dist13.data.cpu().numpy()
            dist23 = harn.xpu.raw(harn.model).pdist(dvecs[1], dvecs[2])
            decoded['dist23'] = dist23.data.cpu().numpy()

        return decoded

    def on_batch(harn, batch, outputs, loss):
        """ custom callback

        Example:
            >>> from ggr_matching import *
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='vali')
            >>> outputs, loss = harn.run_batch(batch)
            >>> decoded = harn._decode(outputs)
            >>> stacked = harn._draw_batch(batch, decoded, limit=42)
        """

        bx = harn.bxs[harn.current_tag]
        decoded = harn._decode(outputs)
        if bx < 8:
            stacked = harn._draw_batch(batch, decoded)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'batch_{}_epoch_{}.jpg'.format(bx, harn.epoch))
            nh.util.imwrite(fpath, stacked)

        POS_LABEL = 1
        NEG_LABEL = 0

        if harn.datasets['train'].triple:
            n = len(decoded['dist12'])
            harn.confusion_vectors.append(([POS_LABEL] * n, decoded['dist12']))
            harn.confusion_vectors.append(([NEG_LABEL] * n, decoded['dist13']))
            harn.confusion_vectors.append(([NEG_LABEL] * n, decoded['dist23']))
        else:
            label = batch['label']
            l2_dist_tensor = decoded['dist12']
            label_tensor = torch.squeeze(label).data.cpu()

            # Record metrics for epoch scores
            y_true = label_tensor
            y_dist = l2_dist_tensor
            harn.confusion_vectors.append((y_true, y_dist))
            if False:
                # Distance
                is_pos = (y_true == POS_LABEL)

                pos_dists = y_dist[is_pos]
                neg_dists = y_dist[~is_pos]

                # Average positive / negative distances
                pos_dist = pos_dists.sum() / max(1, len(pos_dists))
                neg_dist = neg_dists.sum() / max(1, len(neg_dists))

                # accuracy
                margin = harn.hyper.criterion_params['margin']
                pred_pos_flags = (y_dist <= margin).long()

                pred = pred_pos_flags
                n_correct = (pred == y_true).sum()
                fraction_correct = n_correct / len(y_true)

                metrics = {
                    'accuracy': float(fraction_correct),
                    'pos_dist': float(pos_dist),
                    'neg_dist': float(neg_dist),
                }
                return metrics

    def on_epoch(harn):
        """ custom callback """
        from sklearn import metrics
        margin = harn.hyper.criterion_params['margin']
        epoch_metrics = {}

        if harn.confusion_vectors:
            y_true = np.hstack([r for r, p in harn.confusion_vectors])
            y_dist = np.hstack([p for r, p in harn.confusion_vectors])

            POS_LABEL = 1  # NOQA
            NEG_LABEL = 0  # NOQA
            pos_dist = np.nanmean(y_dist[y_true == POS_LABEL])
            neg_dist = np.nanmean(y_dist[y_true == NEG_LABEL])

            # Transform distance into a probability-like space
            y_probs = torch.sigmoid(torch.Tensor(-(y_dist - margin))).numpy()

            brier = y_probs - y_true

            y_pred = (y_dist <= margin).astype(y_true.dtype)
            accuracy = (y_true == y_pred).mean()
            mcc = metrics.matthews_corrcoef(y_true, y_pred)
            brier = ((y_probs - y_true) ** 2).mean()

            epoch_metrics = {
                'mcc': mcc,
                'brier': brier,
                'accuracy': accuracy,
                'pos_dist': pos_dist,
                'neg_dist': neg_dist,
            }

        # Clear scores for next epoch
        harn.confusion_vectors.clear()
        return epoch_metrics

    def _draw_batch(harn, batch, decoded, limit=32):
        """
        Example:
            >>> from ggr_matching import *
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='vali')
            >>> outputs, loss = harn.run_batch(batch)
            >>> decoded = harn._decode(outputs)
            >>> stacked = harn._draw_batch(batch, decoded, limit=42)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import netharn as nh
            >>> nh.util.autompl()
            >>> nh.util.imshow(stacked, colorspace='rgb', doclf=True)
            >>> nh.util.show_if_requested()
        """
        if 'img3' in batch:
            imgs = [batch[k].data.cpu().numpy() for k in ['img1', 'img2', 'img3']]
        else:
            imgs = [batch[k].data.cpu().numpy() for k in ['img1', 'img2']]
            labels = batch['label'].data.cpu().numpy()

        tostack = []
        fontkw = {
            'fontScale': 1.0,
            'thickness': 2
        }
        n = min(limit, len(imgs[0]))
        for i in range(n):
            ims = [g[i].transpose(1, 2, 0) for g in imgs]
            img = nh.util.stack_images(ims, overlap=-2, axis=1)
            if 'dist13' in decoded:
                text = 'dist12={:.2f} --- dist13={:.2f} --- dist23={:.2f}'.format(
                    decoded['dist12'][i],
                    decoded['dist13'][i],
                    decoded['dist23'][i],
                )
            else:
                dist = decoded['dist12'][i]
                label = labels[i]
                text = 'dist={:.2f}, label={}'.format(dist, label)
            img = (img * 255).astype(np.uint8)
            img = nh.util.draw_text_on_image(img, text,
                                             org=(2, img.shape[0] - 2),
                                             color='blue', **fontkw)
            tostack.append(img)
        stacked = nh.util.stack_images_grid(tostack, overlap=-10,
                                            bg_value=(10, 40, 30),
                                            axis=1, chunksize=3)
        return stacked


class MatchingNetworkLP(torch.nn.Module):
    """
    L2 pairwise distance matching network

    Example:
        >>> self = MatchingNetworkLP()
        >>> input_shapes = [(4, 3, 244, 244), (4, 3, 244, 244)]
        >>> self.output_shape_for(*input_shapes)  # todo pdist module
    """

    def __init__(self, p=2, branch=None, input_shape=(1, 3, 416, 416),
                 norm_desc=False):
        super(MatchingNetworkLP, self).__init__()
        if branch is None:
            self.branch = torchvision.models.resnet50(pretrained=True)
        else:
            self.branch = branch
        assert isinstance(self.branch, torchvision.models.ResNet)

        # Note the advanced usage of output-shape-for
        branch_shape = nh.OutputShapeFor(self.branch)(input_shape)
        prepool_shape = branch_shape.hidden.shallow(1)['layer4']
        # replace the last layer of resnet with a linear embedding to learn the
        # LP distance between pairs of images.
        # Also need to replace the pooling layer in case the input has a
        # different size.
        self.prepool_shape = prepool_shape
        pool_channels = prepool_shape[1]
        pool_dims = prepool_shape[2:]
        self.branch.avgpool = torch.nn.AvgPool2d(pool_dims, stride=1)
        self.branch.fc = torch.nn.Linear(pool_channels, 1024)

        self.norm_desc = norm_desc

        self.p = p
        self.pdist = torch.nn.PairwiseDistance(p=p)

    def forward(self, *inputs):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.

        Example:
            >>> input1 = nh.XPU(None).variable(torch.rand(4, 3, 224, 224))
            >>> input2 = nh.XPU(None).variable(torch.rand(4, 3, 224, 224))
            >>> self = MatchingNetworkLP(input_shape=input2.shape[1:])
            >>> output = self(input1, input2)

        Ignore:
            >>> input1 = nh.XPU(None).variable(torch.rand(1, 3, 416, 416))
            >>> input2 = nh.XPU(None).variable(torch.rand(1, 3, 416, 416))
            >>> input_shape1 = input1.shape
            >>> self = MatchingNetworkLP(input_shape=input2.shape[1:])
            >>> self(input1, input2)
        """
        dvecs = [self.branch(i) for i in inputs]
        if self.norm_desc:
            # LP normalize the vectors
            dvecs = [torch.nn.functional.normalize(d, p=self.p) for d in dvecs]
        if len(inputs) == 2:
            # dist = self.pdist(*dvecs)
            output = {
                'dvecs': dvecs,
                # 'dist12': dist,
            }
        else:
            output = {
                'dvecs': dvecs,
            }
        return output

    def output_shape_for(self, *input_shapes):
        inputs = input_shapes
        dvecs = [nh.OutputShapeFor(self.branch)(i) for i in inputs]
        if len(inputs) == 2:
            dist = nh.OutputShapeFor(self.pdist)(*dvecs)
            output = {
                'dvecs': dvecs,
                'dist': dist,
            }
        else:
            output = {
                'dvecs': dvecs,
            }
        return output


def extract_ggr_pccs(coco_dset):
    import graphid
    graph = graphid.api.GraphID()
    graph.add_annots_from(coco_dset.annots().aids)
    infr = graph.infr
    infr.params['inference.enabled'] = False
    all_aids = list(coco_dset.annots().aids)
    aids_set = set(all_aids)
    for aid1 in ub.ProgIter(all_aids, desc='construct graph'):
        annot = coco_dset.anns[aid1]
        for review in annot['review_ids']:
            aid2, decision = review
            if aid2 not in aids_set:
                # hack because data is setup wrong
                continue
            edge = (aid1, aid2)
            if decision == 'positive':
                infr.add_feedback(edge, evidence_decision=graphid.core.POSTV)
            elif decision == 'negative':
                infr.add_feedback(edge, evidence_decision=graphid.core.NEGTV)
            elif decision == 'incomparable':
                infr.add_feedback(edge, evidence_decision=graphid.core.INCMP)
            else:
                raise KeyError(decision)
    infr.params['inference.enabled'] = True
    infr.apply_nondynamic_update()
    print('status = {}' + ub.repr2(infr.status(True)))
    pccs = list(map(frozenset, infr.positive_components()))
    for pcc in pccs:
        for aid in pcc:
            print('aid = {!r}'.format(aid))
            assert aid in coco_dset.anns
    return pccs


class MatchingCocoDataset(torch.utils.data.Dataset):
    """
    Example:
        >>> harn = setup_harn(dbname='ggr2', xpu='cpu').initialize()
        >>> self = harn.datasets['train']
    """
    def __init__(self, sampler, coco_dset, workdir=None, augment=False,
                 dim=416, triple=True):
        print('make MatchingCocoDataset')

        cacher = ub.Cacher('pccs', cfgstr=coco_dset.tag, verbose=True)
        pccs = cacher.tryload()
        if pccs is None:
            pccs = extract_ggr_pccs(coco_dset)
            cacher.save(pccs)

        self.pccs = pccs

        self.sampler = sampler

        print('target index')
        self.aid_to_tx = {aid: tx for tx, aid in
                          enumerate(sampler.regions.targets['aid'])}

        self.coco_dset = coco_dset

        self.triple = triple

        self.max_num = int(1e5)
        self.pos_ceil = sum((n * (n - 1)) // 2 for n in map(len, self.pccs))
        self.max_num = min(self.pos_ceil, self.max_num)

        if self.triple:
            self.sample_gen = sample_triples(pccs, rng=0)
            self.samples = [next(self.sample_gen) for _ in range(self.max_num)]
            nh.util.shuffle(self.samples, rng=0)
        else:
            print('Find Samples')
            self.sample_gen = sample_edges_inf(self.pccs)
            self.samples = sample_edges_finite(self.pccs, max_num=self.max_num,
                                               pos_neg_ratio=1.0)
            nh.util.shuffle(self.samples, rng=0)

        print('Finished sampling')
        window_dim = dim
        self.dim = window_dim
        self.window_dim = window_dim

        self.rng = nh.util.ensure_rng(0)
        if augment:
            import imgaug.augmenters as iaa
            # NOTE: we are only using `self.augmenter` to make a hyper hashid
            # in __getitem__ we invoke transform explicitly for fine control
            self.hue = nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)
            self.crop = iaa.Crop(percent=(0, .2))
            self.flip = iaa.Fliplr(p=.5)

            self.dependent = self.flip
            self.independent = iaa.Sequential([
                self.hue,
                self.crop,
                # self.flip
            ])

            self.augmenter = iaa.Sequential([
                # self.hue,
                self.crop,
                self.flip
            ])
        else:
            self.augmenter = None
        target_size = (window_dim, window_dim)
        self.letterbox = nh.data.transforms.Resize(target_size=target_size,
                                                   mode='letterbox')

        self.preselect()

    def preselect(self, verbose=0):
        if self.augmenter:
            n = len(self)
            self.samples = [next(self.sample_gen) for _ in it.repeat(None, n)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import graphid

        if self.triple:
            aids = self.samples[index]
        else:
            aid1, aid2, label = self.samples[index]
            aids = [aid1, aid2]
            if label == graphid.core.POSTV:
                label = 1
            elif label == graphid.core.NEGTV:
                label = 0
            else:
                raise KeyError(label)
        txs = [self.aid_to_tx[aid] for aid in aids]

        samples = [self.sampler.load_positive(index=tx) for tx in txs]
        chips = [s['im'] for s in samples]

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            # if False:
            #     deps = [self.dependent.to_deterministic() for _ in chips]
            #     chips = [d(c) for d, c in zip(deps, chips)]
            # dependent2 = self.dependent.to_deterministic()
            if self.rng.rand() > .5:
                chips = [np.fliplr(c) for c in chips]
            chips = self.independent.augment_images(chips)

        chips = self.letterbox.augment_images(chips)

        chips = [
            torch.FloatTensor(c.transpose(2, 0, 1).astype(np.float32) / 255)
            for c in chips
        ]

        if self.triple:
            item = {
                'img1': chips[0],
                'img2': chips[1],
                'img3': chips[2],
            }
        else:
            item = {
                'img1': chips[0],
                'img2': chips[1],
                'label': label,
            }

        return item


def e_(e):
    u, v = e
    return (u, v) if u < v else (v, u)


def generate_positives(pccs, rng=None):
    rng = nh.util.ensure_rng(rng)
    generators = {i: nh.util.random_combinations(pcc, 2, rng=rng)
                  for i, pcc in enumerate(pccs)}
    while generators:
        to_remove = set()
        for i, gen in generators.items():
            try:
                yield e_(next(gen))
            except StopIteration:
                to_remove.add(i)
        for i in to_remove:
            generators.pop(i)


def generate_negatives(pccs, rng=None):
    rng = nh.util.ensure_rng(rng, api='python')
    generators = None
    unfinished = True
    generators = {}

    while unfinished:
        finished = set()
        if unfinished is True:
            combos = nh.util.random_combinations(pccs, 2, rng=rng)
        else:
            combos = unfinished

        for pcc1, pcc2 in combos:
            key = (pcc1, pcc2)

            if key not in generators:
                generators[key] = nh.util.random_product([pcc1, pcc2], rng=rng)
            gen = generators[key]
            try:
                edge = e_(next(gen))
                yield edge
            except StopIteration:
                finished.add(key)

        if unfinished is True:
            unfinished = set(generators.keys())

        unfinished.difference_update(finished)


def sample_triples(pccs, rng=None):
    """
    Note: does not take into account incomparable

    Example:
        >>> pccs = list(map(frozenset, [{1, 2, 3}, {4, 5}, {6}]))
        >>> gen = (sample_triples(pccs))
        >>> next(gen)
    """
    assert len(pccs) > 1
    rng = nh.util.ensure_rng(rng, api='python')
    aid_to_pcc = {aid: pcc for pcc in pccs for aid in pcc}
    all_aids = sorted(aid_to_pcc.keys())
    pos_gen = generate_positives(pccs, rng=rng)
    pos_inf = it.cycle(pos_gen)
    for u, v in pos_inf:
        this_pcc = aid_to_pcc[u]
        aid3 = rng.choice(all_aids)
        while aid3 in this_pcc:
            aid3 = rng.choice(all_aids)
        yield u, v, aid3


def sample_edges_inf(pccs, rng=None):
    """
    Note: does not take into account incomparable

    Example:
        >>> pccs = list(map(frozenset, [{1, 2, 3}, {4, 5}, {6}]))
        >>> gen = (sample_edges_inf(pccs))
        >>> next(gen)
    """
    import graphid
    rng = nh.util.ensure_rng(rng, api='python')
    pos_gen = generate_positives(pccs, rng=rng)
    neg_gen = generate_negatives(pccs, rng=rng)
    pos_inf = it.cycle((u, v, graphid.core.POSTV) for u, v in pos_gen)
    neg_inf = it.cycle((u, v, graphid.core.NEGTV) for u, v in neg_gen)
    for u, v, label in it.chain.from_iterable(zip(pos_inf, neg_inf)):
        yield u, v, label


def sample_edges_finite(pccs, max_num=1000, pos_neg_ratio=None, rng=None):
    """
    Note: does not take into account incomparable

    >>> pccs = list(map(frozenset, [{1, 2, 3}, {4, 5}, {6}]))
    >>> list(sample_edges_finite(pccs))
    """
    import graphid
    rng = nh.util.ensure_rng(rng, api='python')
    # Simpler very randomized sample strategy
    max_pos = int(max_num) // 2
    max_neg = int(max_num) - max_pos

    pos_pairs = [edge for i, edge in zip(range(max_pos), generate_positives(pccs, rng=rng))]

    if pos_neg_ratio is not None:
        max_neg = min(int(pos_neg_ratio * len(pos_pairs)), max_neg)

    neg_pairs = [edge for i, edge in zip(range(max_neg), generate_negatives(pccs, rng=rng))]

    labeled_pairs = [
        (graphid.core.POSTV, pos_pairs),
        (graphid.core.NEGTV, neg_pairs),
    ]

    samples = [(aid1, aid2, label)
                    for label, pairs in labeled_pairs
                    for aid1, aid2 in pairs]
    return samples


def main():
    """
    CommandLine:
        python examples/ggr_matching.py --help

        # Test Runs:
            # use a very small input dimension to test things out
            python examples/ggr_matching.py --dbname PZ_MTEST --workers=0 --dim=32 --xpu=cpu

            # test that GPU works
            python examples/ggr_matching.py --dbname PZ_MTEST --workers=0 --dim=32 --xpu=gpu0

            # test that running at a large size works
            python examples/ggr_matching.py --dbname PZ_MTEST --workers=6 --dim=416 --xpu=gpu0

        # Real Run:
        python examples/ggr_matching.py --dbname GZ_Master1 --workers=6 --dim=512 --xpu=gpu0 --batch_size=10 --lr=0.00001 --nice=gzrun
        python examples/ggr_matching.py --dbname GZ_Master1 --workers=6 --dim=512 --xpu=gpu0 --batch_size=6 --lr=0.001 --nice=gzrun

    Notes:
        # Some database names
        PZ_Master1
        GZ_Master1
        RotanTurtles
        humpbacks_fb
    """
    print('PARSE')
    import xinspect
    parser = xinspect.auto_argparse(setup_harn)
    args, unknown = parser.parse_known_args()
    ns = args.__dict__.copy()
    print('SETUP')
    print('ns = {!r}'.format(ns))
    harn = setup_harn(**ns)
    print('INIT')
    harn.initialize()
    print('RUN')
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/examples/ggr_matching.py --help
        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=8 --nice=ggr2-test
        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=8 --nice=ggr2-test-v2
    """
    main()
