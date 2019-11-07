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
import torchvision  # NOQA
import ndsampler
import itertools as it
import torch.nn.functional as F
import torch.utils.data.sampler as torch_sampler


def setup_sampler(config):
    workdir = nh.configure_workdir(config,
                                   workdir=join('~/work/siam-ibeis2', config['dbname']))

    # TODO: cleanup and hook into ibeis AI
    if config['dbname'] == 'ggr2':
        print('Creating torch CocoDataset')

        root = ub.truepath('~/data/')
        print('root = {!r}'.format(root))

        train_dset = ndsampler.CocoDataset(
            data=join(root, 'ggr2-coco/annotations/instances_train2018.json'),
            img_root=join(root, 'ggr2-coco/images/train2018'),
        )
        train_dset.hashid = 'ggr2-coco-train2018'

        vali_dset = ndsampler.CocoDataset(
            data=join(root, 'ggr2-coco/annotations/instances_val2018.json'),
            img_root=join(root, 'ggr2-coco/images/val2018'),
        )
        vali_dset.hashid = 'ggr2-coco-val2018'

        print('Creating samplers')
        samplers = {
            'train': ndsampler.CocoSampler(train_dset, workdir=workdir),
            'vali': ndsampler.CocoSampler(vali_dset, workdir=workdir),
        }
    else:
        raise KeyError(config['dbname'])

    return samplers, workdir


def setup_datasets(config):
    """
    config = parse_config(dbname='ggr2')
    datasets, workdir = setup_datasets(config)
    """

    workdir = nh.configure_workdir(config,
                                   workdir=join('~/work/siam-ibeis2', config['dbname']))
    # TODO: cleanup and hook into ibeis AI
    dim = config['dim']
    if config['dbname'] == 'ggr2':
        samplers, workdir = setup_sampler(config)
        datasets = {
            'train': AnnotCocoDataset(samplers['train'], workdir, dim=dim,
                                      augment=True),
            'vali': AnnotCocoDataset(samplers['vali'], workdir, dim=dim,
                                     augment=False),
        }
    else:
        from ibeis_utils import randomized_ibeis_dset
        # TODO: triple
        datasets = randomized_ibeis_dset(config['dbname'], dim=dim)

    for k, v in datasets.items():
        print('* len({}) = {}'.format(k, len(v)))

    return datasets, workdir


def parse_config(**kwargs):
    """
    Ignore:
        I think xdoctest.auto_argparse is pretty cool. It can pick up what the
        kwargs are even though they are passed between functions.

        print('PARSE')
        import xinspect
        parser = xinspect.auto_argparse(setup_harn)
        args, unknown = parser.parse_known_args()
        ns = args.__dict__.copy()
        print(parser.format_help())
        print(ub.repr2(ns))
    """
    config = {}
    config['norm_desc'] = kwargs.get('norm_desc', False)

    config['init'] = kwargs.get('init', 'kaiming_normal')
    config['pretrained'] = config.get('pretrained', ub.argval('--pretrained', default=None))

    config['margin'] = kwargs.get('margin', 1.0)
    config['soft'] = kwargs.get('soft', True)
    config['xpu'] = kwargs.get('xpu', 'argv')
    config['nice'] = kwargs.get('nice', 'untitled')
    config['workdir'] = kwargs.get('workdir', None)
    config['workers'] = int(kwargs.get('workers', 0))
    config['dbname'] = kwargs.get('dbname', 'ggr2')
    config['bstep'] = int(kwargs.get('bstep', 1))

    # config['dim'] = int(kwargs.get('dim', 416))
    # config['batch_size'] = int(kwargs.get('batch_size', 6))
    config['optim'] = kwargs.get('optim', 'sgd')
    config['scheduler'] = kwargs.get('scheduler', 'onecycle70')
    config['lr'] = float(kwargs.get('lr', 0.001))
    config['decay'] = float(kwargs.get('decay', 1e-5))

    config['dim'] = int(kwargs.get('dim', 300))
    config['batch_size'] = kwargs.get('batch_size', None)
    config['p'] = int(kwargs.get('p', 10))
    config['k'] = int(kwargs.get('k', 4))
    # config['scheduler'] = kwargs.get('scheduler', 'dolphin')
    # config['optim'] = kwargs.get('optim', 'adamw')
    # config['lr'] = float(kwargs.get('lr', 3e-4))
    # config['decay'] = float(kwargs.get('decay', 0))

    try:
        config['batch_size'] = int(config['batch_size'])
    except Exception:
        pass

    try:
        config['lr'] = float(config['lr'])
    except Exception:
        pass

    try:
        config['margin'] = float(config['margin'])
    except ValueError:
        pass

    return config


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
        norm_desc (bool): if True normalizes the descriptors
        pretrained (PathLike): path to a compatible pretrained model
        margin (float): margin for loss criterion
        soft (bool): use soft margin

    Example:
        >>> harn = setup_harn(dbname='PZ_MTEST')
        >>> harn.initialize()
    """
    config = parse_config(**kwargs)

    nh.configure_hacks(config)
    datasets, workdir = setup_datasets(config)

    # if config['scheduler'] == 'dolphin':
    #     scheduler_ = (nh.schedulers.ListedScheduler, { 'points': {
    #             'lr': {
    #                 0  : config['lr'] * 1.0,
    #                 10  : config['lr'] / 2,
    #                 20  : config['lr'] / 4,
    #                 30  : config['lr'] / 8,
    #             },
    #             # 'momentum': {
    #             #     0  : 0.95,
    #             #     1  : 0.85,
    #             #     2  : 0.95,
    #             # },
    #         },
    #         'interpolation': 'left',
    #     })
    # else:
    # scheduler_ = nh.Scheduler.coerce(config, scheduler='onecycle70')

    hyper = nh.HyperParams(**{
        'nice': config['nice'],
        'workdir': config['workdir'],
        'datasets': datasets,
        'loaders': {
            # nh.Loaders.coerce(datasets, config),
            tag: dset.make_loader(
                shuffle=(tag == 'train'),
                batch_size=config['batch_size'],
                k=config['k'],
                p=config['p'],
                num_workers=config['workers'],
            )
            for tag, dset in datasets.items()
        },
        'xpu': nh.XPU.cast(config['xpu']),

        'model': (nh.models.DescriptorNetwork, {
            'input_shape': (1, 3, config['dim'], config['dim']),
            'norm_desc': config['norm_desc'],
            'hidden_channels': [512, 256]
        }),
        'initializer': nh.Initializer.coerce(config),
        'optimizer': nh.Optimizer.coerce(config),
        'scheduler': nh.Scheduler.coerce(config, scheduler='onecycle70'),
        'criterion': (nh.criterions.TripletLoss, {
            'margin': config['margin'],
            'soft': config['soft'],
        }),
        # 'scheduler': scheduler_,
        # 'criterion': criterion_,
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

    def after_initialize(harn, **kw):
        harn.confusion_vectors = []
        harn._has_preselected = False
        harn.POS_LABEL = 1
        harn.NEG_LABEL = 0

    def before_epochs(harn):
        verbose = 0
        for tag, dset in harn.datasets.items():
            if isinstance(dset, torch.utils.data.Subset):
                dset = dset.dataset

            if not hasattr(dset, 'preselect'):
                continue

            # Presample negatives before each epoch.
            if tag == 'train':
                if not harn._has_preselected:
                    harn.log('Presample {} dataset'.format(tag))
                # Randomly resample training negatives every epoch
                dset.preselect(verbose=verbose)
            else:
                if not harn._has_preselected:
                    harn.log('Presample {} dataset'.format(tag))
                    dset.preselect(verbose=verbose)
        harn._has_preselected = True

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure

        Ignore:
            from ggr_matching import *
            harn = setup_harn(dbname='ggr2', batch_size=21).initialize()
            raw_batch = harn._demo_batch(raw=1)
            raw_batch['chip'].shape
            raw_batch['nx'].shape
        """
        batch = {
            'chip': harn.xpu.move(raw_batch['chip']),
            'nx': harn.xpu.move(raw_batch['nx']),
        }
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader
        """
        inputs = batch['chip']
        outputs = harn.model(inputs)
        dvecs = outputs['dvecs']

        pos_dists, neg_dists, triples = harn.criterion.hard_triples(
            dvecs, batch['nx'])

        loss = harn.criterion(pos_dists, neg_dists)
        outputs['triples'] = triples
        outputs['chip'] = batch['chip']
        outputs['nx'] = batch['nx']
        return outputs, loss

    def _decode(harn, outputs):
        decoded = {}

        A, P, N = np.array(outputs['triples']).T
        chips_ = outputs['chip'].data.cpu().numpy()
        nxs_ = outputs['nx'].data.cpu().numpy()
        decoded['imgs'] = [chips_[A], chips_[P], chips_[N]]
        decoded['nxs'] = [nxs_[A], nxs_[P], nxs_[N]]

        dvecs_ = outputs['dvecs']
        dvecs = [dvecs_[A], dvecs_[P], dvecs_[N]]
        # decoded['dvecs'] = [d.data.cpu().numpy() for d in dvecs]

        distAP = F.pairwise_distance(dvecs[0], dvecs[1], p=2)
        distAN = F.pairwise_distance(dvecs[0], dvecs[2], p=2)
        decoded['distAP'] = distAP.data.cpu().numpy()
        decoded['distAN'] = distAN.data.cpu().numpy()
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

        # Record metrics for epoch scores
        n = len(decoded['distAP'])
        harn.confusion_vectors.append(([harn.POS_LABEL] * n, decoded['distAP']))
        harn.confusion_vectors.append(([harn.NEG_LABEL] * n, decoded['distAN']))

    def on_epoch(harn):
        """ custom callback """
        from sklearn import metrics
        margin = harn.hyper.criterion_params['margin']
        epoch_metrics = {}

        if harn.confusion_vectors:
            y_true = np.hstack([r for r, p in harn.confusion_vectors])
            y_dist = np.hstack([p for r, p in harn.confusion_vectors])

            pos_dist = np.nanmean(y_dist[y_true == harn.POS_LABEL])
            neg_dist = np.nanmean(y_dist[y_true == harn.NEG_LABEL])

            # Transform distance into a probability-like space
            y_probs = torch.sigmoid(torch.Tensor(-(y_dist - margin))).numpy()

            brier = y_probs - y_true

            y_pred1 = (y_dist <= margin).astype(y_true.dtype)
            y_pred2 = (y_dist > neg_dist).astype(y_true.dtype)

            import xdev
            with xdev.embed_on_exception_context:

                accuracy = (y_true == y_pred1).mean()
                mcc = metrics.matthews_corrcoef(y_true, y_pred1)
                brier = ((y_probs - y_true) ** 2).mean()

                epoch_metrics = {
                    'mcc': mcc,
                    'brier': brier,
                    'accuracy': accuracy,
                    'pos_dist': pos_dist,
                    'neg_dist': neg_dist,
                    'triple_acc': (y_true == y_pred2).mean(),
                    'triple_mcc': metrics.matthews_corrcoef(y_true, y_pred2),
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
        tostack = []
        fontkw = {
            'fontScale': 1.0,
            'thickness': 2
        }
        n = min(limit, len(decoded['imgs'][0]))
        dsize = (300, 300)
        import cv2
        for i in range(n):
            ims = [g[i].transpose(1, 2, 0) for g in decoded['imgs']]
            ims = [cv2.resize(g, dsize) for g in ims]
            img = nh.util.stack_images(ims, overlap=-2, axis=1)
            img = nh.util.atleast_3channels(img)
            triple_nxs = [n[i] for n in decoded['nxs']]

            if False:
                triple_dvecs = [d[i] for d in decoded['dvecs']]
                da, dp, dn = triple_dvecs
                distAP = np.sqrt(((da - dp) ** 2).sum())
                distAN = np.sqrt(((da - dn) ** 2).sum())
                print('distAP = {!r}'.format(distAP))
                print('distAN = {!r}'.format(distAN))
                print('----')

            text = 'distAP={:.3f} -- distAN={:.3f} -- {}'.format(
                decoded['distAP'][i],
                decoded['distAN'][i],
                str(triple_nxs),
            )
            img = (img * 255).astype(np.uint8)
            img = nh.util.draw_text_on_image(img, text,
                                             org=(2, img.shape[0] - 2),
                                             color='blue', **fontkw)
            tostack.append(img)
        stacked = nh.util.stack_images_grid(tostack, overlap=-10,
                                            bg_value=(10, 40, 30),
                                            axis=1, chunksize=3)
        return stacked


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


class MatchingSamplerPK(ub.NiceRepr, torch_sampler.BatchSampler):
    """

    Samples random triples from a PCC-complient dataset

    Args:
        torch_dset (Dataset): something that can sample PCC-based annots
        p (int): number of individuals sampled per batch
        k (int): number of annots sampled per individual within a batch
        batch_size (int): if specified, k is adjusted to an appropriate length
        drop_last (bool): ignored
        num_batches (int): length of the loader
        rng (int | Random, default=None): random seed
        shuffle (bool): if False rng is ignored and getitem is deterministic

    Example:
        >>> datasets, workdir = setup_datasets()
        >>> torch_dset = datasets['vali']
        >>> batch_sampler = self = MatchingSamplerPK(torch_dset, shuffle=True)
        >>> for indices in batch_sampler:
        >>>     print('indices = {!r}'.format(indices))

    Ignore:
        import sys
        sys.path.append('/home/joncrall/code/netharn/examples')
        from ggr_matching import *
        config = parse_config(dbname='ggr2')
        datasets, workdir = setup_datasets(config)

        torch_dset = datasets['vali']
        batch_sampler = self = MatchingSamplerPK(torch_dset, shuffle=True)

        for indices in batch_sampler:
            print('indices = {!r}'.format(indices))
    """
    def __init__(self, torch_dset, p=21, k=4, batch_size=None, drop_last=False,
                 rng=None, shuffle=False, num_batches=None, replace=True):
        self.torch_dset = torch_dset
        self.drop_last = drop_last

        self.shuffle = shuffle

        if replace is False:
            raise NotImplementedError(
                'We currently cant sample without replacement')

        self.pccs = torch_dset.pccs

        self.multitons = [pcc for pcc in self.pccs if len(pcc) > 1]

        self.aid_to_index = self.torch_dset.aid_to_index

        if getattr(self.torch_dset, '__hasgraphid__', False):
            raise NotImplementedError('TODO: graphid API sampling')
        else:
            # Compute the total possible number of triplets
            # Its probably a huge number, but lets do it anyway.
            # --------
            # In the language of graphid (See Jon Crall's 2017 thesis)
            # The matching graph for MNIST is fully connected graph.  All
            # pairs of annotations with the same label have a positive edge
            # between them. All pairs of annotations with a different label
            # have a negative edge between them. There are no incomparable
            # edges. Therefore for each PCC, A the number of triples it can
            # contribute is the number of internal positive edges
            # ({len(A) choose 2}) times the number of outgoing negative edges.
            # ----
            # Each pair of positive examples could be a distinct triplet
            # For each of these any negative could be chosen
            # The number of distinct triples contributed by this PCC is the
            # product of num_pos_edges and num_neg_edges.
            import scipy
            self.num_triples = 0
            self.num_pos_edges = 0
            default_num_items = 0
            for pcc in self.pccs:
                num_pos_edges = scipy.special.comb(len(pcc), 2)
                if num_pos_edges > 0:
                    default_num_items += len(pcc)
                other_pccs = [c for c in self.pccs if c is not pcc]
                num_neg_edges = sum(len(c) for c in other_pccs)
                self.num_triples += num_pos_edges * num_neg_edges
                self.num_pos_edges += num_pos_edges

        # self.p = batch_size // k
        # self.k = k
        p = min(len(self.multitons), p)
        k = min(max(len(p) for p in self.pccs), k)

        if batch_size is None:
            batch_size = p * k
        else:
            k = batch_size // p

        if num_batches is None:
            num_batches = default_num_items // batch_size

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.p = p  # PCCs per batch
        self.k = k  # Items per PCC per batch
        assert self.k > 1
        assert self.p > 1
        print('self.p = {!r}'.format(self.p))
        print('self.k = {!r}'.format(self.k))

        self.rng = nh.util.ensure_rng(rng, api='python')

    def __iter__(self):
        for index in range(len(self)):
            indices = self[index]
            yield indices

    def __getitem__(self, index):
        if not self.shuffle:
            self.rng = nh.util.ensure_rng(index, api='python')

        sub_pccs = self.rng.sample(self.multitons, self.p)

        groups = []
        for sub_pcc in sub_pccs:
            aids = self.rng.sample(sub_pcc, min(self.k, len(sub_pcc)))
            groups.append(aids)

        nhave = sum(map(len, groups))
        while nhave < self.batch_size:
            sub_pcc = self.rng.choice(self.pccs)
            aids = self.rng.sample(sub_pcc, min(self.k, len(sub_pcc)))
            groups.append(aids)
            nhave = sum(map(len, groups))
            overshoot = nhave - self.batch_size
            if overshoot:
                groups[-1] = groups[-1][:-overshoot]

        batch_aids = sorted(ub.flatten(groups))
        indices = [self.aid_to_index[aid] for aid in batch_aids]
        return indices

    def __len__(self):
        return self.num_batches


class AnnotCocoDataset(torch.utils.data.Dataset, ub.NiceRepr):
    """
    Generates annotations with name labels. This is essentially a dataset for
    graphid AnnotInfr nodes.

    Example:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/netharn/examples')
        >>> from ggr_matching import *
        >>> config = parse_config(dbname='ggr2')
        >>> samplers, workdir = setup_sampler(config)
        >>> torch_dset = self = AnnotCocoDataset(samplers['vali'], workdir, augment=True)
        >>> assert len(torch_dset) == 454
        >>> index = 0
        >>> item = torch_dset[index]
        >>> import netharn as nh
        >>> nh.util.autompl()
        >>> nh.util.imshow(item['chip'])
        >>> torch_loader = torch_dset.make_loader()
        >>> raw_batch = ub.peek(torch_loader)
        >>> stacked = nh.util.stack_images_grid(raw_batch['chip'].numpy().transpose(0, 2, 3, 1), overlap=-1)
        >>> nh.util.imshow(stacked)

        for batch_idxs in torch_loader.batch_sampler:
            print('batch_idxs = {!r}'.format(batch_idxs))
    """
    def __init__(self, sampler, workdir=None, augment=False, dim=416):
        print('make MatchingCocoDataset')

        cacher = ub.Cacher('pccs_v1', cfgstr=sampler.dset.tag, verbose=True)
        pccs = cacher.tryload()
        if pccs is None:
            pccs = extract_ggr_pccs(sampler.dset)
            cacher.save(pccs)
        self.pccs = pccs

        self.sampler = sampler

        self.aids = sorted(ub.flatten(self.pccs))
        self.index_to_aid = {index: aid for index, aid in enumerate(self.aids)}
        self.nx_to_pcc = {nx: pcc for nx, pcc in enumerate(self.pccs)}

        self.aid_to_index = {aid: index for index, aid in self.index_to_aid.items()}
        self.aid_to_nx = {aid: nx for nx, pcc in self.nx_to_pcc.items() for aid in pcc}
        self.aid_to_tx = {aid: tx for tx, aid in
                          enumerate(sampler.regions.targets['aid'])}

        print('Finished sampling')
        window_dim = dim
        self.dim = window_dim
        self.window_dim = window_dim
        self.dims = (window_dim, window_dim)

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
        self.letterbox = nh.data.transforms.Resize(target_size=self.dims,
                                                   mode='letterbox')

    def make_loader(self, batch_size=None, num_workers=0, shuffle=False,
                    pin_memory=False, k=4, p=21, drop_last=False):
        """
        Example:
            >>> config = parse_config(dbname='ggr2')
            >>> samplers, workdir = setup_sampler(config)
            >>> torch_dset = self = AnnotCocoDataset(samplers['vali'], workdir, augment=True)
            >>> torch_loader = torch_dset.make_loader()
            >>> raw_batch = ub.peek(torch_loader)
            >>> for batch_idxs in torch_loader.batch_sampler:
            >>>     print('batch_idxs = {!r}'.format(batch_idxs))
        """
        torch_dset = self
        batch_sampler = MatchingSamplerPK(self, batch_size=batch_size, k=k,
                                          p=p, shuffle=shuffle)
        torch_loader = torch.utils.data.DataLoader(torch_dset,
                                                   num_workers=num_workers,
                                                   # batch_size=batch_size,
                                                   # shuffle=shuffle,
                                                   batch_sampler=batch_sampler,
                                                   pin_memory=pin_memory)
        return torch_loader

    def __nice__(self):
        return str(len(self))

    def __len__(self):
        return len(self.pccs)

    def __getitem__(self, index):
        # import graphid
        aid = self.aids[index]
        tx = self.aid_to_tx[aid]
        nx = self.aid_to_nx[aid]
        sample = self.sampler.load_positive(index=tx)
        chip = sample['im']

        if self.augmenter:
            # Ensure the same augmentor is used for bboxes and iamges
            # if False:
            #     deps = [self.dependent.to_deterministic() for _ in chips]
            #     chips = [d(c) for d, c in zip(deps, chips)]
            # dependent2 = self.dependent.to_deterministic()
            # if self.rng.rand() > .5:
            #     chips = [np.fliplr(c) for c in chips]
            chip = self.independent.augment_image(chip)

        chip = self.letterbox.augment_image(chip)
        chip_chw = chip.transpose(2, 0, 1).astype(np.float32)
        chip_chw = torch.FloatTensor(chip_chw / 255.0)

        item = {
            'chip': chip_chw,
            'aid': aid,
            'nx': nx,
            # TODO: note if flipped left/right
        }
        return item


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

    parser.add_argument(
        '--lrtest', action='store_true',
        help='Run Leslie Smith\'s LR range test')
    parser.add_argument(
        '--interact', action='store_true',
        help='Interact with the range test')
    args, unknown = parser.parse_known_args()
    ns = args.__dict__.copy()

    args.interact |= args.lr == 'interact'
    args.lrtest |= (args.lr == 'test' or args.interact)

    if args.lrtest or args.interact:
        # TODO:
        # - [ ] tweak setup_harn so running the lr-range-test isn't awkward
        from netharn.prefit.lr_tests import lr_range_test
        ns['lr'] = 1e-99

        if args.interact:
            nh.util.autompl()
            import matplotlib.pyplot as plt

        harn = setup_harn(**ns)
        harn.initialize()
        # TODO: We could cache the result based on the netharn
        # hyperparameters. This would let us integrate with the
        # default fit harness.
        result = lr_range_test(harn)
        print('result.recommended_lr = {!r}'.format(result.recommended_lr))

        if args.interact:
            result.draw()
            plt.show()

        # Seed value with test result
        ns['lr'] = result.recommended_lr
        harn = setup_harn(**ns).initialize()
    else:
        harn = setup_harn(**ns)

    harn.initialize()
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/examples/ggr_matching.py --help
        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=8 --nice=ggr2-test

        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=64 --nice=TEST_JAN21 --lrtest --interact

        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=8 --nice=ggr2-test-v2
        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=8 --nice=ggr2-test-v3
        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=8 --nice=ggr2-triple-test-v4

        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=26 --dims=416 --nice=ggr2-test-bugfix
        python ~/code/netharn/examples/ggr_matching.py --workers=6 --xpu=0 --dbname=ggr2 --batch_size=32 --dims=416 --nice=ggr2-test-bugfix-v2 --lr=0.003


    """
    main()
