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
import cv2
import netharn as nh
import torch
import torchvision  # NOQA
import ndsampler
from sklearn import metrics
import kwimage
import kwarray


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
        harn._has_preselected = False
        harn.POS_LABEL = 1
        harn.NEG_LABEL = 0
        # BUG: should have one for each tag
        harn.confusion_vectors = kwarray.DataFrameLight(
            columns=['y_true', 'y_dist']
        )

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
            'cpu_nx': raw_batch['nx'],
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

        try:
            # Takes roughly 30% of the time for batch sizes of 128
            labels = batch['cpu_nx']
            if harn.current_tag == 'train':
                info = harn.criterion.mine_negatives(
                    dvecs, labels,
                    mode='hardest',
                    # mode='moderate'
                )
            else:
                info = harn.criterion.mine_negatives(dvecs, labels,
                                                     mode='consistent')
            pos_dists = info['pos_dists']
            neg_dists = info['neg_dists']
            triples = info['triples']
        except RuntimeError:
            raise nh.exceptions.SkipBatch

        _loss_parts = {}
        _loss_parts['triplet'] = harn.criterion(pos_dists, neg_dists)

        if False and harn.epoch > 0:
            # Experimental extra loss term:
            y_dist = torch.cat([pos_dists, neg_dists], dim=0)
            margin = harn.criterion.margin
            y_probs = (-(y_dist - margin)).sigmoid()

            # Use MSE to encourage the average batch-hard-case prob to be 0.5
            prob_mean = y_probs.mean()
            prob_std = y_probs.std()

            target_mean = torch.FloatTensor([0.5]).to(prob_mean.device)
            target_std = torch.FloatTensor([0.1]).to(prob_mean.device)

            _loss_parts['pmean'] = 0.1 * torch.nn.functional.mse_loss(prob_mean, target_mean)
            _loss_parts['pstd'] = 0.1 * torch.nn.functional.mse_loss(prob_std.clamp(0, 0.1), target_std)

            # Encourage descriptor vecstors to have a small squared-gradient-magnitude (ref girshik)
            _loss_parts['sgm'] = 0.0003 * (dvecs ** 2).sum(dim=0).mean()

        loss = sum(_loss_parts.values())
        harn._loss_parts = _loss_parts

        outputs['triples'] = triples
        outputs['chip'] = batch['chip']
        outputs['nx'] = batch['nx']
        outputs['distAP'] = pos_dists
        outputs['distAN'] = neg_dists
        return outputs, loss

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
        batch_metrics = ub.odict()
        for key, value in harn._loss_parts.items():
            if value is not None and torch.is_tensor(value):
                batch_metrics[key + '_loss'] = float(
                    value.data.cpu().numpy().item())

        bx = harn.bxs[harn.current_tag]

        if bx < 4:
            decoded = harn._decode(outputs)
            stacked = harn._draw_batch(batch, decoded)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'batch_{}_epoch_{}.jpg'.format(bx, harn.epoch))
            kwimage.imwrite(fpath, stacked)

        # Record metrics for epoch scores
        n = len(outputs['distAP'])

        harn.confusion_vectors._data['y_true'].extend([harn.POS_LABEL] * n)
        harn.confusion_vectors._data['y_dist'].extend(outputs['distAP'].data.cpu().numpy().tolist())

        harn.confusion_vectors._data['y_true'].extend([harn.NEG_LABEL] * n)
        harn.confusion_vectors._data['y_dist'].extend(outputs['distAN'].data.cpu().numpy().tolist())
        return batch_metrics

    def on_epoch(harn):
        """ custom callback """
        margin = harn.hyper.criterion_params['margin']
        assert margin == harn.criterion.margin
        epoch_metrics = {}

        if len(harn.confusion_vectors):
            margin = harn.criterion.margin
            y_true = np.array(harn.confusion_vectors['y_true'], dtype=np.uint8)
            y_dist = np.array(harn.confusion_vectors['y_dist'], dtype=np.float32)

            # Transform distance into a probability-like space
            y_probs = torch.Tensor(-(y_dist - margin)).sigmoid().numpy()

            pos_flags = np.where(y_true == harn.POS_LABEL)[0]
            neg_flags = np.where(y_true == harn.NEG_LABEL)[0]

            pos_dists = y_dist[pos_flags]
            neg_dists = y_dist[neg_flags]

            pos_probs = y_probs[pos_flags]
            neg_probs = y_probs[neg_flags]

            pos_dist = np.nanmean(pos_dists)
            neg_dist = np.nanmean(neg_dists)

            y_pred = (pos_probs < neg_probs).astype(np.uint8)

            # How should be choose a threshold here?
            thresh = median_prob = np.median(y_probs)
            # thresh = 0.5
            y_pred = (y_probs > thresh).astype(np.uint8)

            accuracy = (y_true == y_pred).astype(np.uint8).mean()
            brier = ((y_probs - y_true) ** 2).mean()
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'invalid value')
                mcc = metrics.matthews_corrcoef(y_true, y_pred)

            epoch_metrics = {
                'mcc': mcc,
                'brier': brier,
                'accuracy': accuracy,
                'pos_dist': pos_dist,
                'neg_dist': neg_dist,
                'median_prob': median_prob,
            }

        # Clear scores for next epoch
        harn.confusion_vectors.clear()
        return epoch_metrics

    def _decode(harn, outputs):
        decoded = {}

        triple_idxs = np.array(outputs['triples']).T
        A, P, N = triple_idxs
        chips_ = outputs['chip'].data.cpu().numpy()
        nxs_ = outputs['nx'].data.cpu().numpy()

        decoded['triple_idxs'] = triple_idxs
        decoded['triple_imgs'] = [chips_[A], chips_[P], chips_[N]]
        decoded['triple_nxs'] = [nxs_[A], nxs_[P], nxs_[N]]

        decoded['distAP'] = outputs['distAP'].data.cpu().numpy()
        decoded['distAN'] = outputs['distAN'].data.cpu().numpy()
        return decoded

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
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked, colorspace='rgb', doclf=True)
            >>> kwplot.show_if_requested()
        """
        tostack = []
        fontkw = {
            'fontScale': 1.0,
            'thickness': 2
        }
        n = min(limit, len(decoded['triple_imgs'][0]))
        dsize = (300, 300)
        for i in range(n):
            ims = [g[i].transpose(1, 2, 0) for g in decoded['triple_imgs']]
            ims = [cv2.resize(g, dsize) for g in ims]
            ims = [kwimage.atleast_3channels(g) for g in ims]
            triple_nxs = [n[i] for n in decoded['triple_nxs']]

            text = 'distAP={:.3g} -- distAN={:.3g} -- {}'.format(
                decoded['distAP'][i],
                decoded['distAN'][i],
                str(triple_nxs),
            )
            color = (
                'dodgerblue' if decoded['distAP'][i] < decoded['distAN'][i]
                else 'orangered')

            img = kwimage.stack_images(
                ims, overlap=-2, axis=1,
                bg_value=(10 / 255, 40 / 255, 30 / 255)
            )
            img = (img * 255).astype(np.uint8)
            img = kwimage.draw_text_on_image(img, text,
                                             org=(2, img.shape[0] - 2),
                                             color=color, **fontkw)
            tostack.append(img)

        stacked = kwimage.stack_images_grid(tostack, overlap=-10,
                                            bg_value=(30, 10, 40),
                                            axis=1, chunksize=3)
        return stacked


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
        >>> kwplot.autompl()
        >>> kwplot.util.imshow(item['chip'])
        >>> torch_loader = torch_dset.make_loader()
        >>> raw_batch = ub.peek(torch_loader)
        >>> stacked = kwplot.stack_images_grid(raw_batch['chip'].numpy().transpose(0, 2, 3, 1), overlap=-1)
        >>> kwplot.imshow(stacked)

        for batch_idxs in torch_loader.batch_sampler:
            print('batch_idxs = {!r}'.format(batch_idxs))
    """
    def __init__(self, sampler, workdir=None, augment=False, dim=416):
        print('make AnnotCocoDataset')

        cacher = ub.Cacher('aid_pccs_v2', cfgstr=sampler.dset.tag, verbose=True)
        aid_pccs = cacher.tryload()
        if aid_pccs is None:
            aid_pccs = extract_ggr_pccs(sampler.dset)
            cacher.save(aid_pccs)
        self.aid_pccs = aid_pccs
        self.sampler = sampler

        self.aids = sorted(ub.flatten(self.aid_pccs))
        self.aid_to_index = aid_to_index = {aid: index for index, aid in enumerate(self.aids)}

        # index pccs
        self.index_pccs = [frozenset(aid_to_index[aid] for aid in pcc)
                           for pcc in self.aid_pccs]

        self.nx_to_aidpcc   = {nx: pcc for nx, pcc in enumerate(  self.aid_pccs)}
        self.nx_to_indexpcc = {nx: pcc for nx, pcc in enumerate(self.index_pccs)}

        self.aid_to_nx   = {  aid: nx for nx, pcc in self.nx_to_aidpcc.items()   for aid   in pcc}
        self.index_to_nx = {index: nx for nx, pcc in self.nx_to_indexpcc.items() for index in pcc}

        self.aid_to_tx = {aid: tx for tx, aid in enumerate(sampler.regions.targets['aid'])}

        window_dim = dim
        self.dim = window_dim
        self.window_dim = window_dim
        self.dims = (window_dim, window_dim)

        self.rng = kwarray.ensure_rng(0)
        if augment:
            import imgaug.augmenters as iaa
            self.independent = iaa.Sequential([
                iaa.Sometimes(0.2, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Crop(percent=(0, .2)),
            ])
            self.dependent = iaa.Sequential([
                iaa.Fliplr(p=.5)
            ])
            # NOTE: we are only using `self.augmenter` to make a hyper hashid
            # in __getitem__ we invoke transform explicitly for fine control
            self.augmenter = iaa.Sequential([
                self.independent,
                self.dependent,
            ])
        else:
            self.augmenter = None
        self.letterbox = nh.data.transforms.Resize(
            target_size=self.dims,
            fill_color=0,
            mode='letterbox'
        )

    def make_loader(self, batch_size=None, num_workers=0, shuffle=False,
                    num_batches=None, pin_memory=False, k=4, p=21,
                    drop_last=False):
        """
        Example:
            >>> config = parse_config(dbname='ggr2')
            >>> samplers, workdir = setup_sampler(config)
            >>> torch_dset = self = AnnotCocoDataset(samplers['vali'], workdir, augment=True)
            >>> torch_loader = torch_dset.make_loader()
            >>> raw_batch = ub.peek(torch_loader)
            >>> for batch_idxs in torch_loader.batch_sampler:
            >>>     print('batch_idxs = {!r}'.format(batch_idxs))
            >>>     print('batch_aids = {!r}'.format([self.aids[x] for x in batch_idxs]))
            >>>     print('batch_nxs = {!r}'.format(ub.dict_hist([self.index_to_nx[x] for x in batch_idxs])))
        """
        torch_dset = self
        batch_sampler = nh.data.MatchingSamplerPK(self.index_pccs,
                                                  batch_size=batch_size, k=k,
                                                  num_batches=num_batches,
                                                  p=p, shuffle=shuffle)
        print('batch_sampler.batch_size = {!r}'.format(batch_sampler.batch_size))
        print('batch_sampler.k = {!r}'.format(batch_sampler.k))
        print('batch_sampler.p = {!r}'.format(batch_sampler.p))
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
        return len(self.aids)

    def __getitem__(self, index):
        # import graphid
        aid = self.aids[index]
        tx = self.aid_to_tx[aid]
        nx = self.aid_to_nx[aid]

        assert nx == self.index_to_nx[index]

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

        # resolve duplicate reviews (take the last one)
        aid2_to_decision = {}
        for aid2, decision in annot['review_ids']:
            aid2_to_decision[aid2] = decision

        for aid2, decision in aid2_to_decision.items():
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


def setup_sampler(config):
    workdir = nh.configure_workdir(config,
                                   workdir=join('~/work/siam-ibeis2', config['dbname']))

    # TODO: cleanup and hook into ibeis AI
    if config['dbname'] == 'ggr2':
        print('Creating torch CocoDataset')

        root = ub.expandpath('~/data/')
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
    if config['dbname'] == 'ggr2-revised':
        print('Creating torch CocoDataset')

        root = ub.expandpath('~/data/ggr2.coco.revised')
        print('root = {!r}'.format(root))

        train_dset = ndsampler.CocoDataset(
            data=join(root, 'annotations/instances_train2019.json'),
            img_root=join(root, 'images/train2019'),
        )
        train_dset.hashid = 'ggr2-coco-revised-train2019'

        vali_dset = ndsampler.CocoDataset(
            data=join(root, 'annotations/instances_val2019.json'),
            img_root=join(root, 'images/val2019'),
        )
        vali_dset.hashid = 'ggr2-coco-revised-val2019'

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
                                   workdir=join('~/work/ggr', config['dbname']))
    # TODO: cleanup and hook into ibeis AI
    dim = config['dim']
    if config['dbname'].startswith('ggr2'):
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

    config['margin'] = kwargs.get('margin', 3.0)
    config['soft'] = kwargs.get('soft', False)
    config['xpu'] = kwargs.get('xpu', 'argv')
    config['nice'] = kwargs.get('nice', 'untitled')
    config['workdir'] = kwargs.get('workdir', None)
    config['workers'] = int(kwargs.get('workers', 6))
    config['dbname'] = kwargs.get('dbname', 'ggr2')
    config['bstep'] = int(kwargs.get('bstep', 1))

    # config['dim'] = int(kwargs.get('dim', 416))
    # config['batch_size'] = int(kwargs.get('batch_size', 6))
    config['optim'] = kwargs.get('optim', 'sgd')
    config['scheduler'] = kwargs.get('scheduler', 'onecycle70')
    config['lr'] = float(kwargs.get('lr', 0.0001))
    config['decay'] = float(kwargs.get('decay', 1e-5))

    config['dim'] = int(kwargs.get('dim', 300))

    config['batch_size'] = kwargs.get('batch_size', 48)
    config['p'] = int(kwargs.get('p', 16))
    config['k'] = int(kwargs.get('k', 3))

    config['num_batches'] = int(kwargs.get('num_batches', 1000))
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

    loaders = {
        tag: dset.make_loader(
            shuffle=(tag == 'train'),
            batch_size=config['batch_size'],
            num_batches=(config['num_batches'] if tag == 'train' else config['num_batches'] // 10),
            k=config['k'],
            p=config['p'],
            num_workers=config['workers'],
        )
        for tag, dset in datasets.items()
    }

    if config['scheduler'] == 'steplr':
        from torch.optim import lr_scheduler
        scheduler_ = (lr_scheduler.StepLR,
                      dict(step_size=8, gamma=0.1, last_epoch=-1))
    else:
        scheduler_ = nh.Scheduler.coerce(config, scheduler='onecycle70')

    hyper = nh.HyperParams(**{
        'nice': config['nice'],
        'workdir': config['workdir'],
        'datasets': datasets,
        'loaders': loaders,
        'xpu': nh.XPU.coerce(config['xpu']),

        'model': (nh.models.DescriptorNetwork, {
            'input_shape': (1, 3, config['dim'], config['dim']),
            'norm_desc': config['norm_desc'],
            # 'hidden_channels': [512, 256]
            'hidden_channels': [256],
            'desc_size': 128,
        }),
        'initializer': nh.Initializer.coerce(config),
        'optimizer': nh.Optimizer.coerce(config),
        'scheduler': scheduler_,
        'criterion': (nh.criterions.TripletLoss, {
            'margin': config['margin'],
            'soft': config['soft'],
        }),
        'monitor': nh.Monitor.coerce(
            config,
            minimize=['loss', 'pos_dist', 'brier'],
            maximize=['accuracy', 'neg_dist', 'mcc'],
            patience=100,
            max_epoch=100,
        ),
        'dynamics': nh.Dynamics.coerce(config),

        'other': {
            'n_classes': 2,
        },
    })
    harn = MatchingHarness(hyper=hyper)
    harn.preferences['prog_backend'] = 'progiter'
    harn.intervals['log_iter_train'] = 1
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None

    return harn


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
            import kwplot
            kwplot.autompl()
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

    print('ABOUT TO INIT')
    harn.initialize()
    print('ABOUT TO RUN')
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/examples/ggr_matching.py --help

        srun -p priority --gres=gpu:2 python ~/code/netharn/examples/ggr_matching.py --workers=0 --xpu=0 --dbname=ggr2-revised --batch_size=46 --nice=ggr2-revised-test-v9003 --lr=0.01 --decay=1e-4 --margin=1


        python ~/code/netharn/examples/ggr_matching.py --workers=8 --xpu=0 --dbname=ggr2-revised \
                --batch_size=44 --k=4 --p=11 --bstep=4 \
                --lr=0.01 --decay=1e-5 --margin=1 --scheduler=steplr \
                --nice=ggr2-revised-test-v9004

        srun -p priority --gres=gpu:2 -c 13 python ~/code/netharn/examples/ggr_matching.py \
                --workers=12 --gpus=0,1 --dbname=ggr2-revised \
                --batch_size=92 \
                --nice=ggr2-revised-test-v9003 \
                --lr=0.01 --decay=1e-4 --margin=1

        srun -p priority --gres=gpu:4 -c 13 python ~/code/netharn/examples/ggr_matching.py \
                --workers=12 --gpus=0,1,2,3 --dbname=ggr2-revised \
                --batch_size=92 --k=4 --p=40 \
                --nice=ggr2-revised-test-bigbatch \
                --lr=0.01 --decay=1e-4 --margin=1
    """
    main()
