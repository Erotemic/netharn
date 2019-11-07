import itertools as it
import torch.nn.functional as F
from os.path import join
import netharn as nh
import numpy as np
import torch
import torchvision
import ubelt as ub
import torch.utils.data.sampler as torch_sampler
from torch import nn


class MNISTEmbeddingNet(nh.layers.Module):
    """
    References:
        https://github.com/adambielski/siamese-triplet/blob/master/networks.py

    Example:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/netharn/examples')
        >>> from mnist_matching import *
        >>> input_shape = (None, 1, 28, 28)
        >>> self = MNISTEmbeddingNet(input_shape)
        >>> print('self = {!r}'.format(self))
        >>> print('flat_shape = {!r}'.format(self._flat_shape))
        >>> print(ub.repr2(self.output_shape_for(input_shape).hidden.shallow(2), nl=2))
        {
            'convnet': {
                '0': (None, 32, 24, 24),
                '1': (None, 32, 24, 24),
                '2': (None, 32, 12, 12),
                '3': (None, 64, 8, 8),
                '4': (None, 64, 8, 8),
                '5': (None, 64, 4, 4),
            },
            'reshape': (
                None,
                1024,
            ),
            'fc': {
                '0': (None, 256),
                '1': (None, 256),
                '2': (None, 256),
                '3': (None, 256),
                '4': (None, 256),
            },
        }
        >>> nh.OutputShapeFor(self)(self.input_shape)
        ...dvecs...(None, 256)...
        >>> input_shape = [4] + list(self.input_shape[1:])
        >>> nh.OutputShapeFor(self)._check_consistency(input_shape)
        >>> inputs = torch.rand(input_shape)
        >>> dvecs = self(inputs)['dvecs']
        >>> pdists = torch.nn.functional.pdist(dvecs, p=2)
        >>> pos_dist = pdists[0::2]
        >>> neg_dist = pdists[1::2]
        >>> margin = 1
        >>> x = pos_dist - neg_dist + margin
        >>> loss = torch.nn.functional.softplus(x).mean()
        >>> loss.backward()
    """
    def __init__(self, input_shape=None, desc_size=256):
        super(MNISTEmbeddingNet, self).__init__()
        self.input_shape = input_shape
        print('input_shape = {!r}'.format(input_shape))
        self.in_channels = self.input_shape[1]
        self.out_channels = desc_size

        self.convnet = nh.layers.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                            nn.MaxPool2d(2, stride=2),
                                            nn.Conv2d(32, 64, 5), nn.PReLU(),
                                            nn.MaxPool2d(2, stride=2))
        self._conv_output_shape = self.convnet.output_shape_for(self.input_shape)
        self._num_flat = np.prod(self._conv_output_shape[1:])
        self.reshape = nh.layers.Reshape(-1, self._num_flat)
        self._flat_shape = self.reshape.output_shape_for(self._conv_output_shape)
        # print('self._conv_output_shape = {!r}'.format(self._conv_output_shape))
        # print('self._num_flat = {!r}'.format(self._num_flat))
        # print('self._flat_shape = {!r}'.format(self._flat_shape))
        self.fc = nh.layers.Sequential(nn.Linear(self._num_flat, 256), nn.PReLU(),
                                       nn.Linear(256, 256), nn.PReLU(),
                                       nn.Linear(256, desc_size))

    def forward(self, inputs):
        conv_out = self.convnet(inputs)
        flat_conv = self.reshape(conv_out)
        dvecs = self.fc(flat_conv)
        outputs = {'dvecs': dvecs}
        return outputs

    # def forward(self, inputs):
    #     class _ForwardOutput(object):
    #         def coerce(self, x, hidden=None):
    #             return x
    #     _Hidden = ub.odict
    #     _OutputFor = ub.identity
    #     _Output = _ForwardOutput
    #     return self._output_for(inputs, _Hidden, _OutputFor, _Output)

    def _output_for(self, inputs, _Hidden, _OutputFor, _Output):
        hidden = _Hidden()
        hidden['convnet'] = output = _OutputFor(self.convnet)(inputs)
        hidden['reshape'] = output = _OutputFor(self.reshape)(output)  # .view(output.size()[0], -1)
        hidden['fc'] = output = _OutputFor(self.fc)(output)
        return _Output.coerce({'dvecs': output}, hidden)

    def output_shape_for(self, input_shape):
        inputs = input_shape
        _OutputFor = nh.OutputShapeFor
        _Hidden = nh.HiddenShapes
        _Output = nh.OutputShape
        return self._output_for(inputs, _Hidden, _OutputFor, _Output)

    def get_embedding(self, x):
        return self.forward(x)


class MNIST_MatchingHarness(nh.FitHarn):
    """
    Define how to process a batch, compute loss, and evaluate validation
    metrics.

    Example:
        >>> harn = setup_harn().initialize()
        >>> batch = harn._demo_batch(0, 'train')
        >>> batch = harn._demo_batch(0, 'vali')
    """

    def prepare_batch(harn, raw_batch):
        """
        One - prepare the batch

        Args:
            raw_batch (object): raw collated items returned by the loader

        Returns:
            Dict: a standardized variant of the raw batch on the XPU

        Example:
            >>> harn = setup_harn().initialize()
            >>> raw_batch = harn._demo_batch(raw=1)
            >>> batch = harn.prepare_batch(raw_batch)
        """
        image, label = raw_batch
        batch = {
            'chip': image,
            'nx': label,
        }
        batch = harn.xpu.move(batch)
        return batch

    def run_batch(harn, batch):
        """
        Two - run the batch

        Args:
            batch (object):
                XPU-mounted, standarized, collated, and prepared items

        Returns:
            Tuple: outputs, loss


        SeeAlso:
            ~/code/netharn/netharn/criterions/triplet.py


        Example:
            >>> harn = setup_harn().initialize()
            >>> raw_batch = harn._demo_batch(raw=1)
            >>> batch = harn.prepare_batch(raw_batch)
            >>> harn, outputs = harn.run_batch(batch)
        """
        inputs = batch['chip']
        outputs = harn.model(inputs)

        dvecs = outputs['dvecs']

        def pos_cand_idxs(groupxs):
            for g in groupxs:
                for p in it.product(g, g):
                    yield p
        try:
            labels = batch['nx']
            pos_dists, neg_dists, triples = harn.criterion._get_pairs(dvecs, labels)
            # num = 1 if harn.current_tag == 'train' else 4
            # unique_nxs, groupxs = nh.util.group_indices(batch['nx'].data.cpu().numpy())
            # pos_idxs = list(pos_cand_idxs(groupxs))
            # pos_dists, neg_dists, triples = harn.criterion.mine_negatives(
            #     dvecs, pos_idxs, num=num, nxs=batch['nx'])
        except RuntimeError:
            # try:
            #     pos_dists, neg_dists, triples = harn.criterion.mine_negatives(
            #         dvecs, pos_idxs, num=num, nxs=batch['nx'], eps=0)
            # except RuntimeError:
            raise nh.exceptions.SkipBatch

        loss = harn.criterion(pos_dists, neg_dists)
        # loss = harn.criterion(neg_dists, pos_dists)
        outputs['triples'] = triples
        outputs['chip'] = batch['chip']
        outputs['nx'] = batch['nx']
        return outputs, loss

    # --- EVERYTHING AFER THIS POINT IS SIMPLY MONITORING AND VISUALIZING
    # Note that these are still netharn callbacks

    def after_initialize(harn, **kw):
        """ custom netharn callback """
        harn.confusion_vectors = []
        harn._has_preselected = False
        harn.POS_LABEL = 1
        harn.NEG_LABEL = 0

    def before_epochs(harn):
        """ custom netharn callback """
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

    def on_batch(harn, batch, outputs, loss):
        """
        custom netharn callback

        Example:
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='vali')
            >>> outputs, loss = harn.run_batch(batch)
            >>> decoded = harn._decode(outputs)
            >>> stacked = harn._draw_batch(batch, decoded, limit=42)
            >>> # xdoctest: +REQUIRES(--show)
            >>> nh.util.imshow(stacked)
            >>> nh.util.show_if_requested()
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
        """ custom netharn callback """
        from sklearn import metrics
        margin = harn.hyper.criterion_params['margin']
        epoch_metrics = {}

        if harn.confusion_vectors:
            y_true = np.hstack([r for r, p in harn.confusion_vectors])
            y_dist = np.hstack([p for r, p in harn.confusion_vectors])

            pos_dist = np.nanmean(y_dist[y_true == harn.POS_LABEL])
            neg_dist = np.nanmean(y_dist[y_true == harn.NEG_LABEL])

            # Transform distance into a probability-like space
            y_probs = torch.Tensor(-(y_dist - margin)).sigmoid().numpy()

            y_pred = (y_dist <= margin).astype(y_true.dtype)
            accuracy = (y_true == y_pred).mean()
            brier = ((y_probs - y_true) ** 2).mean()
            mcc = metrics.matthews_corrcoef(y_true, y_pred)

            alt_margin = np.median(y_dist)
            alt_y_pred = (y_dist < alt_margin).astype(y_true.dtype)
            triple_mcc = metrics.matthews_corrcoef(y_true, alt_y_pred)
            triple_acc = (y_true == alt_y_pred).mean()

            epoch_metrics = {
                'mcc': mcc,
                'brier': brier,
                'accuracy': accuracy,
                'pos_dist': pos_dist,
                'neg_dist': neg_dist,
                'triple_acc': triple_acc,
                'triple_mcc': triple_mcc,
            }

        # Clear scores for next epoch
        harn.confusion_vectors.clear()
        return epoch_metrics

    # --- Non-netharn helper functions

    def _decode(harn, outputs):
        """
        Convert raw network outputs to something interpretable
        """
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

    def _draw_batch(harn, batch, decoded, limit=12):
        """
        Example:
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='vali')
            >>> outputs, loss = harn.run_batch(batch)
            >>> decoded = harn._decode(outputs)
            >>> stacked = harn._draw_batch(batch, decoded)
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
        dsize = (254, 254)
        import cv2
        for i in range(n):
            ims = [g[i].transpose(1, 2, 0) for g in decoded['imgs']]
            ims = [cv2.resize(g, dsize) for g in ims]
            ims = [nh.util.atleast_3channels(g) for g in ims]
            triple_nxs = [n[i] for n in decoded['nxs']]
            if False:
                triple_dvecs = [d[i] for d in decoded['dvecs']]
                da, dp, dn = triple_dvecs
                distAP = np.sqrt(((da - dp) ** 2).sum())
                distAN = np.sqrt(((da - dn) ** 2).sum())
                print('distAP = {!r}'.format(distAP))
                print('distAN = {!r}'.format(distAN))
                print('----')

            text = 'distAP={:.3g} -- distAN={:.3g} -- {}'.format(
                decoded['distAP'][i],
                decoded['distAN'][i],
                str(triple_nxs),
            )
            if decoded['distAP'][i] < decoded['distAN'][i]:
                color = 'dodgerblue'
            else:
                color = 'orangered'

            img = nh.util.stack_images(
                ims, overlap=-2, axis=1,
                bg_value=(10 / 255, 40 / 255, 30 / 255)
            )
            img = (img * 255).astype(np.uint8)
            img = nh.util.draw_text_on_image(img, text,
                                             org=(2, img.shape[0] - 2),
                                             color=color, **fontkw)
            tostack.append(img)
        stacked = nh.util.stack_images_grid(tostack, overlap=-10,
                                            bg_value=(30, 10, 40),
                                            axis=1, chunksize=3)
        return stacked


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
    """
    def __init__(self, torch_dset, p=21, k=4, batch_size=None, drop_last=False,
                 rng=None, shuffle=False, num_batches=None, replace=True):
        self.torch_dset = torch_dset
        self.drop_last = drop_last
        self.replace = replace

        if replace is False:
            raise NotImplementedError(
                'We currently cant sample without replacement')

        try:
            self.pccs = torch_dset.pccs
        except AttributeError:
            raise AttributeError('The `torch_dset` must have the `pcc` attribute')

        self.shuffle = shuffle
        self.multitons = [pcc for pcc in self.pccs if len(pcc) > 1]

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
        self.rng = nh.util.ensure_rng(rng, api='python')

    def __nice__(self):
        return ('p={p}, k={k}, batch_size={batch_size}, '
                'len={num_batches}').format(**self.__dict__)

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

        indices = sorted(ub.flatten(groups))
        return indices

    def __len__(self):
        return self.num_batches


def setup_datasets(workdir=None):
    if workdir is None:
        workdir = ub.expandpath('~/data/mnist/')

    # Define your dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    learn_dset = nh.data.MNIST(workdir, transform=transform, train=True,
                               download=True)

    test_dset = nh.data.MNIST(workdir, transform=transform, train=False,
                              download=True)

    # split the learning dataset into training and validation
    # take a subset of data
    factor = .15
    n_vali = int(len(learn_dset) * factor)
    learn_idx = np.arange(len(learn_dset))

    rng = np.random.RandomState(0)
    rng.shuffle(learn_idx)

    reduction = int(ub.argval('--reduction', default=1))
    vali_idx  = torch.LongTensor(learn_idx[:n_vali][::reduction])
    train_idx = torch.LongTensor(learn_idx[n_vali:][::reduction])

    train_dset = torch.utils.data.Subset(learn_dset, train_idx)
    vali_dset = torch.utils.data.Subset(learn_dset, vali_idx)

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }
    if not ub.argflag('--test'):
        del datasets['test']
    for tag, dset in datasets.items():
        # Construct the PCCs (positive connected components)
        # These are groups of item indices which are positive matches
        if isinstance(dset, torch.utils.data.Subset):
            labels = dset.dataset.train_labels[dset.indices]
        else:
            labels = dset.labels
        unique_labels, groupxs = nh.util.group_indices(labels.numpy())
        dset.pccs = [xs.tolist() for xs in groupxs]

    # Give the training dataset an input_id
    datasets['train'].input_id = 'mnist_' + ub.hash_data(train_idx.numpy())[0:8]
    return datasets, workdir


def setup_harn(**kwargs):
    """
    Args:
        nice (str): Custom tag for this run
        workdir (PathLike): path to dump all the intermedate results

        batch_size (int):
            Base batch size. Number of examples in GPU at any time.
        p (int): num individuals per batch
        k (int): num annots-per-individual per batch

        bstep (int): Multiply by batch_size to simulate a larger batches.
        lr (float|str): Base learning rate
        decay (float): Weight decay (L2 regularization)

        workers (int): Number of parallel data loader workers
        xpu (str): Device to train on. Can be either `'cpu'`, `'gpu'`, a number
            indicating a GPU (e.g. `0`), or a list of numbers (e.g. `[0,1,2]`)
            indicating multiple GPUs

        norm_desc (bool): if True normalizes the descriptors
        pretrained (PathLike): path to a compatible pretrained model

        margin (float): margin for loss criterion
        soft (bool): use soft margin
    """
    config = {}
    config['init'] = kwargs.get('init', 'kaiming_normal')
    config['pretrained'] = config.get('pretrained',
                                      ub.argval('--pretrained', default=None))
    config['margin'] = kwargs.get('margin', 1.0)
    config['soft'] = kwargs.get('soft', False)
    config['xpu'] = kwargs.get('xpu', 'argv')
    config['nice'] = kwargs.get('nice', 'untitled')
    config['workdir'] = kwargs.get('workdir', None)
    config['workers'] = int(kwargs.get('workers', 1))
    config['bstep'] = int(kwargs.get('bstep', 1))
    config['optim'] = kwargs.get('optim', 'sgd')
    config['scheduler'] = kwargs.get('scheduler', 'onecycle70')
    config['lr'] = kwargs.get('lr', 0.00011)
    config['decay'] = float(kwargs.get('decay', 1e-5))

    config['batch_size'] = int(kwargs.get('batch_size', 120))
    config['p'] = float(kwargs.get('p', 10))
    config['k'] = float(kwargs.get('k', 12))

    config['arch'] = kwargs.get('arch', 'simple')
    config['hidden'] = kwargs.get('hidden', [256])
    config['desc_size'] = kwargs.get('desc_size', 128)

    try:
        import ast
        config['hidden'] = ast.literal_eval(config['hidden'])
    except Exception:
        pass

    try:
        config['lr'] = float(config['lr'])
    except Exception:
        pass

    config['norm_desc'] = kwargs.get('norm_desc', False)

    config['dim'] = 28

    xpu = nh.XPU.coerce(config['xpu'])
    nh.configure_hacks(config)
    datasets, workdir = setup_datasets()

    loaders = {
        tag: torch.utils.data.DataLoader(
            dset,
            batch_sampler=MatchingSamplerPK(
                dset,
                shuffle=(tag == 'train'),
                batch_size=config['batch_size'],
                k=config['k'],
                p=config['p'],
            ),
            num_workers=config['workers'],
        )
        for tag, dset in datasets.items()
    }

    if config['arch'] == 'simple':
        model_ = (MNISTEmbeddingNet, {
            'input_shape': (1, 1, config['dim'], config['dim']),
            'desc_size': config['desc_size'],
        })
    elif config['arch'] == 'resnet50':
        model_ = (nh.models.DescriptorNetwork, {
            'input_shape': (1, 1, config['dim'], config['dim']),
            'norm_desc': config['norm_desc'],
            'hidden_channels': config['hidden'],
            'desc_size': config['desc_size'],
        })
    else:
        raise KeyError(config['arch'])

    # Here is the FitHarn magic.
    # They nh.HyperParams object keeps track of and helps log all declarative
    # info related to training a model.
    hyper = nh.hyperparams.HyperParams(
        nice='mnist',
        xpu=xpu,
        workdir=workdir,
        datasets=datasets,
        loaders=loaders,
        model=model_,
        initializer=nh.Initializer.coerce(config),
        optimizer=nh.Optimizer.coerce(config),
        scheduler=nh.Scheduler.coerce(config, scheduler='onecycle70'),
        criterion=(nh.criterions.TripletLoss, {
            'margin': config['margin'],
            'soft': config['soft'],
        }),
        monitor=nh.Monitor.coerce(
            config,
            minimize=['loss', 'pos_dist', 'brier'],
            maximize=['accuracy', 'neg_dist', 'mcc'],
            patience=300,
            max_epoch=300,
            smoothing=0.4,
        ),
    )

    harn = MNIST_MatchingHarness(hyper=hyper)
    harn.config
    return harn


def main():
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
        # Show help
        python ~/code/netharn/examples/mnist_matching.py --help

        # Run with default values
        python ~/code/netharn/examples/mnist_matching.py

        # Try new hyperparam values and run the LR-range-test
        python ~/code/netharn/examples/mnist_matching.py \
                --batch_size=128 \
                --scheduler=step50 \
                --optim=sgd \
                --soft=True \
                --margin=2 \
                --lr=0.0007

        python ~/code/netharn/examples/mnist_matching.py \
                --batch_size=128 \
                --scheduler=step50 \
                --optim=sgd \
                --soft=True \
                --margin=2 \
                --lr=0.0007
                --arch=resnet

        python ~/code/netharn/examples/mnist_matching.py \
                --batch_size=256 \
                --scheduler=step50 \
                --optim=sgd \
                --soft=False \
                --margin=1 \
                --lr=0.001 \
                --arch=simple

        # Reproduce state-of-the-art experiment
        python ~/code/netharn/examples/mnist_matching.py \
                --scheduler=ReduceLROnPlateau \
                --batch_size=128 \
                --optim=adamw \
                --lr=0.000177

        python ~/code/netharn/examples/mnist_matching.py \
                --scheduler=ReduceLROnPlateau \
                --batch_size=2048 \
                --optim=adamw \
                --pretrained=/home/joncrall/data/mnist/fit/runs/mnist/ljqpwgzr/torch_snapshots/_epoch_00000020.pt \
                --lr=interact

                --lr=0.0001778

        python ~/code/netharn/examples/mnist_matching.py         --batch_size=256         --scheduler=step10         --optim=sgd         --lr=0.003 --workers=1 --pretrained=/home/joncrall/data/mnist/fit/runs/mnist/sjmldjdl/torch_snapshots/_epoch_00000001.pt

        python ~/code/netharn/examples/mnist_matching.py         --batch_size=2048         --scheduler=step10         --optim=sgd         --lr=0.003 --workers=1 --pretrained=/home/joncrall/data/mnist/fit/runs/mnist/jefmqvid/torch_snapshots/_epoch_00000005.pt

        python ~/code/netharn/examples/mnist_matching.py --batch_size=2048 --scheduler=step10 --optim=adamw --lr=0.001 --workers=1 --pretrained=/home/joncrall/data/mnist/fit/runs/mnist/jefmqvid/torch_snapshots/_epoch_00000005.pt --decay=0
    """
    main()
