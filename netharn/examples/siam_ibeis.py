"""
This module can be used as both a script and an importable module.
Run `python siam_ibeis.py --help` for more details.
See docstring in fit for more details on the importable module.


conda install opencv
conda install pytorch torchvision -c pytorch

TestMe:
    xdoctest ~/code/netharn/netharn/examples/siam_ibeis.py all
"""
import os
import ubelt as ub
import numpy as np
import netharn as nh
import torch
import torchvision
import itertools as it

__all__ = [
    'RandomBalancedIBEISSample',
    'SiameseLP',
    'SiamHarness',
    'randomized_ibeis_dset',
    'setup_harness',
    'fit',
]


class SiameseLP(torch.nn.Module):
    """
    Siamese pairwise distance

    Example:
        >>> self = SiameseLP()
    """

    def __init__(self, p=2, branch=None, input_shape=(1, 3, 416, 416)):
        super(SiameseLP, self).__init__()
        if branch is None:
            self.branch = torchvision.models.resnet50(pretrained=True)
        else:
            self.branch = branch
        assert isinstance(self.branch, torchvision.models.ResNet)
        prepool_shape = self.resnet_prepool_output_shape(input_shape)
        # replace the last layer of resnet with a linear embedding to learn the
        # LP distance between pairs of images.
        # Also need to replace the pooling layer in case the input has a
        # different size.
        self.prepool_shape = prepool_shape
        pool_channels = prepool_shape[1]
        pool_kernel = prepool_shape[2:4]
        self.branch.avgpool = torch.nn.AvgPool2d(pool_kernel, stride=1)
        self.branch.fc = torch.nn.Linear(pool_channels, 500)

        self.pdist = torch.nn.PairwiseDistance(p=p)

    def resnet_prepool_output_shape(self, input_shape):
        """
        self = SiameseLP(input_shape=input_shape)
        input_shape = (1, 3, 224, 224)
        self.resnet_prepool_output_shape(input_shape)
        self = SiameseLP(input_shape=input_shape)
        input_shape = (1, 3, 416, 416)
        self.resnet_prepool_output_shape(input_shape)
        """
        # Figure out how big the output will be and redo the average pool layer
        # to account for it
        branch = self.branch
        shape = input_shape
        shape = nh.OutputShapeFor(branch.conv1)(shape)
        shape = nh.OutputShapeFor(branch.bn1)(shape)
        shape = nh.OutputShapeFor(branch.relu)(shape)
        shape = nh.OutputShapeFor(branch.maxpool)(shape)

        shape = nh.OutputShapeFor(branch.layer1)(shape)
        shape = nh.OutputShapeFor(branch.layer2)(shape)
        shape = nh.OutputShapeFor(branch.layer3)(shape)
        shape = nh.OutputShapeFor(branch.layer4)(shape)
        prepool_shape = shape
        return prepool_shape

    def forward(self, input1, input2):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.

        Example:
            >>> input1 = nh.XPU(None).variable(torch.rand(4, 3, 224, 224))
            >>> input2 = nh.XPU(None).variable(torch.rand(4, 3, 224, 224))
            >>> self = SiameseLP(input_shape=input2.shape[1:])
            >>> output = self(input1, input2)

        Ignore:
            >>> input1 = nh.XPU(None).variable(torch.rand(1, 3, 416, 416))
            >>> input2 = nh.XPU(None).variable(torch.rand(1, 3, 416, 416))
            >>> input_shape1 = input1.shape
            >>> self = SiameseLP(input_shape=input2.shape[1:])
            >>> self(input1, input2)
        """
        output1 = self.branch(input1)
        output2 = self.branch(input2)
        output = self.pdist(output1, output2)
        return output

    def output_shape_for(self, input_shape1, input_shape2):
        shape1 = nh.OutputShapeFor(self.branch)(input_shape1)
        shape2 = nh.OutputShapeFor(self.branch)(input_shape2)
        assert shape1 == shape2
        output_shape = (shape1[0], 1)
        return output_shape


class RandomBalancedIBEISSample(torch.utils.data.Dataset):
    """
    Construct a pairwise image training dataset.

    CommandLine:
        xdoctest ~/code/netharn/netharn/examples/siam_ibeis.py RandomBalancedIBEISSample --show

    Example:
        >>> self = RandomBalancedIBEISSample.from_dbname('PZ_MTEST')
        >>> # xdoctest +REQUIRES(--show)
        >>> self.show_sample()
        >>> nh.util.show_if_requested()
    """
    SEED = 563401

    def __init__(self, pblm, pccs, dim=224, augment=True):
        chip_config = {
            # preserve aspect ratio, use letterbox to fit into network
            'resize_dim': 'maxwh',
            'dim_size': dim,

            # 'resize_dim': 'wh',
            # 'dim_size': (dim, dim)
        }
        self.pccs = pccs
        all_aids = list(ub.flatten(pccs))
        all_fpaths = pblm.infr.ibs.depc_annot.get(
            'chips', all_aids, read_extern=False, colnames='img',
            config=chip_config)

        self.aid_to_fpath = dict(zip(all_aids, all_fpaths))

        # self.multitons_pccs = [pcc for pcc in pccs if len(pcc) > 1]
        self.pos_pairs = []

        # SAMPLE ALL POSSIBLE POS COMBINATIONS AND IGNORE INCOMPARABLE
        self.infr = pblm.infr
        # TODO: each sample should be weighted depending on n_aids in its pcc
        for pcc in pccs:
            if len(pcc) >= 2:
                edges = np.array(list(it.starmap(self.infr.e_, it.combinations(pcc, 2))))
                is_comparable = self.is_comparable(edges)
                pos_edges = edges[is_comparable]
                self.pos_pairs.extend(list(pos_edges))
        rng = nh.util.ensure_rng(self.SEED, 'numpy')
        self.pyrng = nh.util.ensure_rng(self.SEED + 1, 'python')
        self.rng = rng

        # Be good data citizens, construct a dataset identifier
        depends = [
            sorted(map(sorted, self.pccs)),
        ]
        hashid = ub.hash_data(depends)[:12]
        self.input_id = '{}-{}'.format(len(self), hashid)

        if augment:
            import imgaug.augmenters as iaa
            # NOTE: we are only using `self.augmenter` to make a hyper hashid
            # in __getitem__ we invoke transform explicitly for fine control
            self.hue = nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)
            self.crop = iaa.Crop(percent=(0, .2))
            self.flip = iaa.Fliplr(p=.5)
            self.augmenter = iaa.Sequential([self.hue, self.crop, self.flip])
        else:
            self.augmenter = None
        self.letterbox = nh.data.transforms.Resize(target_size=(dim, dim),
                                                   mode='letterbox')
        # self.colorspace = 'RGB'
        # self.center_inputs = None

    @classmethod
    def from_dbname(RandomBalancedIBEISSample, dbname='PZ_MTEST', dim=224):
        """
        dbname = 'PZ_MTEST'
        dim = 244
        """
        from ibeis.algo.verif import vsone
        pblm = vsone.OneVsOneProblem.from_empty(dbname)
        pccs = list(pblm.infr.positive_components())
        self = RandomBalancedIBEISSample(pblm, pccs, dim=dim)
        return self

    def __len__(self):
        return len(self.pos_pairs) * 2

    def show_sample(self):
        """
        CommandLine:
            python ~/code/netharn/netharn/examples/siam_ibeis.py RandomBalancedIBEISSample.show_sample --show

        Example:
            >>> self = RandomBalancedIBEISSample.from_dbname('PZ_MTEST')
            >>> ut.qtensure()
            >>> self.show_sample()
            >>> nh.util.show_if_requested()
        """
        vis_dataloader = torch.utils.data.DataLoader(self, shuffle=True,
                                                     batch_size=8)
        example_batch = next(iter(vis_dataloader))
        concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
        tensor = torchvision.utils.make_grid(concatenated)
        im = tensor.numpy().transpose(1, 2, 0)
        nh.util.imshow(im)
        # import matplotlib.pyplot as plt
        # plt.imshow(im)

    def class_weights(self):
        class_weights = torch.FloatTensor([1.0, 1.0])
        return class_weights

    def is_comparable(self, edges):
        from ibeis.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
        infr = self.infr
        def _check(u, v):
            if infr.incomp_graph.has_edge(u, v):
                return False
            elif infr.pos_graph.has_edge(u, v):
                # Only override if the evidence in the graph says its positive
                # otherwise guess
                ed = infr.get_edge_data((u, v)).get('evidence_decision', UNREV)
                if ed == POSTV:
                    return True
                else:
                    return np.nan
            elif infr.neg_graph.has_edge(u, v):
                return True
            return np.nan
        flags = np.array([_check(*edge) for edge in edges])
        # hack guess if comparable based on viewpoint
        guess_flags = np.isnan(flags)
        need_edges = edges[guess_flags]
        need_flags = infr.ibeis_guess_if_comparable(need_edges)
        flags[guess_flags] = need_flags
        return np.array(flags, dtype=np.bool)

    def get_aidpair(self, index):
        if index % 2 == 0:
            # Get a positive pair if the index is even
            aid1, aid2 = self.pos_pairs[index // 2]
            label = 1
        else:
            # Get a random negative pair if the index is odd
            pcc1, pcc2 = self.pyrng.sample(self.pccs, k=2)
            while pcc1 is pcc2:
                pcc1, pcc2 = self.pyrng.sample(self.pccs, k=2)
            aid1 = self.pyrng.sample(pcc1, k=1)[0]
            aid2 = self.pyrng.sample(pcc2, k=1)[0]
            label = 0
        return aid1, aid2, label

    def load_from_edge(self, aid1, aid2):
        """
        Example:
            >>> self = RandomBalancedIBEISSample.from_dbname('PZ_MTEST')
            >>> img1, img2 = self.load_from_edge(1, 2)
            >>> # xdoctest +REQUIRES(--show)
            >>> self.show_sample()
            >>> nh.util.qtensure()  # xdoc: +SKIP
            >>> nh.util.imshow(img1, pnum=(1, 2, 1), fnum=1)
            >>> nh.util.imshow(img2, pnum=(1, 2, 2), fnum=1)
            >>> nh.util.show_if_requested()
        """
        fpath1 = self.aid_to_fpath[aid1]
        fpath2 = self.aid_to_fpath[aid2]

        img1 = nh.util.imread(fpath1)
        img2 = nh.util.imread(fpath2)
        assert img1 is not None and img2 is not None

        if self.augmenter is not None:
            # Augment hue and crop independently
            img1 = self.hue.forward(img1, self.rng)
            img2 = self.hue.forward(img2, self.rng)
            img1 = self.crop.augment_image(img1)
            img2 = self.crop.augment_image(img2)

            # Do the same flip for both images
            flip_det = self.flip.to_deterministic()
            img1 = flip_det.augment_image(img1)
            img2 = flip_det.augment_image(img2)

        # Always embed images in a letterbox to preserve aspect ratio
        img1 = self.letterbox.forward(img1)
        img2 = self.letterbox.forward(img2)

        return img1, img2

    def __getitem__(self, index):
        """
        Example:
            >>> self = RandomBalancedIBEISSample.from_dbname('PZ_MTEST')
            >>> index = 0
            >>> img1, img2, label = self[index]
        """
        aid1, aid2, label = self.get_aidpair(index)
        img1, img2 = self.load_from_edge(aid1, aid2)
        if self.augmenter is not None:
            if self.rng.rand() > .5:
                img1, img2 = img2, img1
        img1 = torch.FloatTensor(img1.transpose(2, 0, 1))
        img2 = torch.FloatTensor(img2.transpose(2, 0, 1))
        return img1, img2, label


def randomized_ibeis_dset(dbname, dim=416):
    """
    CommandLine:
        xdoctest ~/code/netharn/netharn/examples/siam_ibeis.py randomized_ibeis_dset --show

    Example:
        >>> dbname = 'PZ_MTEST'
        >>> datasets = randomized_ibeis_dset(dbname)
        >>> # xdoctest: +REQUIRES(--show)
        >>> nh.util.qtensure()
        >>> self = datasets['train']
        >>> self.show_sample()
        >>> nh.util.show_if_requested()
    """
    from ibeis.algo.verif import vsone
    pblm = vsone.OneVsOneProblem.from_empty(dbname)

    # Simpler very randomized sample strategy
    pcc_sets = {
        'train': set(),
        'vali': set(),
        'test': set(),
    }

    vali_frac = .1
    test_frac = .1
    train_frac = 1 - (vali_frac + test_frac)

    category_probs = ub.odict([
        ('train', train_frac),
        ('test', test_frac),
        ('vali', vali_frac),
    ])

    # Gather all PCCs
    pccs = list(map(frozenset, pblm.infr.positive_components()))

    # Group PCCs by the number of annotations they contain
    pcc_freq = list(map(len, pccs))
    freq_grouped = ub.group_items(pccs, pcc_freq)

    rng = nh.util.ensure_rng(989540621)

    # Perform splits over these groups so test / train / vali roughly see the
    # same proportion of differently sized PCCs
    for i, group in freq_grouped.items():
        # Each PCC in this group has a probability of going into the
        # either test / train / or vali split
        choices = rng.choice(list(category_probs.keys()),
                             p=list(category_probs.values()), size=len(group))
        for key, group in zip(choices, group):
            pcc_sets[key].add(group)

    if __debug__:
        # Ensure sets of PCCs are disjoint!
        intersections = {}
        for key1, key2 in it.combinations(pcc_sets.keys(), 2):
            isect = pcc_sets[key1].intersection(pcc_sets[key2])
            intersections[(key1, key2)] = isect
        num_isects = ub.map_vals(len, intersections)
        if any(num_isects.values()):
            msg = 'Splits are not disjoint: {}'.format(ub.repr2(
                num_isects, sk=1))
            print(msg)
            raise AssertionError(msg)

    if True:
        num_pccs = ub.map_vals(len, pcc_sets)
        total = sum(num_pccs.values())
        fracs = {k: v / total for k, v in num_pccs.items()}
        print('Splits use the following fractions of data: {}'.format(
            ub.repr2(fracs, precision=4)))

        for key, want in category_probs.items():
            got = fracs[key]
            absdiff = abs(want - got)
            if absdiff > 0.1:
                raise AssertionError(
                    'Sampled fraction of {} for {!r} is significantly '
                    'different than what was requested: {}'.format(
                        got, key, want))

    test_dataset = RandomBalancedIBEISSample(pblm, pcc_sets['test'], dim=dim)
    train_dataset = RandomBalancedIBEISSample(pblm, pcc_sets['train'], dim=dim,
                                              augment=False)
    vali_dataset = RandomBalancedIBEISSample(pblm, pcc_sets['vali'], dim=dim,
                                             augment=False)

    datasets = {
        'train': train_dataset,
        'vali': vali_dataset,
        'test': test_dataset,
    }
    datasets.pop('test', None)  # dont test for now (speed consideration)
    return datasets


class SiamHarness(nh.FitHarn):
    """
    Define how to process a batch, compute loss, and evaluate validation
    metrics.
    """

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        img1, img2, label = raw_batch
        inputs = harn.xpu.variables(img1, img2)
        label = harn.xpu.variable(label)
        batch = (inputs, label)
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader
        """
        inputs, label = batch
        output = harn.model(*inputs)
        loss = harn.criterion(output, label).sum()
        return output, loss

    def on_batch(harn, batch, output, loss):
        """ custom callback """
        label = batch[-1]
        l2_dist_tensor = torch.squeeze(output.data.cpu())
        label_tensor = torch.squeeze(label.data.cpu())

        # Distance
        POS_LABEL = 1  # NOQA
        NEG_LABEL = 0  # NOQA
        is_pos = (label_tensor == POS_LABEL)

        pos_dists = l2_dist_tensor[is_pos]
        neg_dists = l2_dist_tensor[~is_pos]

        # Average positive / negative distances
        pos_dist = pos_dists.sum() / max(1, len(pos_dists))
        neg_dist = neg_dists.sum() / max(1, len(neg_dists))

        # accuracy
        margin = harn.hyper.criterion_params['margin']
        pred_pos_flags = (l2_dist_tensor <= margin).long()

        pred = pred_pos_flags

        n_correct = (pred == label_tensor).sum()
        fraction_correct = n_correct / len(label_tensor)

        metrics = {
            'accuracy': fraction_correct,
            'pos_dist': pos_dist,
            'neg_dist': neg_dist,
        }
        return metrics

    def on_epoch(harn):
        """ custom callback """
        pass


def setup_harness(**kwargs):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/siam_ibeis.py setup_harness

    Example:
        >>> harn = setup_harness(dbname='PZ_MTEST')
        >>> harn.initialize()
    """
    nice = kwargs.get('nice', 'untitled')
    bsize = int(kwargs.get('bsize', 6))
    bstep = int(kwargs.get('bstep', 4))
    workers = int(kwargs.get('workers', 0))
    decay = float(kwargs.get('decay', 0.0005))
    lr = float(kwargs.get('lr', 0.001))
    dim = int(kwargs.get('dim', 416))
    xpu = kwargs.get('xpu', 'cpu')
    workdir = kwargs.get('workdir', None)
    dbname = kwargs.get('dbname', 'PZ_MTEST')

    datasets = randomized_ibeis_dset(dbname, dim=dim)
    if workdir is None:
        workdir = ub.truepath(os.path.join('~/work/siam-ibeis2', dbname))
    ub.ensuredir(workdir)

    for k, v in datasets.items():
        print('* len({}) = {}'.format(k, len(v)))

    if workers > 0:
        import cv2
        cv2.setNumThreads(0)

    loaders = {
        key:  torch.utils.data.DataLoader(
            dset, batch_size=bsize, num_workers=workers,
            shuffle=(key == 'train'), pin_memory=True)
        for key, dset in datasets.items()
    }

    xpu = nh.XPU.cast(xpu)

    hyper = nh.HyperParams(**{
        'nice': nice,
        'workdir': workdir,
        'datasets': datasets,
        'loaders': loaders,

        'xpu': xpu,

        'model': (SiameseLP, {
            'p': 2,
            'input_shape': (1, 3, dim, dim),
        }),

        'criterion': (nh.criterions.ContrastiveLoss, {
            'margin': 4,
            'weight': None,
        }),

        'optimizer': (torch.optim.SGD, {
            'lr': lr * 0.1,
            'weight_decay': decay,
            'momentum': 0.9,
            'nesterov': True,
        }),

        'initializer': (nh.initializers.NoOp, {}),

        'scheduler': (nh.schedulers.ListedLR, {
            'points': {
                0:  lr * 0.1,
                1:  lr * 1.0,
                59: lr * 1.1,
                60: lr * 0.1,
                90: lr * 0.01,
            },
            'interpolate': True
        }),

        'monitor': (nh.Monitor, {
            'minimize': ['loss', 'pos_dist'],
            'maximize': ['accuracy', 'neg_dist'],
            'patience': 160,
            'max_epoch': 160,
        }),

        'augment': datasets['train'].augmenter,

        'dynamics': {
            # Controls how many batches to process before taking a step in the
            # gradient direction. Effectively simulates a batch_size that is
            # `bstep` times bigger.
            'batch_step': bstep,
        },

        'other': {
            'n_classes': 2,
        },
    })
    harn = SiamHarness(hyper=hyper)
    harn.config['prog_backend'] = 'progiter'
    harn.intervals['log_iter_train'] = 1
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None

    return harn


def _auto_argparse(func):
    """
    Transform a function with a Google Style Docstring into an
    `argparse.ArgumentParser`.  Custom utility. Not sure where it goes yet.
    """
    from xdoctest import docscrape_google as scrape
    import argparse
    import inspect

    # Parse default values from the function dynamically
    spec = inspect.getargspec(func)
    kwdefaults = dict(zip(spec.args[-len(spec.defaults):], spec.defaults))

    # Parse help and description information from a google-style docstring
    docstr = func.__doc__
    description = scrape.split_google_docblocks(docstr)[0][1][0].strip()
    google_args = {argdict['name']: argdict
                   for argdict in scrape.parse_google_args(docstr)}

    # Create the argument parser and register each argument
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for arg in spec.args:
        argkw = {}
        if arg in kwdefaults:
            argkw['default'] = kwdefaults[arg]
        if arg in google_args:
            garg = google_args[arg]
            argkw['help'] = garg['desc']
            try:
                argkw['type'] = eval(garg['type'], {})
            except Exception:
                pass
        parser.add_argument('--' + arg, **argkw)
    return parser


def fit(dbname='PZ_MTEST', nice='untitled', dim=416, bsize=6, bstep=4,
        lr=0.001, decay=0.0005, workers=0, xpu='cpu'):
    """
    Train a siamese chip descriptor for animal identification.

    Args:
        dbname (str): Name of IBEIS database to use
        nice (str): Custom tag for this run
        dim (int): Width and height of the network input
        bsize (int): Base batch size. Number of examples in GPU at any time.
        bstep (int): Multiply by bsize to simulate a larger batches.
        lr (float): Base learning rate
        decay (float): Weight decay (L2 regularization)
        workers (int): Number of parallel data loader workers
        xpu (str): Device to train on. Can be either `'cpu'`, `'gpu'`, a number
            indicating a GPU (e.g. `0`), or a list of numbers (e.g. `[0,1,2]`)
            indicating multiple GPUs
    """
    # There has to be a good way to use argparse and specify params only once.
    # Pass all args down to setup_harness
    import inspect
    kw = ub.dict_subset(locals(), inspect.getargspec(fit).args)
    harn = setup_harness(**kw)
    harn.run()


def main():
    """
    CommandLine:
        python examples/siam_ibeis.py --help

        # Test Runs:
            # use a very small input dimension to test things out
            python examples/siam_ibeis.py --dbname PZ_MTEST --workers=0 --dim=32 --xpu=cpu

            # test that GPU works
            python examples/siam_ibeis.py --dbname PZ_MTEST --workers=0 --dim=32 --xpu=gpu0

            # test that running at a large size works
            python examples/siam_ibeis.py --dbname PZ_MTEST --workers=2 --dim=416 --xpu=gpu0

        # Real Run:
        python examples/siam_ibeis.py --dbname GZ_Master1 --workers=2 --dim=512 --xpu=gpu0

    Notes:
        # Some database names
        PZ_Master1
        GZ_Master1
        RotanTurtles
        humpbacks_fb
    """
    parser = _auto_argparse(fit)
    args, unknown = parser.parse_known_args()
    ns = args.__dict__.copy()
    fit(**ns)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/siam_ibeis.py --help
    """
    main()
