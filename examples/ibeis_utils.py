import ubelt as ub
import numpy as np
import netharn as nh
import torch
import torchvision
import itertools as it


def randomized_ibeis_dset(dbname, dim=416):
    """
    CommandLine:
        xdoctest ~/code/netharn/netharn/examples/ggr_matching.py randomized_ibeis_dset --show

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

    vali_frac = .0
    test_frac = .1
    train_frac = 1 - (vali_frac + test_frac)

    category_probs = ub.odict([
        ('train', train_frac),
        ('test', test_frac),
        ('vali', vali_frac),
    ])

    rng = nh.util.ensure_rng(989540621)

    # Gather all PCCs
    pccs = sorted(map(frozenset, pblm.infr.positive_components()))

    # Each PCC in this group has a probability of going into the
    # either test / train / or vali split
    choices = rng.choice(list(category_probs.keys()),
                         p=list(category_probs.values()), size=len(pccs))
    for key, pcc in zip(choices, pccs):
        pcc_sets[key].add(pcc)

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
    # datasets.pop('test', None)  # dont test for now (speed consideration)
    return datasets


class RandomBalancedIBEISSample(torch.utils.data.Dataset):
    """
    Construct a pairwise image training dataset.

    CommandLine:
        xdoctest ~/code/netharn/netharn/examples/ggr_matching.py RandomBalancedIBEISSample --show

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
                # ut.random_combinations
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
            python ~/code/netharn/netharn/examples/ggr_matching.py RandomBalancedIBEISSample.show_sample --show

        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/netharn/examples')
            >>> from ggr_matching import *
            >>> self = RandomBalancedIBEISSample.from_dbname('PZ_MTEST')
            >>> nh.util.autompl()
            >>> self.show_sample()
            >>> nh.util.show_if_requested()
        """
        vis_dataloader = torch.utils.data.DataLoader(self, shuffle=True,
                                                     batch_size=8)
        example_batch = ub.peek(vis_dataloader)

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
        return {
            'img1': img1,
            'img2': img2,
            'label': label,
        }
