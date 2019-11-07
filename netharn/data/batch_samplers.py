import netharn as nh
import ubelt as ub
import torch.utils


class MatchingSamplerPK(ub.NiceRepr, torch.utils.data.sampler.BatchSampler):
    """
    Samples random triples from a PCC-complient dataset

    Args:
        pccs (List[FrozenSet]):
            Groups of annotation-indices, where each group contains all annots
            with the same name (individual identity).
        p (int): number of individuals sampled per batch
        k (int): number of annots sampled per individual within a batch
        batch_size (int): if specified, k is adjusted to an appropriate length
        drop_last (bool): ignored
        num_batches (int): length of the loader
        rng (int | Random, default=None): random seed
        shuffle (bool): if False rng is ignored and getitem is deterministic

    TODO:
        Look at
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
        to see if we can try using all examples of a class before repeating
        them

    Example:
        >>> pccs = [(0, 1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11), (12,)]
        >>> batch_sampler = self = MatchingSamplerPK(pccs, p=2, k=2, shuffle=True)
        >>> print('batch_sampler = {!r}'.format(batch_sampler))
        >>> for indices in batch_sampler:
        >>>     print('indices = {!r}'.format(indices))
    """
    def __init__(self, pccs, p=21, k=4, batch_size=None, drop_last=False,
                 rng=None, shuffle=False, num_batches=None, replace=True):
        self.drop_last = drop_last
        self.replace = replace
        self.shuffle = shuffle
        self.pccs = pccs

        assert k > 1

        if replace is False:
            raise NotImplementedError(
                'We currently cant sample without replacement')

        if getattr(pccs, '__hasgraphid__', False):
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
            default_num_batches = 0
            for pcc in ub.ProgIter(self.pccs, 'pccs',  enabled=0):
                num_pos_edges = scipy.special.comb(len(pcc), 2)
                if num_pos_edges > 0:
                    default_num_batches += len(pcc)
                other_pccs = [c for c in self.pccs if c is not pcc]
                num_neg_edges = sum(len(c) for c in other_pccs)
                self.num_triples += num_pos_edges * num_neg_edges
                self.num_pos_edges += num_pos_edges

        self.multitons = [pcc for pcc in self.pccs if len(pcc) > 1]

        p = min(len(self.multitons), p)
        k = min(max(len(p) for p in self.pccs), k)
        assert k > 1

        if batch_size is not None:
            p = batch_size // k

        batch_size = p * k

        if not num_batches:
            num_batches = default_num_batches

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
