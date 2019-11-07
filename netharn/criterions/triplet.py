
import ubelt as ub
import numpy as np
import torch
import torch.nn.functional as F
import itertools as it


def all_pairwise_distances(x, y=None, hack=1):
    """
    Fast pairwise L2 squared distances between two sets of d-dimensional vectors

    Args:
        x (Tensor): an Nxd matrix
        y (Tensor, default=None): an optional Mxd matirx

    Returns:
        Tensor: dist: an NxM matrix where dist[i,j] is the square norm between
        x[i,:] and y[j,:] if y is not given then use 'y=x'.

        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    References:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

    Example:
        >>> from netharn.criterions.triplet import *
        >>> N, d = 5, 3
        >>> x = torch.rand(N, d)
        >>> dist = all_pairwise_distances(x, hack=0)
        >>> assert dist.shape == (N, N)

        >>> a = x[None, :].expand(x.shape[0], -1, -1)
        >>> b = x[:, None].expand(-1, x.shape[0], -1)
        >>> real_dist = ((a - b) ** 2).sum(dim=2).sqrt_()
        >>> delta = (real_dist - dist.sqrt()).numpy()
        >>> assert delta.max() < 1e-5

    Example:
        >>> N, M, d = 5, 7, 3
        >>> x = torch.rand(N, d)
        >>> y = torch.rand(M, d)
        >>> dist = all_pairwise_distances(x, y, hack=0)
        >>> assert dist.shape == (N, M)

    Ignore:
        N, d = 2000, 128
        x = torch.rand(N, d)
        import ubelt as ub
        for timer in ub.Timerit(100, bestof=10, label='time'):
            with timer:
                all_pairwise_distances(x)
                torch.cuda.synchronize()

        for timer in ub.Timerit(100, bestof=10, label='time'):
            with timer:
                dist = torch.nn.functional.pdist(x)
                torch.cuda.synchronize()


    """
    if hack:
        # hack
        assert y is None
        dist = torch.zeros(x.shape[0], x.shape[0])
        triu_idx = np.triu_indices(dist.shape[0], k=1)
        dist[triu_idx] = torch.nn.functional.pdist(x)
        dist += dist.t()
        return dist
    else:
        x_norm = x.pow(2).sum(1).view(-1, 1)
        if y is None:
            y = x
            y_norm = x_norm.view(1, -1)
        else:
            y_norm = y.pow(2).sum(1).view(1, -1)

        yT = torch.transpose(y, 0, 1)
        if True:
            # Benchmarks as faster
            xy_norm = x_norm + y_norm
            xy = torch.mm(x, yT).mul_(-2.0)
            dist = xy_norm.add_(xy)
        else:
            # x_norm = (x ** 2).sum(1).view(-1, 1)
            # y_norm = (y ** 2).sum(1).view(1, -1)
            dist = x_norm + y_norm - 2.0 * torch.mm(x, yT)
        return dist.clamp_(0, None)


class TripletLoss(torch.nn.TripletMarginLoss):
    """
    Triplet loss with either hard or soft margin

    Example:
        >>> dvecs = torch.randn(21, 128)
        >>> pos_idxs = ([(1, 2), (1, 3), (2, 3), (3, 4)])
        >>> num = 1
        >>> pos_dists, neg_dists, triples = TripletLoss.mine_negatives(dvecs, pos_idxs, num)
        >>> self = TripletLoss(soft=1)
        >>> loss = self(pos_dists, neg_dists)
        >>> #
        >>> loss_s = TripletLoss(soft=1, reduction='none')(pos_dists, neg_dists)
        >>> loss_h = TripletLoss(margin=1, reduction='none')(pos_dists, neg_dists)

    Ignore:
        >>> import netharn as nh
        >>> xdata = torch.linspace(-10, 10)
        >>> ydata = {
        >>>     'soft_margin[0]': F.softplus(0 + xdata).numpy(),
        >>>     'soft_margin[1]': F.softplus(1 + xdata).numpy(),
        >>>     'soft_margin[4]': F.softplus(4 + xdata).numpy(),
        >>>     'hard_margin[0]': (0 + xdata).clamp_(0, None).numpy(),
        >>>     'hard_margin[1]': (1 + xdata).clamp_(0, None).numpy(),
        >>>     'hard_margin[4]': (4 + xdata).clamp_(0, None).numpy(),
        >>> }
        >>> nh.util.autompl()
        >>> nh.util.multi_plot(xdata.numpy(), ydata, fnum=1)
    """

    def __init__(self, margin=1.0, eps=1e-6, reduction='mean', soft=True):
        super(TripletLoss, self).__init__(margin=margin, eps=eps,
                                          reduction=reduction)
        self.soft = soft

    @classmethod
    def mine_negatives(cls, dvecs, pos_idxs, num=1, nxs=None, eps=1e-9):
        """
        triplets =
             are a selection of anchor, positive, and negative annots
             chosen to be the hardest for each annotation (with a valid
             pos and neg partner) in the batch.

        Example:
            >>> from netharn.criterions.triplet import *
            >>> dvecs = torch.FloatTensor([
            ...     # Individual 1
            ...     [1.0, 0.0, 0.0, ],
            ...     [0.9, 0.1, 0.0, ],  # Looks like 2 [1]
            ...     # Individual 2
            ...     [0.0, 1.0, 0.0, ],
            ...     [0.0, 0.9, 0.1, ],  # Looks like 3 [3]
            ...     # Individual 3
            ...     [0.0, 0.0, 1.0, ],
            ...     [0.1, 0.0, 0.9, ],  # Looks like 1 [5]
            >>> ])
            >>> import itertools as it
            >>> clique2 = np.array(list(it.combinations(range(2), 2)))
            >>> pos_idxs = np.vstack([clique2, clique2 + 2, clique2 + 4])
            >>> num = 1
            >>> pos_dists, neg_dists, triples = TripletLoss.mine_negatives(dvecs, pos_idxs, num)
            >>> print('neg_dists = {!r}'.format(neg_dists))
            >>> print('pos_dists = {!r}'.format(pos_dists))
            >>> assert torch.all(pos_dists < neg_dists)

        Example:
            >>> # xdoctest: +SKIP
            >>> import itertools as it
            >>> p = 3
            >>> k = 10
            >>> d = p
            >>> for p, k in it.product(range(3, 13), range(2, 13)):
            >>>     d = p
            >>>     def make_individual_dvecs(i):
            >>>         vecs = torch.zeros((k, d))
            >>>         vecs[:, i] = torch.linspace(0.9, 1.0, k)
            >>>         return vecs
            >>>     dvecs = torch.cat([make_individual_dvecs(i) for i in range(p)], dim=0)
            >>>     cliquek = np.array(list(it.combinations(range(k), 2)))
            >>>     pos_idxs = np.vstack([cliquek + (i * k) for i in range(p)])
            >>>     num = 1
            >>>     pos_dists, neg_dists, triples = TripletLoss.mine_negatives(dvecs, pos_idxs, num)
            >>>     assert torch.all(pos_dists < neg_dists)
            >>>     print('neg_dists = {!r}'.format(neg_dists))
            >>>     print('pos_dists = {!r}'.format(pos_dists))
            >>>     print('triples = {!r}'.format(triples))
            >>>     base = k
            >>>     for a, p, n in triples:
            >>>         x = a // base
            >>>         y = p // base
            >>>         z = n // base
            >>>         assert x == y, str([a, p, n])
            >>>         assert x != z, str([a, p, n])
        """
        dist = all_pairwise_distances(dvecs)

        with torch.no_grad():
            eps = 1e-9
            eye_adjm = torch.eye(len(dist)).byte()

            pos_adjm = torch.zeros(dist.shape).byte()
            # p1, p2 = list(zip(*pos_idxs))
            # pos_adjm[p1, p2] = 1

            flat_pos_idxs = np.ravel_multi_index(list(zip(*pos_idxs)), dist.shape)
            pos_adjm.view(-1)[flat_pos_idxs] = 1

            pos_adjm[eye_adjm] = 0
            pos_adjm[pos_adjm.t()] = 1

            neg_adjm = ~pos_adjm
            neg_adjm[eye_adjm] = 0
            neg_adjm[neg_adjm.t()] = 1

            # never consider near duplicates
            neg_adjm[dist < eps] = 0
            pos_adjm[dist < eps] = 0

            # Ignore reflexive cases
            # pos_adjm.triu_()
            # neg_adjm.triu_()

            # Each row gives a ranking of the indices
            sortx = dist.argsort(dim=1)

            triples = []
            for anchor_idx, (x, m, n) in enumerate(zip(sortx.data.cpu(), pos_adjm.data.cpu(), neg_adjm.data.cpu())):
                if torch.any(m) and torch.any(n):
                    if 0:
                        d = dist[anchor_idx]
                        all_pos_cands = torch.nonzero(m).view(-1)
                        all_pos_dists = d[m]

                        all_neg_cands = torch.nonzero(n).view(-1)
                        all_neg_dists = d[n]

                        # The hardest positives are at the back
                        pos_cands = all_pos_cands[all_pos_dists.argsort()]

                        # The hardest negatives are at the front
                        neg_cands = all_neg_cands[all_neg_dists.argsort()]
                        # print('pos_cands = {!r}'.format(pos_cands))
                        # print('neg_cands = {!r}'.format(neg_cands))
                    else:
                        # Hardest remaining positive and negative
                        pos_cands = x[m[x]]
                        neg_cands = x[n[x]]
                        # print('pos_cands = {!r}'.format(pos_cands))
                        # print('neg_cands = {!r}'.format(neg_cands))

                    num_ = min(len(neg_cands), len(pos_cands), num)
                    pos_idxs = pos_cands[-num_:]
                    neg_idxs = neg_cands[:num_]
                    for pox_idx, neg_idx in it.product(pos_idxs, neg_idxs):
                        triples.append((anchor_idx, int(pox_idx), int(neg_idx)))

            if len(triples) == 0:
                raise RuntimeError('unable to mine triples')

            A, P, N = np.array(triples).T

            if 0 and __debug__:
                if nxs is not None:
                    for a, p, n in triples:
                        na_ = nxs[a]
                        np_ = nxs[p]
                        nn_ = nxs[n]
                        import xdev
                        with xdev.embed_on_exception_context:
                            assert na_ == np_
                            assert np_ != nn_

        pos_dists = dist[A, P].sqrt()
        neg_dists = dist[A, N].sqrt()
        return pos_dists, neg_dists, triples

    @classmethod
    def hard_triples(cls, dvecs, labels):
        """
        https://github.com/adambielski/siamese-triplet/blob/master/utils.py#L58

        Example:
            >>> from netharn.criterions.triplet import *
            >>> dvecs = torch.FloatTensor([
            ...     # Individual 1
            ...     [1.0, 0.0, 0.0, ],
            ...     [0.9, 0.1, 0.0, ],  # Looks like 2 [1]
            ...     # Individual 2
            ...     [0.0, 1.0, 0.0, ],
            ...     [0.0, 0.9, 0.1, ],  # Looks like 3 [3]
            ...     # Individual 3
            ...     [0.0, 0.0, 1.0, ],
            ...     [0.1, 0.0, 0.9, ],  # Looks like 1 [5]
            >>> ])
            >>> import itertools as it
            >>> labels = torch.LongTensor([1, 1, 2, 2, 3, 3])
            >>> num = 1
            >>> pos_dists, neg_dists, triples = TripletLoss.hard_triples(dvecs, labels)
            >>> print('neg_dists = {!r}'.format(neg_dists))
            >>> print('pos_dists = {!r}'.format(pos_dists))
            >>> assert torch.all(pos_dists < neg_dists)
        """
        def pdist(vectors):
            distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
                dim=1).view(-1, 1)
            return distance_matrix

        # distance_matrix = torch.nn.functional.pdist(dvecs)
        dist = distance_matrix = pdist(dvecs)

        labels_ = labels.cpu().data.numpy()
        all_pairs = np.array(list(it.combinations(range(len(labels_)), 2)))
        all_pairs = torch.LongTensor(all_pairs)

        pos_flags = (labels_[all_pairs[:, 0]] == labels_[all_pairs[:, 1]]).nonzero()
        neg_flags = (labels_[all_pairs[:, 0]] != labels_[all_pairs[:, 1]]).nonzero()
        positive_pairs = all_pairs[pos_flags]
        negative_pairs = all_pairs[neg_flags]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        ppx_list = positive_pairs.data.cpu().numpy().tolist()
        npx_list = top_negative_pairs.data.cpu().numpy().tolist()

        triples = []
        for ppx, npx in zip(ppx_list, npx_list):
            ppx = set(ppx)
            npx = set(npx)
            isect = ppx & npx
            if isect:
                a = ub.peek(isect)
                p = ub.peek(ppx - {a})
                n = ub.peek(npx - {a})
                triples.append((a, p, n))

        A, P, N = np.array(triples).T
        pos_dists = dist[A, P]
        neg_dists = dist[A, N]

        return pos_dists, neg_dists, triples

    def _softmargin(self, pos_dists, neg_dists):
        # log1 = torch.zeros_like(x)[None, :]  # log(1) = 0
        # dist_and_log1 = torch.cat([self.margin + x[None, :], log1], dim=0)
        # loss = torch.logsumexp(dist_and_log1, dim=0)
        x = pos_dists - neg_dists
        loss = F.softplus(self.margin + x)  # log(1 + exp(x))
        return loss

    def _hardmargin(self, pos_dists, neg_dists):
        x = pos_dists - neg_dists
        loss = (self.margin + x).clamp_(0, None)  # [margin + x]_{+}
        return loss

    def forward(self, pos_dists, neg_dists):
        """
        Args:
            pos_dists (Tensor) [A x 1]: distance between the anchor and a positive
            neg_dists (Tensor) [A x 1]: distance between the anchor and a negative

        Notes:
            soft_triplet_loss = (1 / len(triplets)) * sum(log(1 + exp(dist[a, p] - dist[a, n])) for a, p, n in triplets)

        Example:
            >>> from netharn.criterions.triplet import *
            >>> xbasis = ybasis = np.linspace(0, 5, 16)
            >>> pos_dists, neg_dists = map(torch.FloatTensor, np.meshgrid(xbasis, ybasis))
            >>> hard_loss = TripletLoss(reduction='none', soft=False).forward(pos_dists, neg_dists)
            >>> soft_loss = TripletLoss(reduction='none', soft=True).forward(pos_dists, neg_dists)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import netharn as nh
            >>> nh.util.autompl()
            >>> nh.util.plot_surface3d(pos_dists, neg_dists, hard_loss.numpy(), pnum=(1, 2, 1),
            >>>                        xlabel='d_pos', ylabel='d_neg', zlabel='hard-loss', contour=True, cmap='magma')
            >>> nh.util.plot_surface3d(pos_dists, neg_dists, soft_loss.numpy(), pnum=(1, 2, 2),
            >>>                        xlabel='d_pos', ylabel='d_neg', zlabel='hard-loss', contour=True, cmap='magma')
        """
        if self.soft:
            loss = self._softmargin(pos_dists, neg_dists)
        else:
            loss = self._hardmargin(pos_dists, neg_dists)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction != 'none':
            raise KeyError(self.reduction)
        return loss
