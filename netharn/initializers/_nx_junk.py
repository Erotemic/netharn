

@profile
def _lcs2(hash1, hash2, open_to_close, node_affinity, open_to_tok, _memo, hash_decomp1, hash_decomp2, seq_to_hash1, seq_to_hash2):
    if not hash_decomp1[hash1][0] or not hash_decomp1[hash2][0]:
        seq1, a1, b1, head1, tail1, head1_tail1 = hash_decomp1[hash1]
        seq2, a2, b2, head2, tail2, head2_tail2 = hash_decomp2[hash2]
        return (seq1, seq2), 0
    else:
        # if len(seq2) < len(seq1):
        #     seq1, seq2 = seq2, seq1
        # key = (seq1, seq2)
        key1 = hash1
        key2 = hash2
        key = hash((key1, key2))
        if key in _memo:
            return _memo[key]

        seq1, a1, b1, head1_hash, tail1_hash, head1_tail1_hash = hash_decomp1[hash1]
        seq2, a2, b2, head2_hash, tail2_hash, head2_tail2_hash = hash_decomp2[hash2]

        # Case 2: The current edge in sequence1 is deleted
        best, val = _lcs2(head1_tail1_hash, hash2, open_to_close, node_affinity, open_to_tok, _memo, hash_decomp1, hash_decomp2, seq_to_hash1, seq_to_hash2)

        # Case 3: The current edge in sequence2 is deleted
        cand, val_alt = _lcs2(hash1, head2_tail2_hash, open_to_close, node_affinity, open_to_tok, _memo, hash_decomp1, hash_decomp2, seq_to_hash1, seq_to_hash2)
        if val_alt > val:
            best = cand

        # Case 1: The LCS involves this edge
        t1 = open_to_tok[a1[0]]
        t2 = open_to_tok[a2[0]]
        # if node_affinity(a1[0], a2[0]):
        if node_affinity(t1, t2):
            new_heads, pval_h = _lcs2(head1_hash, head2_hash, open_to_close, node_affinity, open_to_tok, _memo, hash_decomp1, hash_decomp2, seq_to_hash1, seq_to_hash2)
            new_tails, pval_t = _lcs2(tail1_hash, tail2_hash, open_to_close, node_affinity, open_to_tok, _memo, hash_decomp1, hash_decomp2, seq_to_hash1, seq_to_hash2)

            new_head1, new_head2 = new_heads
            new_tail1, new_tail2 = new_tails

            subseq1 = a1 + new_head1 + b1 + new_tail1
            subseq2 = a2 + new_head2 + b2 + new_tail2

            cand = (subseq1, subseq2)
            val_alt = pval_h + pval_t + 1
            if val_alt > val:
                best = cand

        found = (best, val)
        _memo[key] = found
        return found



def balanced_decomp_index(sequence, open_to_close):
    """
    open_to_close = {0: 1}
    sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    open_to_close = {'{': '}', '(': ')', '[': ']'}
    sequence = '({[[]]})[[][]]{{}}'
    seq = balanced_decomp_index(sequence, open_to_close)

    a1, b1, head1, tail1, head_tail = seq.decomp()
    print('tail1 = {!r}'.format(tail1))
    print('head1 = {!r}'.format(head1))
    print('head_tail = {!r}'.format(head_tail))


    a1, b1, head1, tail1 = balanced_decomp_unsafe2(sequence, open_to_close)
    head_tail = head1 + tail1
    print('tail1 = {!r}'.format(tail1))
    print('head1 = {!r}'.format(head1))
    print('head_tail = {!r}'.format(head_tail))

    """
    paired_idxs = [-1] * len(sequence)
    stack = []
    for idx, token in enumerate(sequence):
        if token in open_to_close:
            stack.append((token, idx))
        else:
            # Check that closing elements
            if not stack:
                raise UnbalancedException
            prev_open, prev_idx = stack.pop()
            want_close = open_to_close[prev_open]
            paired_idxs[prev_idx] = idx
            paired_idxs[idx] = prev_idx

            if token != want_close:
                raise UnbalancedException

    if USE_FAST_CAT_SHIFT_INDEX:
        paired_idxs = FastCatShiftIndex.from_single(paired_idxs)
    else:
        paired_idxs = np.array(paired_idxs)
    self = DecomposableSequence(sequence, paired_idxs, 0, len(sequence))
    return self
    # open_tok, close_tok, head, tail = self.decomp()
    # print('self = {!r}'.format(self))
    # print('head = {!r}'.format(head))
    # print('tail = {!r}'.format(tail))
    # open_tok1, close_tok1, head1, tail1 = tail.decomp()
    # print('head1 = {!r}'.format(head1))
    # print('tail1 = {!r}'.format(tail1))
    # open_tok2, close_tok2, head2, tail2 = tail1.decomp()
    # print('head2 = {!r}'.format(head2))
    # print('tail2 = {!r}'.format(tail2))

    # head_tail = head + tail
    # print('head_tail = {!r}'.format(head_tail))

    # return pop_open, pop_close, head, tail


class DecomposableSequence(ub.NiceRepr):
    def __init__(self, seq, paired_idxs, offset=0, length=None):
        self.seq = seq
        self.paired_idxs = paired_idxs
        self.offset = offset
        self.length = length

    def __nice__(self):
        return self.seq[self.offset:self.offset + self.length]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.seq[idx + self.offset]

    @profile
    def decomp(self):
        """
        from netharn.initializers._nx_extensions import *  # NOQA
        open_to_close = {0: 1}
        sequence = [0, 0, 0, 1, 1, 1, 0, 1]
        open_to_close = {'{': '}', '(': ')', '[': ']'}
        sequence = '({[[]]})[[][]]{{}}'
        self = balanced_decomp_index(sequence, open_to_close)
        a1, b1, head1, tail1, head_tail = self.decomp()

        tail1.decomp()
        """
        offset = self.offset
        open_idx = offset
        close_idx = self.paired_idxs[open_idx]

        open_tok = self.seq[open_idx:open_idx + 1]
        close_tok = self.seq[close_idx:close_idx + 1]

        head_len = close_idx - open_idx - 1
        tail_len = self.length - (close_idx - offset) - 1
        # print('head_len = {!r}, tail_len={}'.format(head_len, tail_len))
        head_pos = offset + 1
        tail_pos = close_idx + 1

        head = DecomposableSequence(self.seq, self.paired_idxs, head_pos, head_len)
        tail = DecomposableSequence(self.seq, self.paired_idxs, tail_pos, tail_len)

        head_tail = head + tail
        return open_tok, close_tok, head, tail, head_tail

    def __eq__(self, other):
        return self.seq == other.seq

    def __hash__(self):
        return hash(self.seq)

    @profile
    def rebase(self, new_offset=0):
        offset = self.offset
        shift = (offset - new_offset)
        sl = slice(offset, offset + self.length)
        newseq = self.seq[sl]
        new_paired_idxs = self.paired_idxs[sl]
        if shift:
            if USE_FAST_CAT_SHIFT_INDEX:
                new_paired_idxs.add_inplace(-shift)
            else:
                new_paired_idxs = new_paired_idxs - shift
        return newseq, new_paired_idxs

    @profile
    def __add__(self, other):
        """
        self = head1
        other = tail1
        """
        # Each rebase is 37% of the computation for a total 74%
        newseq1, new_paired_idxs1 = self.rebase()
        newseq2, new_paired_idxs2 = other.rebase(new_offset=len(newseq1))
        newseq = newseq1 + newseq2
        # This is about 15% of the computation
        if USE_FAST_CAT_SHIFT_INDEX:
            new_paired_idxs = new_paired_idxs1.concat(new_paired_idxs2)
        else:
            new_paired_idxs = np.concatenate([new_paired_idxs1, new_paired_idxs2], axis=0)
        new = DecomposableSequence(newseq, new_paired_idxs, 0, len(newseq))
        return new

    @profile
    def combine(self, a, b, other):
        """
        self = head1
        other = tail1
        """
        newseq1, new_paired_idxs1 = self.rebase(new_offset=1)
        new_head_len = len(newseq1)
        newseq2, new_paired_idxs2 = other.rebase(new_offset=(new_head_len + 2))
        newseq = a + newseq1 + b + newseq2

        if USE_FAST_CAT_SHIFT_INDEX:
            apart = FastCatShiftIndex.from_single([new_head_len + 1])
            bpart = FastCatShiftIndex.from_single([0])
            new_paired_idxs = apart + new_paired_idxs1 + bpart + new_paired_idxs2
        else:
            new_paired_idxs = np.r_[new_head_len + 1, new_paired_idxs1, 0, new_paired_idxs2]
        new = DecomposableSequence(newseq, new_paired_idxs, 0, len(newseq))
        return new


class FastCatShiftIndex(ub.NiceRepr):
    """
    The idea is to make the operations very fast:
        * adding an offset to each item
        * concatenating two arrays
        * slicing within an array

    Example:
        >>> self = FastCatShiftIndex.from_single([1, 2, 3])
        >>> other = FastCatShiftIndex.from_single([1, 2, 3])
        >>> other.add_inplace(10)
        >>> new = self.concat(other)

        >>> self = FastCatShiftIndex.from_single([1] * 20)
        >>> start = 0
        >>> stop = 16
        >>> self.subslice(0, 25)

        >>> self = new
        >>> start, stop = 4, 5
        >>> new = self.subslice(start, stop)
        >>> index = slice(start, stop)
        >>> self[index]
    """
    # Can we make an efficient data structure fo this?  The concats and the
    # offsets are the culprit for most of the runtime.
    def __init__(self, datas, offsets, indexer):
        self.datas = datas
        self.offsets = offsets
        self.indexer = indexer

    def add_inplace(self, offset):
        self.offsets = [o + offset for o in self.offsets]

    def subslice(self, start, stop):
        outer1, inner1  = self.indexer.unravel(start)
        outer2, inner2  = self.indexer.unravel(stop)

        if outer1 == outer2:
            new_datas = [self.datas[outer1][inner1:inner2]]
            new_offsets = [self.offsets[outer1]]
        else:
            first = [self.datas[outer1][inner1:]]
            inner = self.datas[outer1 + 1:outer2]
            ender = [self.datas[outer2][:inner2]]
            new_datas = first + inner + ender
            new_offsets = self.offsets[outer1:outer2 + 1]
        new_indexer = self.indexer._subslice(outer1, outer2, inner1, inner2)
        new = self.__class__(new_datas, new_offsets, new_indexer)
        return new

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.subslice(index.start, index.stop)
        else:
            outer, inner  = self.indexer.unravel(index)
            offset = self.offsets[outer]
            return self.datas[outer][inner] + offset

    @classmethod
    def from_single(cls, data, offset=0):
        indexer = FlatIndexer([len(data)], np.array([len(data)]))
        self = cls([data], [offset], indexer)
        return self

    def __nice__(self):
        return self.resolve()

    def __add__(self, other):
        return self.concat(other)

    def concat(self, other):
        new_indexer = self.indexer.concat(other.indexer)
        new_datas = self.datas + other.datas
        new_offsets = self.offsets + other.offsets
        new = self.__class__(new_datas, new_offsets, new_indexer)
        return new

    def resolve(self):
        return [d + offset for data, offset in zip(self.datas, self.offsets) for d in data]


@profile
def _lcs(seq1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo):
    if not seq1:
        return (seq1, seq1), 0
    elif not seq2:
        return (seq2, seq2), 0
    else:
        # if len(seq2) < len(seq1):
        #     seq1, seq2 = seq2, seq1
        # key = (seq1, seq2)
        key1 = hash(seq1)
        key2 = hash(seq2)
        key = hash((key1, key2))
        if key in _memo:
            return _memo[key]

        # TODO: we can probably just do a single linear run through the
        # sequences to index the sub-sequence locations and then apply an
        # offset when we run the decomposed sequence.
        if DECOMP_SEQ_INDEX:
            a1, b1, head1, tail1, head1_tail1 = seq1.decomp()
            a2, b2, head2, tail2, head2_tail2 = seq2.decomp()
        else:
            if key1 in _seq_memo:
                a1, b1, head1, tail1, head1_tail1 = _seq_memo[key1]
            else:
                a1, b1, head1, tail1 = balanced_decomp_unsafe2(seq1, open_to_close)
                head1_tail1 = head1 + tail1
                _seq_memo[key1] = a1, b1, head1, tail1, head1_tail1

            if key2 in _seq_memo:
                a2, b2, head2, tail2, head2_tail2 = _seq_memo[key2]
            else:
                a2, b2, head2, tail2 = balanced_decomp_unsafe2(seq2, open_to_close)
                head2_tail2 = head2 + tail2
                _seq_memo[key2] = a2, b2, head2, tail2, head2_tail2

        if 1:
            # TODO: IS THIS THE CORRECT MODIFICATION TO THE RECURRANCE TO
            # ACHIEVE A SUBTREE ISOMORPHISM INSTEAD OF AN EMBEDDING?
            best, val = _lcs(head1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

            cand, val_alt = _lcs(tail1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
            if val_alt > val:
                best = cand
                val = val_alt

            cand, val_alt = _lcs(seq1, head2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
            if val_alt > val:
                best = cand
                val = val_alt

            cand, val_alt = _lcs(seq1, tail2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
            if val_alt > val:
                best = cand
                val = val_alt

        else:
            # Case 2: The current edge in sequence1 is deleted
            best, val = _lcs(head1_tail1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

            # Case 3: The current edge in sequence2 is deleted
            cand, val_alt = _lcs(seq1, head2_tail2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
            if val_alt > val:
                best = cand
                val = val_alt

        # Case 1: The LCS involves this edge
        t1 = open_to_tok[a1[0]]
        t2 = open_to_tok[a2[0]]
        # if node_affinity(a1[0], a2[0]):
        affinity = node_affinity(t1, t2)
        if affinity:
            new_heads, pval_h = _lcs(head1, head2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
            new_tails, pval_t = _lcs(tail1, tail2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

            new_head1, new_head2 = new_heads
            new_tail1, new_tail2 = new_tails

            if DECOMP_SEQ_INDEX:
                subseq1 = new_head1.combine(a1, b1, new_tail1)
                subseq2 = new_head2.combine(a2, b2, new_tail2)
            else:
                subseq1 = a1 + new_head1 + b1 + new_tail1
                subseq2 = a2 + new_head2 + b2 + new_tail2

            cand = (subseq1, subseq2)
            val_alt = pval_h + pval_t + affinity
            if val_alt > val:
                best = cand
                val = val_alt

        found = (best, val)
        _memo[key] = found
        return found


"""

    ndata = [1, 10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
    ndata = [1200, 1500, 1700]
    ydata = []
    for n in ndata:
        print('n = {!r}'.format(n))
        with ub.Timer('check') as timer:
            data3 = data1[0:n]
            data4 = data2[0:n]
            matching = maximum_common_ordered_paths(data3, data4, sep='/')
        ydata.append(timer.elapsed)

    res2 = numpy.polyfit(ndata, ydata, deg=2, full=True)
    coeff2 = res2[0]
    xs2 = np.arange(0, 1200)
    ys2 = np.polyval(coeff2, xs2)

    np.polyval(coeff2, [2000, 3000]) / 60 / 60
    np.polyval(coeff3, [2000, 3000]) / 60 / 60
    np.polyval(coeff4, [2000, 3000]) / 60 / 60

    #  Hours for 2000, 3000 based on coeff2
    # 0.18249042, 0.43731033

    # For Coeff 3
    # 0.31122992, 1.05361379

    # For Coeff 4
    # array([0.3522069 , 1.38993684])


    np.polyval(coeff3, [10000]) / 60 / 60
    np.polyval(coeff4, [10000]) / 60 / 60

    res3 = numpy.polyfit(ndata, ydata, deg=3, full=True)
    coeff3 = res3[0]
    xs3 = np.arange(0, 1200, step=50)
    ys3 = np.polyval(coeff3, xs3)

    res4 = numpy.polyfit(ndata, ydata, deg=4, full=True)
    coeff4 = res4[0]
    xs4 = np.arange(0, 1200, step=50)
    ys4 = np.polyval(coeff4, xs4)
    print('coeff2 = {}'.format(ub.repr2(coeff2, nl=1, precision=3)))
    print('coeff3 = {}'.format(ub.repr2(coeff3, nl=1, precision=3)))
    print('coeff4 = {}'.format(ub.repr2(coeff4, nl=1, precision=3)))

    import kwplot
    xydata = {
        'measured': [ndata, ydata],
        'fit_deg2': [xs2, ys2],
        'fit_deg4': [xs4, ys4],
        'fit_deg3': [xs3, ys3],
    }
    marker = {
        'measured': 'o',
        'fit_deg2': '',
        'fit_deg4': '+',
        'fit_deg3': 'x',
    }
    linestyle = {
        'measured': '-',
        'fit_deg2': '--',
        'fit_deg3': '--',
        'fit_deg4': '--',
    }
    kwplot.multi_plot(xydata=xydata, xlabel='n', ylabel='seconds', linestyle=linestyle, marker=marker, fnum=1, doclf=True)


"""



@profile
def longest_common_balanced_sequence(seq1, seq2, open_to_close, node_affinity=None, open_to_tok=None):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/_nx_extensions.py longest_common_balanced_sequence:0 --profile && cat profile_output.txt

    Example:
        >>> tree1 = random_ordered_tree(100, seed=1)
        >>> tree2 = random_ordered_tree(100, seed=2)
        >>> seq1, open_to_close, toks = tree_to_balanced_sequence(tree1)
        >>> seq2, open_to_close, toks = tree_to_balanced_sequence(tree2, open_to_close, toks)
        >>> longest_common_balanced_sequence(seq1, seq2, open_to_close)
    """
    if node_affinity is None:
        node_affinity = operator.eq
    _memo = {}
    _seq_memo = {}

    if DECOMP_SEQ_INDEX:
        seq1 = balanced_decomp_index(seq1, open_to_close)
        seq2 = balanced_decomp_index(seq2, open_to_close)

    if open_to_tok is None:
        class Dummy:
            def __getitem__(self, key):
                return key
        open_to_tok = Dummy()

    if USE_PRE_DECOMP:
        raise NotImplementedError
        all_decomp1 = _generate_all_decompositions(seq1, open_to_close)
        all_decomp2 = _generate_all_decompositions(seq2, open_to_close)

        def _make_hash_decomp(all_decomp):
            seq_to_hash = {}
            hash_to_decomp = {}

            for seq, decomp1 in all_decomp.items():
                a, b, head, tail, head_tail = decomp1
                seq_hash = hash(seq)
                head_hash = hash(head)
                tail_hash = hash(tail)
                head_tail_hash = hash(head_tail)
                seq_to_hash[seq] = seq_hash
                hash_to_decomp[seq_hash] = seq, a, b, head_hash, tail_hash, head_tail_hash
            return seq_to_hash, hash_to_decomp

        seq_to_hash1, hash_decomp1 = _make_hash_decomp(all_decomp1)
        seq_to_hash2, hash_decomp2 = _make_hash_decomp(all_decomp2)

        hash1 = seq_to_hash1[seq1]
        hash2 = seq_to_hash2[seq2]

        best, value = _lcs2(hash1, hash2, open_to_close, node_affinity, open_to_tok, _memo, hash_decomp1, hash_decomp2, seq_to_hash1, seq_to_hash2)
    else:
        best, value = _lcs(seq1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

    if DECOMP_SEQ_INDEX:
        # unpack
        a, b = best
        best = (a.seq, b.seq)
    return best, value


def _generate_all_decompositions(seq, open_to_close):
    """
    Can doing this a-priori speed up the algorithm?

    open_to_close = {0: 1}
    sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    open_to_close = {'{': '}', '(': ')', '[': ']'}
    seq = '({[[]]})[[][]]{{}}'
    pop_open, pop_close, head, tail = balanced_decomp(seq, open_to_close)

    >>> tree = random_ordered_tree(1000)
    >>> seq, open_to_close, toks = tree_to_balanced_sequence(tree)
    >>> all_decomp = _generate_all_decompositions(seq, open_to_close)
    """
    _memo = {}
    def _gen(seq):
        if not seq:
            pass
            # yield None
        elif seq in _memo:
            pass
            # yield (seq, _memo[seq])
        else:
            pop_open, pop_close, head, tail = balanced_decomp(seq, open_to_close)
            head_tail = head + tail
            _memo[seq] = (pop_open, pop_close, head, tail, head_tail)
            yield (seq, _memo[seq])
            yield from _gen(head_tail)
            yield from _gen(head)
            yield from _gen(tail)
    all_decomp = dict(_gen(seq))
    return all_decomp
