import numpy as np
import operator
import ubelt as ub
from netharn.util.util_misc import FlatIndexer
import networkx as nx

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# These did not help the speed
DECOMP_SEQ_INDEX = 0
USE_FAST_CAT_SHIFT_INDEX = 0
TRY_USE_CYTHON = 0

USE_PRE_DECOMP = 0


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


@profile
def maximum_common_ordered_tree_embedding(tree1, tree2, node_affinity=None):
    """
    Finds the maximum common subtree-embedding between two ordered trees.

    Note this produces a subtree embedding, which is not necessarilly a
    subgraph isomorphism (although a subgraph isomorphism is also an
    embedding.)

    The maximum common embedded subtree problem can be solved in in
    `O(n1 * n2 * min(d1, l1) * min(d2, l2))` time on ordered trees with n1 and
    n2 nodes, of depth d1 and d2 and with l1 and l2 leaves, respectively

    Implements algorithm described in [1]_.

    References:
        On the Maximum Common Embedded Subtree Problem for Ordered Trees
        https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf

        http://algo.inria.fr/flajolet/Publications/FlSiSt90.pdf

    Notes:
        Exact algorithms for computing the tree edit distance between unordered trees - https://pdf.sciencedirectassets.com/271538/1-s2.0-S0304397510X00299/1-s2.0-S0304397510005463/main.pdf ?

        Tree Edit Distance and Common Subtrees - https://upcommons.upc.edu/bitstream/handle/2117/97554/R02-20.pdf

        A Survey on Tree Edit Distance and Related Problems - https://grfia.dlsi.ua.es/ml/algorithms/references/editsurvey_bille.pdf

    Args:

        tree1 (nx.OrderedDiGraph): first ordered tree
        tree2 (nx.OrderedDiGraph): second ordered tree
        node_affinity (callable): function

    Example:
        >>> from netharn.initializers._nx_extensions import *  # NOQA
        >>> from netharn.initializers.functional import _best_prefix_transform
        >>> def random_ordered_tree(n, seed=None):
        >>>     tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
        >>>     otree = nx.OrderedDiGraph()
        >>>     otree.add_edges_from(tree.edges)
        >>>     return otree
        >>> tree1 = random_ordered_tree(10, seed=1)
        >>> tree2 = random_ordered_tree(10, seed=2)
        >>> _print_forest(tree1)
        >>> _print_forest(tree2)

        >>> subtree1, subtree2 = maximum_common_ordered_tree_embedding(tree1, tree2 )
        >>> _print_forest(subtree1)
        >>> _print_forest(subtree2)

    Ignore:
        >>> from networkx import isomorphism
        >>> assert isomorphism.DiGraphMatcher(tree1, subtree1).subgraph_is_isomorphic()
        >>> assert isomorphism.DiGraphMatcher(tree2, subtree2).subgraph_is_isomorphic()

        >>> list(isomorphism.DiGraphMatcher(tree1, tree2).subgraph_isomorphisms_iter())
        >>> list(isomorphism.DiGraphMatcher(tree1, tree2).subgraph_monomorphisms_iter())

        >>> list(isomorphism.DiGraphMatcher(subtree1, subtree2).subgraph_isomorphisms_iter())


        >>> from networkx import isomorphism
        >>> tree1 = nx.DiGraph(nx.path_graph(4, create_using=nx.OrderedDiGraph()))
        >>> tree2 = nx.DiGraph(nx.path_graph(4, create_using=nx.OrderedDiGraph()))

        >>> DiGM = isomorphism.DiGraphMatcher(tree1, tree2)
        >>> DiGM.is_isomorphic()

        >>> list(DiGM.subgraph_isomorphisms_iter())

        # the longest common balanced sequence problem
        def _matchable(tok1, tok2):
            return tok1.value[-1] == tok2.value[-1]
        node_affinity = _matchable
        print([n for n in tree1.nodes if tree1.in_degree[n] > 1])
        print([n for n in tree2.nodes if tree2.in_degree[n] > 1])
        _print_forest(tree1)
        _print_forest(tree2)
        subtree1, subtree2 = maximum_common_ordered_tree_embedding(tree1, tree2, node_affinity=node_affinity)
        # for n in subtree1.nodes:
        #     subtree1.nodes[n]['label'] = n[-1]
        _print_forest(subtree1)
        _print_forest(subtree2)

        tree1_remain = tree1.copy()
        tree1_remain.remove_nodes_from(subtree1.nodes)
        _print_forest(tree1_remain)

        tree = tree1
    """
    if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree1)):
        raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')
    if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree2)):
        raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')

    # Convert the trees to balanced sequences
    sequence1, open_to_close, toks = tree_to_balanced_sequence(tree1, open_to_close=None, toks=None)
    sequence2, open_to_close, toks = tree_to_balanced_sequence(tree2, open_to_close, toks)
    seq1 = sequence1
    seq2 = sequence2

    open_to_tok = ub.invert_dict(toks)

    # Solve the longest common balanced sequence problem
    best, value = longest_common_balanced_sequence(
        seq1, seq2, open_to_close, open_to_tok=open_to_tok, node_affinity=node_affinity)
    subseq1, subseq2 = best

    # Convert the subsequence back into a tree
    subtree1 = seq_to_tree(subseq1, open_to_close, toks)
    subtree2 = seq_to_tree(subseq2, open_to_close, toks)
    return subtree1, subtree2


class UnbalancedException(Exception):
    pass


def tree_to_balanced_sequence(tree, open_to_close=None, toks=None, mode='tuple'):
    from collections import namedtuple
    Token = namedtuple('Token', ['action', 'value'])
    # mapping between opening and closing tokens
    sources = [n for n in tree.nodes if tree.in_degree[n] == 0]
    sequence = []

    if open_to_close is None:
        open_to_close = {}
    if toks is None:
        toks = {}

    for source in sources:
        for u, v, etype in nx.dfs_labeled_edges(tree, source=source):
            if etype == 'forward':
                # u has been visited by v has not
                if v not in toks:
                    if mode == 'tuple':
                        # TODO: token encoding scheme where subdirectories
                        # are matchable via a custom operation.
                        # open_tok = '<{}>'.format(v)
                        # close_tok = '</{}>'.format(v)
                        open_tok = Token('open', v)
                        close_tok = Token('close', v)
                    elif mode == 'number':
                        open_tok = len(toks) + 1
                        close_tok = -open_tok
                    toks[v] = open_tok
                    open_to_close[open_tok] = close_tok
                open_tok = toks[v]
                sequence.append(open_tok)
            elif etype == 'reverse':
                # Both u and v are visited and the edge is in the tree
                close_tok = open_to_close[toks[v]]
                sequence.append(close_tok)
            else:
                raise KeyError(etype)
    sequence = tuple(sequence)
    return sequence, open_to_close, toks


def seq_to_tree(subseq, open_to_close, toks):
    open_to_tok = ub.invert_dict(toks)
    subtree = nx.OrderedDiGraph()
    stack = []
    for token in subseq:
        if token in open_to_close:
            node = open_to_tok[token]
            if stack:
                parent = open_to_tok[stack[-1]]
                subtree.add_edge(parent, node)
            else:
                subtree.add_node(node)
            stack.append(token)
        else:
            if not stack:
                raise Exception
            prev_open = stack.pop()
            want_close = open_to_close[prev_open]
            if token != want_close:
                raise Exception
    return subtree


def random_ordered_tree(n, seed=None):
    tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
    otree = nx.OrderedDiGraph()
    otree.add_edges_from(tree.edges)
    return otree


@profile
def generate_balance_unsafe_python(sequence, open_to_close):
    """
    Benchmark:
        >>> tree = random_ordered_tree(1000)
        >>> sequence, open_to_close, toks = tree_to_balanced_sequence(tree, mode='tuple')
        >>> sequence, open_to_close, toks = tree_to_balanced_sequence(tree, mode='number')
        >>> import timerit
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
        >>> for timer in ti.reset('time'):
        >>>     with timer:
        >>>         list(generate_balance_unsafe(sequence, open_to_close))
        >>> import timerit
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
        >>> for timer in ti.reset('time'):
        >>>     with timer:
        >>>         list(generate_balance_unsafe_cython(sequence, open_to_close))
    """
    stacklen = 0
    for token in sequence:
        if token in open_to_close:
            stacklen += 1
        else:
            stacklen -= 1
        yield stacklen == 0, token


@profile
def balanced_decomp(sequence, open_to_close):
    """
    Note this is not exactly the same as the decomposition in the paper.
    That is because we also return the "wrapping" element, and we let the
    user do the head + tail concatenation.

    Example:
        >>> open_to_close = {0: 1}
        >>> sequence = [0, 0, 0, 1, 1, 1, 0, 1]
        >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
        >>> sequence = '({[[]]})[[][]]'
        >>> a1, b1, head, tail = balanced_decomp(sequence, open_to_close)
        >>> a2, b2, tail1, tail2 = balanced_decomp(tail, open_to_close)
    """
    gen = generate_balance(sequence, open_to_close)

    bal_curr, tok_curr = next(gen)
    pop_open = sequence[0:1]
    want_close = open_to_close[tok_curr]

    head_stop = 1
    for head_stop, (bal_curr, tok_curr) in enumerate(gen, start=1):
        if tok_curr is None:
            break
        elif bal_curr and tok_curr == want_close:
            pop_close = sequence[head_stop:head_stop + 1]
            break
    head = sequence[1:head_stop]
    # if __debug__:
    #     list(gen)  # exhaust the generator to check we are balanced
    tail = sequence[head_stop + 1:]
    return pop_open, pop_close, head, tail


@profile
def balanced_decomp_unsafe(sequence, open_to_close):
    """
    open_to_close = {0: 1}
    sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    open_to_close = {'{': '}', '(': ')', '[': ']'}
    sequence = '({[[]]})[[][]]'
    a1, b1, head, tail = balanced_decomp(sequence, open_to_close)
    a2, b2, tail1, tail2 = balanced_decomp(tail, open_to_close)

    Benchmark:
        >>> from netharn.initializers._nx_extensions import *  # NOQA
        >>> tree = random_ordered_tree(100)
        >>> sequence, open_to_close, toks = tree_to_balanced_sequence(tree)
        >>> import timerit
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2, unit='us')
        >>> for timer in ti.reset('safe-python'):
        >>>     with timer:
        >>>         list(balanced_decomp(sequence, open_to_close))
        >>> for timer in ti.reset('unsafe-python'):
        >>>     with timer:
        >>>         list(balanced_decomp_unsafe(sequence, open_to_close))
        >>> for timer in ti.reset('unsafe-python-v2'):
        >>>     with timer:
        >>>         list(balanced_decomp_unsafe2_python(sequence, open_to_close))
        >>> for timer in ti.reset('unsafe-c/python-v2'):
        >>>     with timer:
        >>>         list(balanced_decomp_unsafe2(sequence, open_to_close))
    """
    gen = generate_balance_unsafe(sequence, open_to_close)

    bal_curr, tok_curr = next(gen)
    pop_open = sequence[0:1]
    want_close = open_to_close[tok_curr]

    head_stop = 1
    for head_stop, (bal_curr, tok_curr) in enumerate(gen, start=1):
        if bal_curr and tok_curr == want_close:
            pop_close = sequence[head_stop:head_stop + 1]
            break
    head = sequence[1:head_stop]
    tail = sequence[head_stop + 1:]
    return pop_open, pop_close, head, tail


@profile
def balanced_decomp_unsafe2_python(sequence, open_to_close):
    stacklen = 0
    seq_iter = iter(sequence)
    tok_curr = next(seq_iter)
    stacklen += 1 if tok_curr in open_to_close else -1
    want_close = open_to_close[tok_curr]

    head_stop = 1
    for head_stop, tok_curr in enumerate(seq_iter, start=1):
        stacklen += 1 if tok_curr in open_to_close else -1
        if stacklen == 0 and tok_curr == want_close:
            break

    pop_close = sequence[head_stop:head_stop + 1]
    pop_open = sequence[0:1]
    head = sequence[1:head_stop]
    tail = sequence[head_stop + 1:]
    return pop_open, pop_close, head, tail


generate_balance_unsafe = generate_balance_unsafe_python
balanced_decomp_unsafe2 = balanced_decomp_unsafe2_python


if TRY_USE_CYTHON:
    try:
        from netharn.initializers import _nx_extensions_cython_backend as cyb

        generate_balance_unsafe_cython = cyb.generate_balance_unsafe_cython
        generate_balance_unsafe        = cyb.generate_balance_unsafe_cython

        balanced_decomp_unsafe2_cython = cyb.balanced_decomp_unsafe2_cython
        balanced_decomp_unsafe2        = cyb.balanced_decomp_unsafe2_cython
    except Exception:
        pass


def generate_balance(sequence, open_to_close, safe=True):
    """
    Args:
        safe (bool): if True we will error if the sequence is not balanced
            if you are SURE the sequence is balanced set safe=False to slightly
            improve runtime.


    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/_nx_extensions.py generate_balance:1 --profile

    Example:
        >>> open_to_close = {0: 1}
        >>> sequence = [0, 0, 0, 1, 1, 1]
        >>> gen = list(generate_balance(sequence, open_to_close))
        >>> for flag, token in gen:
        >>>     print('flag={:d}, token={}'.format(flag, token))

    Example:
        >>> tree = random_ordered_tree(1000)
        >>> sequence, open_to_close, toks = tree_to_balanced_sequence(tree)
        >>> gen = list(generate_balance(sequence, open_to_close))
        >>> for flag, token in gen:
        >>>     print('flag={:d}, token={}'.format(flag, token))

    Benchmark:
        >>> from netharn.initializers._nx_extensions import *  # NOQA
        >>> tree = random_ordered_tree(100)
        >>> sequence, open_to_close, toks = tree_to_balanced_sequence(tree)
        >>> import timerit
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2, unit='us')
        >>> for timer in ti.reset('safe-python'):
        >>>     with timer:
        >>>         list(generate_balance(sequence, open_to_close))
        >>> for timer in ti.reset('unsafe-python'):
        >>>     with timer:
        >>>         list(generate_balance_unsafe(sequence, open_to_close))

    Ignore:
        from netharn.initializers._nx_extensions import *  # NOQA
        from numba import jit
        jit_generate_balance = jit(forceobj=True)(generate_balance)

        open_to_close = {0: 1}
        sequence = [0, 0, 0, 1, 1, 1]
        list(jit_generate_balance(sequence, open_to_close))

        tree = random_ordered_tree(1000)
        sequence, open_to_close, toks = tree_to_balanced_sequence(tree)

        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2, unit='us')

        for timer in ti.reset('safe-python'):
            with timer:
                list(generate_balance(sequence, open_to_close))

        for timer in ti.reset('unsafe-python'):
            with timer:
                list(generate_balance_unsafe(sequence, open_to_close))

        for timer in ti.reset('numba'):
            with timer:
                list(jit_generate_balance(sequence, open_to_close))
    """
    if safe:
        stack = []
        # Traversing the Expression
        for token in sequence:

            if token in open_to_close:
                # Push opening elements onto the stack
                stack.append(token)
            else:
                # Check that closing elements
                if not stack:
                    raise UnbalancedException
                prev_open = stack.pop()
                want_close = open_to_close[prev_open]

                if token != want_close:
                    raise UnbalancedException

            # If the stack is empty the sequence is currently balanced
            currently_balanced = not bool(stack)
            yield currently_balanced, token

        if stack:
            raise UnbalancedException
    else:
        yield from generate_balance_unsafe(sequence, open_to_close)


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

    Benchmark:
        >>> tree1 = random_ordered_tree(20, seed=1)
        >>> tree2 = random_ordered_tree(20, seed=2)
        >>> seq1, open_to_close, toks = tree_to_balanced_sequence(tree1)
        >>> seq2, open_to_close, toks = tree_to_balanced_sequence(tree2, open_to_close, toks)
        >>> longest_common_balanced_sequence(seq1, seq2, open_to_close)

        >>> import timerit
        >>> ti = timerit.Timerit(10, bestof=10, verbose=2, unit='ms')
        >>> from netharn.initializers import _nx_extensions
        >>> _nx_extensions.DECOMP_SEQ_INDEX = 0
        >>> for timer in ti.reset('without-index'):
        >>>     with timer:
        >>>         _nx_extensions.longest_common_balanced_sequence(seq1, seq2, open_to_close)
        >>> _nx_extensions.DECOMP_SEQ_INDEX = 1
        >>> for timer in ti.reset('with-index'):
        >>>     with timer:
        >>>         _nx_extensions.longest_common_balanced_sequence(seq1, seq2, open_to_close)

        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/netharn'))
        from netharn.initializers._nx_extensions import *  # NOQA
        from netharn.initializers._nx_extensions import _best_prefix_transform, _lcs, _print_forest

    open_to_close = {'0': '1'}
    seq1 = '0010010010111100001011011011'
    seq2 = '001000101101110001000100101110111011'

    open_to_close = {'(': ')'}
    seq1 = '(()(()(()())))(((()())())())'
    seq2 = '(()((()())()))((()((()(()()))()))())'
    longest_common_balanced_sequence(seq1, seq2, open_to_close)

    open_to_close = {'0': '1'}
    seq1 = '0010010010111100001011011011'
    seq2 = '001000101101110001000100101110111011'
    longest_common_balanced_sequence(seq1, seq2, open_to_close)

    open_to_close = {'0': '1'}
    seq1 = '001101'
    seq2 = '00110011'
    seq1 = '001101'
    seq2 = '00110011'
    longest_common_balanced_sequence(seq1, seq2, open_to_close)

    open_to_close = {'{': '}', '(': ')', '[': ']'}
    seq1 = '(({}{([])}[{}]))'
    seq2 = '((({}[{{}}])))'

    seq1 = '({[[[]]]}){}'
    seq2 = '{}{[[[]]]}'
    best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close)
    subseq1, subseq2 = best
    print('subseq1 = {!r}'.format(subseq1))
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
            # TODO: need to return the correspondence between the
            # matches and the original nodes.
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
            # TODO: need to return the correspondence between the
            # matches and the original nodes.
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


def _print_forest(graph):
    """
    Nice ascii representation of a forest

    Ignore:
        graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        _print_forest(graph)

        graph = CategoryTree.demo('coco').graph
        _print_forest(graph)
    """
    assert nx.is_forest(graph)
    from kwcoco.category_tree import to_directed_nested_tuples
    encoding = to_directed_nested_tuples(graph)
    def _recurse(encoding, indent=''):
        for idx, item in enumerate(encoding):
            node, data, children = item
            if idx == len(encoding) - 1:
                this_prefix = indent + '└── '
                next_prefix = indent + '    '
            else:
                this_prefix = indent + '├── '
                next_prefix = indent + '│   '
            label = graph.nodes[node].get('label', node)
            print(this_prefix + str(label))
            _recurse(children, indent=next_prefix)
    _recurse(encoding)
