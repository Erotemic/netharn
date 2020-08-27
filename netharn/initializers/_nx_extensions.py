"""
EXPERIMENTAL : NEW WORK ON THIS IS HAPPENING IN NETWORKX ITSELF

ONCE THAT IS DONE I WILL MODIFY THE ALGORITHMS HERE.
"""

import operator
import ubelt as ub
import networkx as nx

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# Cython gives a 40x speed boost in the nx version but not here
TRY_USE_CYTHON = 0


@profile
def maximum_common_ordered_tree_embedding(tree1, tree2, node_affinity='auto'):
    """
    Finds the maximum common subtree-embedding between two ordered trees.

    A tree S is an embedded subtree of T if it can be obtained from T by a
    series of edge contractions.

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
        >>> from netharn.initializers._nx_extensions import _lcs, _print_forest
        >>> def random_ordered_tree(n, seed=None):
        >>>     tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
        >>>     otree = nx.OrderedDiGraph()
        >>>     otree.add_edges_from(tree.edges)
        >>>     return otree
        >>> tree1 = random_ordered_tree(10, seed=1)
        >>> tree2 = random_ordered_tree(10, seed=2)
        >>> print('tree1')
        >>> _print_forest(tree1)
        >>> print('tree2')
        >>> _print_forest(tree2)

        >>> embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2 )
        >>> print('embedding1')
        >>> _print_forest(embedding1)
        >>> print('embedding2')
        >>> _print_forest(embedding2)
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
    embedding1 = seq_to_tree(subseq1, open_to_close, toks)
    embedding2 = seq_to_tree(subseq2, open_to_close, toks)
    return embedding1, embedding2


@profile
def maximum_common_ordered_subtree_isomorphism(tree1, tree2, node_affinity='auto'):
    """
    Isomorphic version of `maximum_common_ordered_tree_embedding`.

    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/_nx_extensions.py maximum_common_ordered_subtree_isomorphism:1 --profile && cat profile_output.txt

    Ignore:
        >>> from netharn.initializers._nx_extensions import *  # NOQA
        >>> from netharn.initializers._nx_extensions import _lcs, _print_forest
        >>> def random_ordered_tree(n, seed=None):
        >>>     tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
        >>>     otree = nx.OrderedDiGraph()
        >>>     otree.add_edges_from(tree.edges)
        >>>     return otree
        >>> tree1 = random_ordered_tree(10, seed=3)
        >>> tree2 = random_ordered_tree(10, seed=2)
        >>> tree1.add_edges_from(tree2.edges, weight=1)
        >>> tree1 = nx.minimum_spanning_arborescence(tree1)
        >>> tree2.add_edges_from(tree1.edges, weight=1)
        >>> tree2 = nx.minimum_spanning_arborescence(tree2)
        >>> tree1.remove_edge(4, 7)
        >>> tree1.remove_edge(4, 9)
        >>> tree1.add_edge(4, 10)
        >>> tree1.add_edge(10, 7)
        >>> tree1.add_edge(10, 9)
        >>> #tree1.add_edges_from([(9, 11), (11, 12), (12, 13), (13, 14)])
        >>> #tree2.add_edges_from([(9, 11), (11, 12), (12, 13), (13, 14)])
        >>> tree1.add_edges_from([(9, 11), (11, 12)])
        >>> tree2.add_edges_from([(9, 11), (11, 12)])
        >>> tree2.add_edge(100, 0)
        >>> tree1.add_edge(102, 100)
        >>> tree1.add_edge(100, 101)
        >>> tree1.add_edge(101, 0)
        >>> tree1.add_edge(5, 201)
        >>> tree1.add_edge(5, 202)
        >>> tree1.add_edge(5, 203)
        >>> tree1.add_edge(201, 2000)
        >>> tree1.add_edge(2000, 2001)
        >>> tree1.add_edge(2001, 2002)
        >>> tree1.add_edge(2002, 2003)
        >>> tree2.add_edge(5, 202)
        >>> tree2.add_edge(5, 203)
        >>> tree2.add_edge(5, 201)
        >>> tree2.add_edge(201, 2000)
        >>> tree2.add_edge(2000, 2001)
        >>> tree2.add_edge(2001, 2002)
        >>> tree2.add_edge(2002, 2003)
        >>> print('-----')
        >>> print('tree1')
        >>> _print_forest(tree1)
        >>> print('tree2')
        >>> _print_forest(tree2)
        >>> subtree1, subtree2 = maximum_common_ordered_subtree_isomorphism(tree1, tree2 )
        >>> print('-----')
        >>> print('subtree1')
        >>> _print_forest(subtree1)
        >>> print('subtree2')
        >>> _print_forest(subtree2)
        >>> embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2)
        >>> print('-----')
        >>> print('embedding1')
        >>> _print_forest(embedding1)
        >>> print('embedding2')
        >>> _print_forest(embedding2)
        >>> if 0:
        >>>     ti = timerit.Timerit(6, bestof=2, verbose=2)
        >>>     for timer in ti.reset('isomorphism'):
        >>>         with timer:
        >>>             maximum_common_ordered_subtree_isomorphism(tree1, tree2 )
        >>>     for timer in ti.reset('embedding'):
        >>>         with timer:
        >>>             maximum_common_ordered_tree_embedding(tree1, tree2 )
        >>> from networkx import isomorphism
        >>> assert isomorphism.DiGraphMatcher(tree1, subtree1).subgraph_is_isomorphic()
        >>> assert isomorphism.DiGraphMatcher(tree2, subtree2).subgraph_is_isomorphic()
        >>> list(isomorphism.DiGraphMatcher(tree1, tree2).subgraph_isomorphisms_iter())
        >>> list(isomorphism.DiGraphMatcher(tree1, tree2).subgraph_monomorphisms_iter())
        >>> list(isomorphism.DiGraphMatcher(subtree1, subtree2).subgraph_isomorphisms_iter())
        >>> list(isomorphism.DiGraphMatcher(tree1, subtree1).subgraph_isomorphisms_iter())
        >>> list(isomorphism.DiGraphMatcher(tree2, subtree2).subgraph_isomorphisms_iter())

    Ignore:
        >>> from netharn.initializers._nx_extensions import *  # NOQA
        >>> from netharn.initializers._nx_extensions import _lcs, _print_forest
        >>> def random_ordered_tree(n, seed=None):
        >>>     if n > 0:
        >>>         tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
        >>>     otree = nx.OrderedDiGraph()
        >>>     if n > 0:
        >>>         otree.add_edges_from(tree.edges)
        >>>     return otree
        >>> import random
        >>> rng = random.Random(90269698983701724775426457020022)
        >>> num = 1000
        >>> def _gen_seeds(num):
        >>>     for _ in range(num):
        >>>         yield (rng.randint(0, 50), rng.randint(0, 50), rng.randint(0, 2 ** 64), rng.randint(0, 2 ** 64))
        >>> for n1, n2, s1, s2 in ub.ProgIter(_gen_seeds(num=num), total=num, verbose=3):
        >>>     tree1 = random_ordered_tree(n1, seed=s1)
        >>>     tree2 = random_ordered_tree(n2, seed=s2)
        >>>     #print('-----')
        >>>     #print('tree1')
        >>>     #_print_forest(tree1)
        >>>     #print('tree2')
        >>>     #_print_forest(tree2)
        >>>     subtree1, subtree2 = maximum_common_ordered_subtree_isomorphism(tree1, tree2, node_affinity='auto')
        >>>     #print('-----')
        >>>     #print('subtree1')
        >>>     #_print_forest(subtree1)
        >>>     #print('subtree2')
        >>>     #_print_forest(subtree2)
        >>>     from networkx import isomorphism
        >>>     assert isomorphism.DiGraphMatcher(tree1, subtree1).subgraph_is_isomorphic()
        >>>     assert isomorphism.DiGraphMatcher(tree2, subtree2).subgraph_is_isomorphic()

    """
    try:
        if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree1)):
            raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')
        if not (isinstance(tree1, nx.OrderedDiGraph) and nx.is_forest(tree2)):
            raise nx.NetworkXNotImplemented('only implemented for directed ordered trees')
    except nx.NetworkXPointlessConcept:
        subtree1 = nx.OrderedDiGraph()
        subtree2 = nx.OrderedDiGraph()
        return subtree1, subtree2

    # Convert the trees to balanced sequences
    sequence1, open_to_close, toks = tree_to_balanced_sequence(tree1, open_to_close=None, toks=None, mode='chr')
    sequence2, open_to_close, toks = tree_to_balanced_sequence(tree2, open_to_close, toks, mode='chr')
    seq1 = sequence1
    seq2 = sequence2

    open_to_tok = ub.invert_dict(toks)

    # Solve the longest common balanced sequence problem
    best, value = longest_common_isomorphic_sequence(
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
                    elif mode == 'paren':
                        open_tok = '{}('.format(v)
                        close_tok = '){}'.format(v)
                    elif mode == 'chr':
                        open_tok = str(v)
                        close_tok = str(v) + u'\u0301'
                        # chr(ord(v) + 128)
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


@profile
def longest_common_balanced_sequence(seq1, seq2, open_to_close, node_affinity='auto', open_to_tok=None):
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
    if node_affinity == 'auto':
        node_affinity = operator.eq
    if node_affinity is None:
        def _matchany(a, b):
            return True
        node_affinity = _matchany
    _memo = {}
    _seq_memo = {}
    if open_to_tok is None:
        class Dummy:
            def __getitem__(self, key):
                return key
        open_to_tok = Dummy()
    best, value = _lcs(seq1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
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
        key1 = hash(seq1)  # using hash(seq) is faster than seq itself
        key2 = hash(seq2)
        key = hash((key1, key2))
        if key in _memo:
            return _memo[key]

        # TODO: we can probably just do a single linear run through the
        # sequences to index the sub-sequence locations and then apply an
        # offset when we run the decomposed sequence.
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
def longest_common_isomorphic_sequence(seq1, seq2, open_to_close, node_affinity='auto', open_to_tok=None):
    if node_affinity == 'auto':
        node_affinity = operator.eq
    if node_affinity is None:
        def _matchany(a, b):
            return True
        node_affinity = _matchany
    _memo = {}
    _seq_memo = {}
    if open_to_tok is None:
        class Dummy:
            def __getitem__(self, key):
                return key
        open_to_tok = Dummy()
    best_lvl, value_lvl, best_low, value_low = _lcsi(seq1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

    if value_lvl > value_low:
        best = best_lvl
        value = value_lvl
    else:
        best = best_low
        value = value_low

    return best, value


@profile
def _lcsi(seq1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo):
    """
    Prototype isomorphic only version
    """
    if not seq1:
        return (seq1, seq1), 0, (seq1, seq1), 0
    elif not seq2:
        return (seq2, seq2), 0, (seq2, seq2), 0
    else:
        key1 = hash(seq1)
        key2 = hash(seq2)
        key = hash((key1, key2))
        if key in _memo:
            return _memo[key]

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

        # TODO: IS THIS THE CORRECT MODIFICATION TO THE RECURRANCE TO
        # ACHIEVE A SUBTREE ISOMORPHISM INSTEAD OF AN EMBEDDING?
        r"""

        tree1 = nx.OrderedDiGraph()
        tree1.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        tree1.add_edges_from([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'e'), ('b', 'f'), ('c', 'g')])

        _print_forest(tree1)

        └── a
            ├── b
            │   ├── e
            │   └── f
            ├── c
            │   └── g
            └── d

        seq1, open_to_close, toks = tree_to_balanced_sequence(tree1, mode='chr')
        a, b, head1, tail1 = balanced_decomp(seq1, open_to_close)
        _print_forest(seq_to_tree(head1, open_to_close, toks))
        _print_forest(seq_to_tree(tail1, open_to_close, toks))

        CONTRACTED NODE:
        a

        HEAD (children of the contracted node)

        ├── b
        │   ├── e
        │   └── f
        ├── c
        │   └── g
        └── d

        TAIL (right siblings of the contracted node)
        --

        a, b, head11, tail11 = balanced_decomp(head1, open_to_close)
        _print_forest(seq_to_tree(head11, open_to_close, toks))
        _print_forest(seq_to_tree(tail11, open_to_close, toks))

        CONTRACTED NODE:
        b

        HEAD OF HEAD
        ├── e
        └── f

        TAIL OF HEAD
        ├── c
        │   └── g
        └── d


        The problem here is that if you are at a level where two levels down
        there are two matches, you will return those two matches as the best
        solution at that layer, and therefore you won't flag if there is a
        feasible solution at this layer. This is a problem because that
        feasible low-value solution might be part of the highest value
        solution.

        Perhaps we return two solutions at each step: the solution value at
        this level if one exists, and the solution value at any other depth.
        We are allowed to add to the first, but can take the second if we want
        to.

        This should work because we know a solution that skipped a layer will
        never be added to, and we are always keeping track of the solution that
        might change. By the time we get to the root level, we have enough info
        to know which is better.
        """

        # If any of these cases are selected we are not choosing the leftmost
        # node as our match
        best_lvl, val_lvl, best_low, val_low = None, -1, None, -1

        # TODO: it may be the case that some of these tests are redundant, in
        # which case we could simplify and speed up the algorithm. We would
        # need to prove that the value in one of these tests was always lower
        # than the value in another one of these tests, in that case we could
        # remove the former.

        # When using the head part of the decomp, we can only update the "low" candidate
        cand_lvl, score_lvl, cand_low, score_low = _lcsi(head1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
        if score_low > val_low:
            val_low = score_low
            best_low = cand_low
        if score_lvl > val_low:
            val_low = score_lvl
            best_low = cand_lvl

        cand_lvl, score_lvl, cand_low, score_low = _lcsi(seq1, head2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
        if score_low > val_low:
            val_low = score_low
            best_low = cand_low
        if score_lvl > val_low:
            val_low = score_lvl
            best_low = cand_lvl

        # As long as we are only using the tail part of the decomp we can update
        # both the lvl and low scores
        cand_lvl, score_lvl, cand_low, score_low = _lcsi(tail1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
        if score_lvl > val_lvl:
            val_lvl = score_lvl
            best_lvl = cand_lvl
        if score_low > val_low:
            val_low = score_low
            best_low = cand_low

        cand_lvl, score_lvl, cand_low, score_low = _lcsi(seq1, tail2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
        if score_lvl > val_lvl:
            val_lvl = score_lvl
            best_lvl = cand_lvl
        if score_low > val_low:
            val_low = score_low
            best_low = cand_low

        # This is the case where we found a matching node
        t1 = open_to_tok[a1[0]]
        t2 = open_to_tok[a2[0]]
        affinity = node_affinity(t1, t2)
        if affinity:

            new_heads_lvl, pval_h_lvl, new_heads_low, pval_h_low = _lcsi(head1, head2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
            new_tails_lvl, pval_t_lvl, new_tails_low, pval_t_low = _lcsi(tail1, tail2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

            # Add to the best solution at the former level
            score_lvl = pval_h_lvl + pval_t_lvl + affinity
            if score_lvl > val_lvl:
                new_head1, new_head2 = new_heads_lvl
                new_tail1, new_tail2 = new_tails_lvl
                subseq1 = a1 + new_head1 + b1 + new_tail1
                subseq2 = a2 + new_head2 + b2 + new_tail2
                cand_lvl = (subseq1, subseq2)
                val_lvl = score_lvl
                best_lvl = cand_lvl

            # In my big tests these were never hit once, is it true that this
            # test was covered by a previous case?
            cand_low = new_heads_low
            score_low = pval_h_low
            if score_low > val_low:
                val_low = score_low
                best_low = cand_low

            cand_low = new_tails_low
            score_low = pval_t_low
            if score_low > val_low:
                val_low = score_low
                best_low = cand_low

        # We return two solutions:
        # the best AT this level (lvl), and the best AT any lowers (low).
        found = (best_lvl, val_lvl, best_low, val_low)
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
    if len(graph.nodes) == 0:
        print('--')
        return
    assert nx.is_forest(graph)

    def _recurse(node, indent='', islast=False):
        if islast:
            this_prefix = indent + '└── '
            next_prefix = indent + '    '
        else:
            this_prefix = indent + '├── '
            next_prefix = indent + '│   '
        label = graph.nodes[node].get('label', node)
        print(this_prefix + str(label))
        graph.succ[node]
        children = graph.succ[node]
        for idx, child in enumerate(children, start=1):
            islast_next = (idx == len(children))
            _recurse(child, indent=next_prefix, islast=islast_next)

    sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
    for idx, node in enumerate(sources, start=1):
        islast_next = (idx == len(sources))
        _recurse(node, indent='', islast=islast_next)


def maximum_common_ordered_paths(paths1, paths2, sep='/'):
    import networkx as nx

    # the longest common balanced sequence problem
    def _affinity(tok1, tok2):
        score = 0
        for t1, t2 in zip(tok1[::-1], tok2[::-1]):
            if t1 == t2:
                score += 1
            else:
                break
        return score
        # return tok1[-1] == tok2[-1]
    node_affinity = _affinity
    # import operator
    # eq = operator.eq

    def paths_to_tree(paths):
        tree = nx.OrderedDiGraph()
        for path in sorted(paths):
            parts = tuple(path.split(sep))
            node_path = []
            for i in range(1, len(parts) + 1):
                node = parts[0:i]
                tree.add_node(node)
                tree.nodes[node]['label'] = node[-1]
                node_path.append(node)
            for u, v in ub.iter_window(node_path, 2):
                tree.add_edge(u, v)
        return tree

    tree1 = paths_to_tree(paths1)
    tree2 = paths_to_tree(paths2)

    subtree1, subtree2 = maximum_common_ordered_tree_embedding(tree1, tree2, node_affinity=node_affinity)
    # subtree1, subtree2 = maximum_common_ordered_subtree_isomorphism(tree1, tree2, node_affinity=node_affinity)

    subpaths1 = [sep.join(node) for node in subtree1.nodes if subtree1.out_degree[node] == 0]
    subpaths2 = [sep.join(node) for node in subtree2.nodes if subtree2.out_degree[node] == 0]
    return subpaths1, subpaths2
