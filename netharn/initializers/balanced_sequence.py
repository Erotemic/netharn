import operator
import ubelt as ub
import networkx as nx

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


# @profile
def longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_tok=None, node_affinity='auto', impl='iter'):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/balanced_sequence.py longest_common_balanced_sequence:0 --profile && cat profile_output.txt

    Example:
        >>> from netharn.initializers.balanced_sequence import *  # NOQA
        >>> from netharn.initializers.balanced_sequence import _lcs_iter_prehash, _lcs_iter_simple, _lcs_recurse, _print_forest
        >>> tree1 = random_ordered_tree(5, seed=10, pool='[{(')
        >>> tree2 = random_ordered_tree(5, seed=3, pool='[{(')

        >>> import kwarray
        >>> rng = kwarray.ensure_rng(3432432, 'python')
        >>> tree1 = random_ordered_tree(100, seed=rng, pool='[{(')
        >>> tree2 = random_ordered_tree(100, seed=rng, pool='[{(')
        >>> if len(tree1.nodes) < 20:
        >>>     _print_forest(tree1)
        >>>     _print_forest(tree2)
        >>> seq1, open_to_close, toks = tree_to_balanced_sequence(tree1, mode='label', strhack=1)
        >>> seq2, open_to_close, toks = tree_to_balanced_sequence(tree2, open_to_close, toks, mode='label', strhack=1)
        >>> full_seq1 = seq1
        >>> full_seq2 = seq2
        >>> print('seq1 = {!r}'.format(seq1))
        >>> print('seq2 = {!r}'.format(seq2))
        >>> open_to_tok = ub.invert_dict(toks)
        >>> node_affinity = operator.eq
        >>> with ub.Timer('iterative-alt2'):
        >>>     best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_tok, impl='iter-alt2')
        >>>     print('val1, best1 = {}, {!r}'.format(val1, best1))
        >>> with ub.Timer('iterative-alt1'):
        >>>     best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_tok, impl='iter-alt1')
        >>>     print('val1, best1 = {}, {!r}'.format(val1, best1))
        >>> with ub.Timer('iterative'):
        >>>     best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_tok, impl='iter')
        >>>     print('val1, best1 = {}, {!r}'.format(val1, best1))
        >>> with ub.Timer('recursive'):
        >>>     best2, val2 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_tok, impl='recurse')
        >>>     print('val2, best2 = {}, {!r}'.format(val2, best2))
        >>> #with ub.Timer('iterative-prehash'):
        >>> #    best1, val1 = longest_common_balanced_sequence(seq1, seq2, open_to_close, open_to_tok, impl='iter-prehash')
        >>> #    print('val1, best1 = {}, {!r}'.format(val1, best1))
    """
    if node_affinity == 'auto' or node_affinity == 'eq':
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
    full_seq1 = seq1
    full_seq2 = seq2
    if impl == 'recurse':
        best, value = _lcs_recurse(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
    elif impl == 'iter':
        best, value = _lcs_iter_simple(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok)
    elif impl == 'iter-prehash':
        best, value = _lcs_iter_prehash(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok)
    elif impl == 'iter-alt1':
        best, value = _lcs_iter_simple_alt1(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok)
    elif impl == 'iter-alt2':
        best, value = _lcs_iter_simple_alt2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok)
    else:
        raise KeyError(impl)
    return best, value


@profile
def _lcs_iter_simple(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok):
    """
    Converts _lcs_recursive to an iterative algorithm using a fairly
    straightforward method that effectivly simulates callstacks
    """
    all_decomp1 = generate_all_decompositions(full_seq1, open_to_close, open_to_tok)
    all_decomp2 = generate_all_decompositions(full_seq2, open_to_close, open_to_tok)

    args0 = (full_seq1, full_seq2)
    frame0 = args0
    stack = [frame0]

    _results = {}
    # Populate base cases
    empty1 = type(ub.peek(all_decomp1.keys()))()
    empty2 = type(ub.peek(all_decomp2.keys()))()
    best = (empty1, empty2)
    base_result = (0, best)
    for seq1 in all_decomp1.keys():
        key1 = seq1
        t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[key1]
        _results[(seq1, empty2)] = base_result
        _results[(head1, empty2)] = base_result
        _results[(tail1, empty2)] = base_result
        _results[(head_tail1, empty2)] = base_result

    for seq2 in all_decomp2.keys():
        key2 = seq2
        t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[key2]
        _results[(empty1, seq2)] = base_result
        _results[(empty1, head2)] = base_result
        _results[(empty1, tail2)] = base_result
        _results[(empty1, head_tail2)] = base_result

    del args0
    del frame0
    del empty1
    del empty2
    del best
    del base_result

    missing_frames = []
    while stack:
        key = stack.pop()
        if key not in _results:
            seq1, seq2 = key
            missing_frames.clear()

            # try:
            t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[seq1]
            # except KeyError:
            #     a1, b1, head1, tail1 = balanced_decomp_unsafe(seq1, open_to_close)
            #     head_tail1 = head1 + tail1
            #     all_decomp1[seq1] = a1, b1, head1, tail1, head_tail1

            # try:
            t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[seq2]
            # except KeyError:
            #     a2, b2, head2, tail2 = balanced_decomp_unsafe(seq2, open_to_close)
            #     head_tail2 = head2 + tail2
            #     all_decomp2[seq2] = a2, b2, head2, tail2, head_tail2

            # Case 2: The current edge in sequence1 is deleted
            try:
                try_key = (head_tail1, seq2)
                cand1 = _results[try_key]
            except KeyError:
                missing_frames.append(try_key)

            # Case 3: The current edge in sequence2 is deleted
            try:
                try_key = (seq1, head_tail2)
                cand2 = _results[try_key]
            except KeyError:
                missing_frames.append(try_key)

            # Case 1: The LCS involves this edge
            affinity = node_affinity(t1, t2)
            if affinity:
                try:
                    try_key = (head1, head2)
                    pval_h, new_heads = _results[try_key]
                except KeyError:
                    missing_frames.append(try_key)

                try:
                    try_key = (tail1, tail2)
                    pval_t, new_tails = _results[try_key]
                except KeyError:
                    missing_frames.append(try_key)

                if not missing_frames:
                    new_head1, new_head2 = new_heads
                    new_tail1, new_tail2 = new_tails

                    subseq1 = a1 + new_head1 + b1 + new_tail1
                    subseq2 = a2 + new_head2 + b2 + new_tail2

                    res3 = (subseq1, subseq2)
                    val3 = pval_h + pval_t + affinity
                    cand3 = (val3, res3)
            else:
                cand3 = (-1, None)

            if missing_frames:
                # We did not solve this frame yet
                stack.append(key)
                stack.extend(missing_frames)
                # stack.extend(missing_frames[::-1])
            else:
                # We solved the frame
                _results[key] = max(cand1, cand2, cand3)

    val, best = _results[key]
    found = (best, val)
    return found


@profile
def _lcs_iter_simple_alt1(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok):
    """
    Depth first stack trajectory
    """
    all_decomp1 = generate_all_decompositions(full_seq1, open_to_close, open_to_tok)
    all_decomp2 = generate_all_decompositions(full_seq2, open_to_close, open_to_tok)

    args0 = (full_seq1, full_seq2)
    frame0 = args0
    stack = [frame0]

    _results = {}
    # Populate base cases
    empty1 = type(ub.peek(all_decomp1.keys()))()
    empty2 = type(ub.peek(all_decomp2.keys()))()
    best = (empty1, empty2)
    base_result = (0, best)
    for seq1 in all_decomp1.keys():
        key1 = seq1
        t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[key1]
        _results[(seq1, empty2)] = base_result
        _results[(head1, empty2)] = base_result
        _results[(tail1, empty2)] = base_result
        _results[(head_tail1, empty2)] = base_result

    for seq2 in all_decomp2.keys():
        key2 = seq2
        t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[key2]
        _results[(empty1, seq2)] = base_result
        _results[(empty1, head2)] = base_result
        _results[(empty1, tail2)] = base_result
        _results[(empty1, head_tail2)] = base_result

    del args0
    del frame0
    del empty1
    del empty2
    del best
    del base_result

    while stack:
        key = stack.pop()
        if key not in _results:
            seq1, seq2 = key

            t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[seq1]

            t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[seq2]

            # Case 2: The current edge in sequence1 is deleted
            try:
                try_key = (head_tail1, seq2)
                cand1 = _results[try_key]
            except KeyError:
                stack.append(key)
                stack.append(try_key)
                continue

            # Case 3: The current edge in sequence2 is deleted
            try:
                try_key = (seq1, head_tail2)
                cand2 = _results[try_key]
            except KeyError:
                stack.append(key)
                stack.append(try_key)
                continue

            # Case 1: The LCS involves this edge
            affinity = node_affinity(t1, t2)
            if affinity:
                try:
                    try_key = (head1, head2)
                    pval_h, new_heads = _results[try_key]
                except KeyError:
                    stack.append(key)
                    stack.append(try_key)
                    continue

                try:
                    try_key = (tail1, tail2)
                    pval_t, new_tails = _results[try_key]
                except KeyError:
                    stack.append(key)
                    stack.append(try_key)
                    continue

                new_head1, new_head2 = new_heads
                new_tail1, new_tail2 = new_tails

                subseq1 = a1 + new_head1 + b1 + new_tail1
                subseq2 = a2 + new_head2 + b2 + new_tail2

                res3 = (subseq1, subseq2)
                val3 = pval_h + pval_t + affinity
                cand3 = (val3, res3)
            else:
                cand3 = (-1, None)

            # We solved the frame
            _results[key] = max(cand1, cand2, cand3)

    val, best = _results[key]
    found = (best, val)
    return found


@profile
def _lcs_iter_simple_alt2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok):
    """
    Depth first stack trajectory and replace try except statements with ifs
    """
    all_decomp1 = generate_all_decompositions(full_seq1, open_to_close, open_to_tok)
    all_decomp2 = generate_all_decompositions(full_seq2, open_to_close, open_to_tok)

    key0 = (full_seq1, full_seq2)
    frame0 = key0
    stack = [frame0]

    _results = {}
    # Populate base cases
    empty1 = type(ub.peek(all_decomp1.keys()))()
    empty2 = type(ub.peek(all_decomp2.keys()))()
    best = (empty1, empty2)
    base_result = (0, best)
    for seq1 in all_decomp1.keys():
        key1 = seq1
        t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[key1]
        _results[(seq1, empty2)] = base_result
        _results[(head1, empty2)] = base_result
        _results[(tail1, empty2)] = base_result
        _results[(head_tail1, empty2)] = base_result

    for seq2 in all_decomp2.keys():
        key2 = seq2
        t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[key2]
        _results[(empty1, seq2)] = base_result
        _results[(empty1, head2)] = base_result
        _results[(empty1, tail2)] = base_result
        _results[(empty1, head_tail2)] = base_result

    del frame0
    del empty1
    del empty2
    del best
    del base_result

    while stack:
        key = stack[-1]
        if key not in _results:
            seq1, seq2 = key

            t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[seq1]
            t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[seq2]

            # Case 2: The current edge in sequence1 is deleted
            try_key = (head_tail1, seq2)
            if try_key in _results:
                cand1 = _results[try_key]
            else:
                # stack.append(key)
                stack.append(try_key)
                continue

            # Case 3: The current edge in sequence2 is deleted
            try_key = (seq1, head_tail2)
            if try_key in _results:
                cand2 = _results[try_key]
            else:
                # stack.append(key)
                stack.append(try_key)
                continue

            # Case 1: The LCS involves this edge
            affinity = node_affinity(t1, t2)
            if affinity:
                try_key = (head1, head2)
                if try_key in _results:
                    pval_h, new_heads = _results[try_key]
                else:
                    # stack.append(key)
                    stack.append(try_key)
                    continue

                try_key = (tail1, tail2)
                if try_key in _results:
                    pval_t, new_tails = _results[try_key]
                else:
                    # stack.append(key)
                    stack.append(try_key)
                    continue

                new_head1, new_head2 = new_heads
                new_tail1, new_tail2 = new_tails

                subseq1 = a1 + new_head1 + b1 + new_tail1
                subseq2 = a2 + new_head2 + b2 + new_tail2

                res3 = (subseq1, subseq2)
                val3 = pval_h + pval_t + affinity
                cand3 = (val3, res3)
            else:
                cand3 = (-1, None)

            # We solved the frame
            _results[key] = max(cand1, cand2, cand3)
        stack.pop()

    val, best = _results[key0]
    found = (best, val)
    return found


@profile
def _lcs_iter_prehash(full_seq1, full_seq2, open_to_close, node_affinity, open_to_tok):
    """
    Version of the lcs iterative algorithm where we precompute hash values

    This is actually slower than the simple version
    """
    def decomp_info(seq, open_to_close):
        pop_open, pop_close, head, tail = balanced_decomp_unsafe(seq, open_to_close)
        head_tail = head + tail
        head_key = hash(head)
        tail_key = hash(tail)
        head_tail_key = hash(head_tail)
        tok = open_to_tok[pop_open[0]]
        a = pop_open
        b = pop_close
        info = (tok, seq, head, tail, head_tail, head_key, tail_key, head_tail_key, a, b)
        return info

    def gen_decomp_v2(seq, open_to_close):
        _genmemo = {}
        def _gen(seq):
            if seq:
                key = hash(seq)
                if key not in _genmemo:
                    info = decomp_info(seq, open_to_close)
                    head, tail, head_tail = info[2:5]
                    _genmemo[key] = info
                    yield (seq, _genmemo[key])
                    yield from _gen(head_tail)
                    yield from _gen(head)
                    yield from _gen(tail)
        all_decomp = dict(_gen(seq))
        return all_decomp

    all_decomp1 = gen_decomp_v2(full_seq1, open_to_close)
    all_decomp2 = gen_decomp_v2(full_seq2, open_to_close)

    key_decomp1 = {}
    key_decomp2 = {}
    _results = {}
    # Populate base cases
    empty1 = type(ub.peek(all_decomp1.keys()))()
    empty2 = type(ub.peek(all_decomp2.keys()))()
    empty1_key = hash(empty1)
    empty2_key = hash(empty2)
    best = (empty1, empty2)
    base_result = (0, best)
    for seq1, info1 in all_decomp1.items():
        seq1_key = hash(seq1)
        head1_key, tail1_key, head_tail1_key = all_decomp1[seq1][5:8]
        _results[(seq1_key, empty2_key)] = base_result
        _results[(head1_key, empty2_key)] = base_result
        _results[(tail1_key, empty2_key)] = base_result
        _results[(head_tail1_key, empty2_key)] = base_result
        key_decomp1[seq1_key] = info1

    for seq2, info2 in all_decomp2.items():
        seq2_key = hash(seq2)
        head2_key, tail2_key, head_tail2_key = all_decomp2[seq2][5:8]
        _results[(empty1_key, seq2_key)] = base_result
        _results[(empty1_key, head2_key)] = base_result
        _results[(empty1_key, tail2_key)] = base_result
        _results[(empty1_key, head_tail2_key)] = base_result
        key_decomp2[seq2_key] = info2

    full_seq1_key = hash(full_seq1)
    full_seq2_key = hash(full_seq2)
    key0 = (full_seq1_key, full_seq2_key)
    frame0 = key0, full_seq1, full_seq2
    stack = [frame0]
    missing_frames = []
    while stack:
        frame = stack.pop()
        key, seq1, seq2 = frame
        seq1_key, seq2_key = key
        if key not in _results:
            missing_frames.clear()

            try:
                info1 = key_decomp1[seq1_key]
            except KeyError:
                info1 = decomp_info(seq1, open_to_close)
                key_decomp1[seq1_key] = info1
            tok1, seq1, head1, tail1, head_tail1, head1_key, tail1_key, head_tail1_key, a1, b1 = info1

            try:
                info2 = key_decomp2[seq2_key]
            except KeyError:
                info2 = decomp_info(seq2, open_to_close)
                key_decomp2[seq2_key] = info2
            tok2, seq2, head2, tail2, head_tail2, head2_key, tail2_key, head_tail2_key, a2, b2 = info2

            affinity = node_affinity(tok1, tok2)

            # Case 2: The current edge in sequence1 is deleted
            try:
                try_key = (head_tail1_key, seq2_key)
                cand1 = _results[try_key]
            except KeyError:
                miss_frame = try_key, head_tail1, seq2
                missing_frames.append(miss_frame)

            # Case 3: The current edge in sequence2 is deleted
            try:
                try_key = (seq1_key, head_tail2_key)
                cand2 = _results[try_key]
            except KeyError:
                miss_frame = try_key, seq1, head_tail2
                missing_frames.append(miss_frame)

            # Case 1: The LCS involves this edge
            if affinity:
                try:
                    try_key = (head1_key, head2_key)
                    pval_h, new_heads = _results[try_key]
                except KeyError:
                    miss_frame = try_key, head1, head2
                    missing_frames.append(miss_frame)

                try:
                    try_key = (tail1_key, tail2_key)
                    pval_t, new_tails = _results[try_key]
                except KeyError:
                    miss_frame = try_key, tail1, tail2
                    missing_frames.append(miss_frame)

                if not missing_frames:
                    new_head1, new_head2 = new_heads
                    new_tail1, new_tail2 = new_tails

                    subseq1 = a1 + new_head1 + b1 + new_tail1
                    subseq2 = a2 + new_head2 + b2 + new_tail2

                    res3 = (subseq1, subseq2)
                    val3 = pval_h + pval_t + affinity
                    cand3 = (val3, res3)
            else:
                cand3 = (-1, None)

            if missing_frames:
                # We did not solve this frame yet
                stack.append(frame)
                stack.extend(missing_frames[::-1])
            else:
                # We solved the frame
                _results[key] = max(cand1, cand2, cand3)

    # The stack pop is our solution
    (val, best) = _results[key]
    found = (best, val)
    return found


def generate_all_decompositions(seq, open_to_close, open_to_tok=None):
    """
    Can doing this a-priori speed up the algorithm?

    open_to_close = {0: 1}
    sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    open_to_close = {'{': '}', '(': ')', '[': ']'}
    seq = '({[[]]})[[][]]{{}}'
    pop_open, pop_close, head, tail = balanced_decomp(seq, open_to_close)

    >>> tree = random_ordered_tree(10)
    >>> seq, open_to_close, toks = tree_to_balanced_sequence(tree)
    >>> all_decomp = generate_all_decompositions(seq, open_to_close)
    """
    if open_to_tok is None:
        class Dummy:
            def __getitem__(self, key):
                return key
        open_to_tok = Dummy()
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
            tok = open_to_tok[pop_open[0]]
            _memo[seq] = (tok, pop_open, pop_close, head, tail, head_tail)
            yield (seq, _memo[seq])
            yield from _gen(head_tail)
            yield from _gen(head)
            yield from _gen(tail)
    all_decomp = dict(_gen(seq))
    return all_decomp


@profile
def _lcs_recurse(seq1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo):
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
            a1, b1, head1, tail1 = balanced_decomp_unsafe(seq1, open_to_close)
            head1_tail1 = head1 + tail1
            _seq_memo[key1] = a1, b1, head1, tail1, head1_tail1

        if key2 in _seq_memo:
            a2, b2, head2, tail2, head2_tail2 = _seq_memo[key2]
        else:
            a2, b2, head2, tail2 = balanced_decomp_unsafe(seq2, open_to_close)
            head2_tail2 = head2 + tail2
            _seq_memo[key2] = a2, b2, head2, tail2, head2_tail2

        # Case 2: The current edge in sequence1 is deleted
        best, val = _lcs_recurse(head1_tail1, seq2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

        # Case 3: The current edge in sequence2 is deleted
        cand, val_alt = _lcs_recurse(seq1, head2_tail2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
        if val_alt > val:
            best = cand
            val = val_alt

        # Case 1: The LCS involves this edge
        t1 = open_to_tok[a1[0]]
        t2 = open_to_tok[a2[0]]
        # if node_affinity(a1[0], a2[0]):
        affinity = node_affinity(t1, t2)
        if affinity:
            new_heads, pval_h = _lcs_recurse(head1, head2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)
            new_tails, pval_t = _lcs_recurse(tail1, tail2, open_to_close, node_affinity, open_to_tok, _memo, _seq_memo)

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


class UnbalancedException(Exception):
    pass


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


def tree_to_balanced_sequence(tree, open_to_close=None, toks=None, mode='tuple', strhack=False):
    from collections import namedtuple
    Token = namedtuple('Token', ['action', 'value'])
    # mapping between opening and closing tokens
    sources = [n for n in tree.nodes if tree.in_degree[n] == 0]
    sequence = []

    if open_to_close is None:
        open_to_close = {}
    if toks is None:
        toks = {}

    if strhack:
        if mode == 'label':
            all_labels = {n['label'] for n in list(tree.nodes.values())}
            assert all(x == 1 for x in map(len, all_labels))

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
                    elif mode == 'label':
                        open_tok = tree.nodes[v]['label']
                        assert strhack
                        if open_tok == '{':
                            close_tok = '}'
                        if open_tok == '[':
                            close_tok = ']'
                        if open_tok == '(':
                            close_tok = ')'
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
    if strhack:
        sequence = ''.join(sequence)
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


def random_ordered_tree(n, seed=None, pool=None):
    import kwarray
    rng = kwarray.ensure_rng(seed, 'python')
    tree = nx.dfs_tree(nx.random_tree(n, seed=seed))
    otree = nx.OrderedDiGraph()
    otree.add_edges_from(tree.edges)
    if pool is not None:
        for node in otree.nodes:
            otree.nodes[node]['label'] = rng.choice(pool)
    return otree


def generate_balance_unsafe(sequence, open_to_close):
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


def balanced_decomp_unsafe(sequence, open_to_close):
    """
    Example:
        >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
        >>> sequence = '({[[]]})[[][]]'
        >>> print('sequence = {!r}'.format(sequence))
        >>> a1, b1, head, tail = balanced_decomp(sequence, open_to_close)
        >>> print('a1 = {!r}'.format(a1))
        >>> print('tail = {!r}'.format(tail))
        >>> print('head = {!r}'.format(head))
        >>> a2, b2, tail1, tail2 = balanced_decomp(tail, open_to_close)
        >>> print('a2 = {!r}'.format(a2))
        >>> print('tail1 = {!r}'.format(tail1))
        >>> print('tail2 = {!r}'.format(tail2))
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


def generate_balance(sequence, open_to_close):
    """
    Safe version

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
    """
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


__notes_ = """

                # if 0:
                #     tuples = [(i + 1, i + 2, i + 3,) for i in range(4)]
                #     import timerit

                #     ti = timerit.Timerit(100, bestof=10, verbose=2)
                #     import itertools as it
                #     for timer in ti.reset('time'):
                #         with timer:
                #             tuple(it.chain.from_iterable(tuples))
                #     for timer in ti.reset('time'):
                #         with timer:
                #             res = tuples[0]
                #             for a in tuples[1:]:
                #                 res = res + a

"""
