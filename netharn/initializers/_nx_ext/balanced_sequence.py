"""
Balanced sequences are used via reduction to solve the maximum common subtree
embedding problem.
"""
import operator


def longest_common_balanced_sequence(
        seq1, seq2, open_to_close, open_to_node=None,
        node_affinity='auto', impl='iter-prehash2'):
    """
    Finds the longest common balanced sequence between two sequences

    Parameters
    ----------
    seq1, seq2: Iterable
        two input balanced sequences

    open_to_close : Dict
        a mapping from opening to closing tokens in the balanced sequence

    open_to_node : Dict | None
        a dictionary that maps a sequence token to a token corresponding to an
        original problem (e.g. a tree node), if unspecified an identity mapping
        is assumed. FIXME: see outstanding issues.
        WILL LIKELY CHANGE IN THE FUTURE

    node_affinity : None | str | callable
        Function for to determine if two nodes can be matched. The return is
        interpreted as a weight that is used to break ties. If None then any
        node can match any other node and only the topology is important.
        The default is "eq", which is the same as ``operator.eq``.

    impl : str
        Determines the backend implementation. There are currently 8 different
        backend implementations:

        recurse, iter, iter-prehash, iter-prehash2, iter-alt1, iter-alt2,
        iter-alt2-cython, and iter-prehash2-cython.

    Example
    -------
    >>> # extremely simple case
    >>> seq1 = '[][[]][]'
    >>> seq2 = '[[]][[]]'
    >>> open_to_close = {'[': ']'}
    >>> best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close)
    >>> subseq1, subseq2 = best
    >>> print('subseq1 = {!r}'.format(subseq1))
    subseq1 = '[][[]]'

    >>> # 1-label case from the paper (see Example 5)
    >>> # https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf
    >>> seq1 = '0010010010111100001011011011'
    >>> seq2 = '001000101101110001000100101110111011'
    >>> open_to_close = {'0': '1'}
    >>> best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close)
    >>> subseq1, subseq2 = best
    >>> print('subseq1 = {!r}'.format(subseq1))
    >>> assert value == 13
    subseq1 = '00100101011100001011011011'

    >>> # 3-label case
    >>> seq1 = '{({})([[]([]){(()(({()[]({}{})}))){}}])}'
    >>> seq2 = '{[({{}}{{[][{}]}(()[(({()})){[]()}])})]}'
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close)
    >>> subseq1, subseq2 = best
    >>> print('subseq1 = {!r}'.format(subseq1))
    >>> assert value == 10
    subseq1 = '{{}[][]()(({()})){}}'
    """
    if node_affinity == 'auto' or node_affinity == 'eq':
        node_affinity = operator.eq
    if node_affinity is None:
        def _matchany(a, b):
            return True
        node_affinity = _matchany
    if open_to_node is None:
        open_to_node = IdentityDict()
    full_seq1 = seq1
    full_seq2 = seq2
    if impl == 'auto':
        if _cython_lcs_backend():
            impl = 'iter-alt2-cython'
        else:
            impl = 'iter-alt2'

    if impl == 'recurse':
        _memo = {}
        _seq_memo = {}
        best, value = _lcs_recurse(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node,
            _memo, _seq_memo)
    elif impl == 'iter':
        best, value = _lcs_iter_simple(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-prehash':
        best, value = _lcs_iter_prehash(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-prehash2':
        best, value = _lcs_iter_prehash2(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-alt1':
        best, value = _lcs_iter_simple_alt1(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-alt2':
        best, value = _lcs_iter_simple_alt2(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-alt2-cython':
        balanced_sequence_cython = _cython_lcs_backend(error='raise')
        best, value = balanced_sequence_cython._lcs_iter_simple_alt2_cython(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    elif impl == 'iter-prehash2-cython':
        balanced_sequence_cython = _cython_lcs_backend(error='raise')
        best, value = balanced_sequence_cython._lcs_iter_prehash2_cython(
            full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    else:
        raise KeyError(impl)
    return best, value


def available_impls_longest_common_balanced_sequence():
    """
    Returns all available implementations for
    :func:`longest_common_balanced_sequence`.
    """
    from netharn.initializers._nx_ext import balanced_sequence
    impls = []
    if balanced_sequence._cython_lcs_backend():
        impls += [
            'iter-alt2-cython',
            'iter-prehash2-cython',
        ]

    # Pure python backends
    impls += [
        'iter-prehash2',
        'iter-alt2',
        'iter-alt1',
        'iter-prehash',
        'iter',
        'recurse',
    ]
    return impls


def _cython_lcs_backend(error='ignore'):
    """
    Returns the cython backend if available, otherwise None
    """
    try:
        from netharn.initializers._nx_ext import balanced_sequence_cython
    except Exception:
        if error == 'ignore':
            return None
        elif error == 'raise':
            raise
        else:
            raise KeyError(error)
    else:
        return balanced_sequence_cython


def _lcs_iter_simple_alt2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Depth first stack trajectory and replace try except statements with ifs

    This is the current best pure-python algorithm candidate

    >>> full_seq1 = '{({})([[]([]){(()(({()[]({}{})}))){}}])}'
    >>> full_seq2 = '{[({{}}{{[][{}]}(()[(({()})){[]()}])})]}'
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> full_seq1 = '[][[]][]'
    >>> full_seq2 = '[[]][[]]'
    >>> open_to_close = {'[': ']'}
    >>> import operator as op
    >>> node_affinity = op.eq
    >>> open_to_node = IdentityDict()
    >>> res = _lcs_iter_simple_alt2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    >>> val, embeddings = res
    """
    all_decomp1 = generate_all_decomp(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp(full_seq2, open_to_close, open_to_node)

    key0 = (full_seq1, full_seq2)
    frame0 = key0
    stack = [frame0]

    # Memoize mapping (seq1, seq2) -> best size, embeddings, deleted edges
    _results = {}

    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
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


def _lcs_iter_prehash2(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Version of the lcs iterative algorithm where we precompute hash values

    See :func:`longest_common_balanced_sequence` for parameter details.
    """

    all_decomp1 = generate_all_decomp_prehash(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp_prehash(full_seq2, open_to_close, open_to_node)

    key_decomp1 = {}
    key_decomp2 = {}
    _results = {}
    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
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
        frame = stack[-1]
        key, seq1, seq2 = frame
        seq1_key, seq2_key = key
        if key not in _results:
            missing_frames.clear()

            info1 = key_decomp1[seq1_key]
            tok1, seq1, head1, tail1, head_tail1, head1_key, tail1_key, head_tail1_key, a1, b1 = info1

            # if seq2_key not in key_decomp2:
            info2 = key_decomp2[seq2_key]
            tok2, seq2, head2, tail2, head_tail2, head2_key, tail2_key, head_tail2_key, a2, b2 = info2

            affinity = node_affinity(tok1, tok2)

            # Case 2: The current edge in sequence1 is deleted
            try_key = (head_tail1_key, seq2_key)
            if try_key in _results:
                cand1 = _results[try_key]
            else:
                miss_frame = try_key, head_tail1, seq2
                stack.append(miss_frame)
                continue

            # Case 3: The current edge in sequence2 is deleted
            try_key = (seq1_key, head_tail2_key)
            if try_key in _results:
                cand2 = _results[try_key]
            else:
                miss_frame = try_key, seq1, head_tail2
                stack.append(miss_frame)
                continue

            # Case 1: The LCS involves this edge
            if affinity:
                try_key = (head1_key, head2_key)
                if try_key in _results:
                    pval_h, new_heads = _results[try_key]
                else:
                    miss_frame = try_key, head1, head2
                    stack.append(miss_frame)
                    continue

                try_key = (tail1_key, tail2_key)
                if try_key in _results:
                    pval_t, new_tails = _results[try_key]
                else:
                    miss_frame = try_key, tail1, tail2
                    stack.append(miss_frame)
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

    # The stack pop is our solution
    (val, best) = _results[key0]
    found = (best, val)
    return found


def _lcs_recurse(seq1, seq2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo):
    """
    Surprisingly, this recursive implementation is one of the faster
    pure-python methods for certain input types. However, its major drawback is
    that it can raise a RecurssionError if the inputs are too deep.
    """
    if not seq1:
        return (seq1, seq1), 0
    elif not seq2:
        return (seq2, seq2), 0
    else:
        key1 = hash(seq1)  # using hash(seq) is faster than seq itself
        key2 = hash(seq2)
        key = hash((key1, key2))
        if key in _memo:
            return _memo[key]

        if key1 in _seq_memo:
            a1, b1, head1, tail1, head1_tail1 = _seq_memo[key1]
        else:
            a1, b1, head1, tail1, head1_tail1 = balanced_decomp_unsafe(seq1, open_to_close)
            _seq_memo[key1] = a1, b1, head1, tail1, head1_tail1

        if key2 in _seq_memo:
            a2, b2, head2, tail2, head2_tail2 = _seq_memo[key2]
        else:
            a2, b2, head2, tail2, head2_tail2 = balanced_decomp_unsafe(seq2, open_to_close)
            _seq_memo[key2] = a2, b2, head2, tail2, head2_tail2

        # Case 2: The current edge in sequence1 is deleted
        best, val = _lcs_recurse(head1_tail1, seq2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)

        # Case 3: The current edge in sequence2 is deleted
        cand, val_alt = _lcs_recurse(seq1, head2_tail2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)
        if val_alt > val:
            best = cand
            val = val_alt

        # Case 1: The LCS involves this edge
        t1 = open_to_node[a1[0]]
        t2 = open_to_node[a2[0]]
        affinity = node_affinity(t1, t2)
        if affinity:
            new_heads, pval_h = _lcs_recurse(head1, head2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)
            new_tails, pval_t = _lcs_recurse(tail1, tail2, open_to_close, node_affinity, open_to_node, _memo, _seq_memo)

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


def _lcs_iter_simple(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Converts _lcs_recursive to an iterative algorithm using a fairly
    straightforward method that effectivly simulates callstacks.
    Uses a breadth-first trajectory and try-except to catch missing
    memoized results (which seems to be slightly slower than if statements).
    """
    all_decomp1 = generate_all_decomp(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp(full_seq2, open_to_close, open_to_node)

    args0 = (full_seq1, full_seq2)
    frame0 = args0
    stack = [frame0]

    _results = {}
    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
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

            t1, a1, b1, head1, tail1, head_tail1 = all_decomp1[seq1]
            t2, a2, b2, head2, tail2, head_tail2 = all_decomp2[seq2]

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


def _lcs_iter_simple_alt1(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Depth first stack trajectory
    """
    all_decomp1 = generate_all_decomp(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp(full_seq2, open_to_close, open_to_node)

    args0 = (full_seq1, full_seq2)
    frame0 = args0
    stack = [frame0]

    _results = {}
    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
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


def _lcs_iter_prehash(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
    """
    Version of the lcs iterative algorithm where we precompute hash values.
    Uses a breadth-first trajectory.
    """
    all_decomp1 = generate_all_decomp_prehash(full_seq1, open_to_close, open_to_node)
    all_decomp2 = generate_all_decomp_prehash(full_seq2, open_to_close, open_to_node)

    key_decomp1 = {}
    key_decomp2 = {}
    _results = {}
    # Populate base cases
    empty1 = type(next(iter(all_decomp1.keys())))()
    empty2 = type(next(iter(all_decomp2.keys())))()
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
                info1 = balanced_decomp_prehash(seq1, open_to_close)
                key_decomp1[seq1_key] = info1
            tok1, seq1, head1, tail1, head_tail1, head1_key, tail1_key, head_tail1_key, a1, b1 = info1

            try:
                info2 = key_decomp2[seq2_key]
            except KeyError:
                info2 = balanced_decomp_prehash(seq2, open_to_close)
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


class UnbalancedException(Exception):
    """
    Denotes that a sequence was unbalanced
    """
    pass


class IdentityDict:
    """
    Used when ``open_to_node`` is unspecified
    """
    def __getitem__(self, key):
        return key


def generate_all_decomp(seq, open_to_close, open_to_node=None):
    """
    Generates all decompositions of a single balanced sequence by
    recursive decomposition of the head, tail, and head|tail.

    Parameters
    ----------
    seq : Tuple | str
        a tuple of hashable items or a string where each character is an item

    open_to_close : Dict
        a dictionary that maps opening tokens to closing tokens in the balanced
        sequence problem.

    open_to_node : Dict
        a dictionary that maps a sequence token to a token corresponding to an
        original problem (e.g. a tree node)

    Returns
    -------
    Dict : mapping from a sub-sequence to its decomposition

    Notes
    -----
    In the paper: See Definition 2, 4, Lemma, 1, 2, 3, 4.

    Example
    -------
    >>> # Example 2 in the paper (one from each column)
    >>> seq = '00100100101111'
    >>> open_to_close = {'0': '1'}
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    >>> assert len(all_decomp) == len(seq) // 2
    >>> import pprint
    >>> pprint.pprint(all_decomp)
    {'00100100101111': ('0', '0', '1', '010010010111', '', '010010010111'),
     '0010010111': ('0', '0', '1', '01001011', '', '01001011'),
     '001011': ('0', '0', '1', '0101', '', '0101'),
     '01': ('0', '0', '1', '', '', ''),
     '010010010111': ('0', '0', '1', '', '0010010111', '0010010111'),
     '01001011': ('0', '0', '1', '', '001011', '001011'),
     '0101': ('0', '0', '1', '', '01', '01')}

    Example
    -------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> seq = '({[[]]})[[][]]{{}}'
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    >>> node, *decomp = all_decomp[seq]
    >>> pop_open, pop_close, head, tail, head_tail = decomp
    >>> print('node = {!r}'.format(node))
    >>> print('pop_open = {!r}'.format(pop_open))
    >>> print('pop_close = {!r}'.format(pop_close))
    >>> print('head = {!r}'.format(head))
    >>> print('tail = {!r}'.format(tail))
    >>> print('head_tail = {!r}'.format(head_tail))
    node = '('
    pop_open = '('
    pop_close = ')'
    head = '{[[]]}'
    tail = '[[][]]{{}}'
    head_tail = '{[[]]}[[][]]{{}}'
    >>> decomp_alt = balanced_decomp(seq, open_to_close)
    >>> assert decomp_alt == tuple(decomp)

    Example
    -------
    >>> from netharn.initializers._nx_ext.demodata import random_balanced_sequence
    >>> seq, open_to_close = random_balanced_sequence(10)
    >>> all_decomp = generate_all_decomp(seq, open_to_close)
    """
    if open_to_node is None:
        open_to_node = IdentityDict()
    all_decomp = {}
    stack = [seq]
    while stack:
        seq = stack.pop()
        if seq not in all_decomp and seq:
            pop_open, pop_close, head, tail, head_tail = balanced_decomp(seq, open_to_close)
            node = open_to_node[pop_open[0]]
            all_decomp[seq] = (node, pop_open, pop_close, head, tail, head_tail)
            if head:
                if tail:
                    stack.append(head_tail)
                    stack.append(tail)
                stack.append(head)
            elif tail:
                stack.append(tail)
    return all_decomp


def balanced_decomp(sequence, open_to_close):
    """
    Generates a decomposition of a balanced sequence.

    Parameters
    ----------
    sequence : str
        balanced sequence to be decomposed

    open_to_close: dict
        a dictionary that maps opening tokens to closing tokens in the balanced
        sequence problem.

    Returns
    -------
    : tuple[T, T, T, T, T]
        where ``T = type(sequence)``
        Contents of this tuple are:

            0. a1 - a sequence of len(1) containing the current opening token
            1. b1 - a sequence of len(1) containing the current closing token
            2. head - head of the sequence
            3. tail - tail of the sequence
            4. head_tail - the concatanted head and tail

    Example
    -------
    >>> # Example 3 from the paper
    >>> sequence = '001000101101110001000100101110111011'
    >>> open_to_close = {'0': '1'}
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('head = {!r}'.format(head))
    >>> print('tail = {!r}'.format(tail))
    head = '010001011011'
    tail = '0001000100101110111011'

    Example
    -------
    >>> open_to_close = {0: 1}
    >>> sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('a1 = {!r}'.format(a1))
    >>> print('b1 = {!r}'.format(b1))
    >>> print('head = {!r}'.format(head))
    >>> print('tail = {!r}'.format(tail))
    >>> print('head_tail = {!r}'.format(head_tail))
    a1 = [0]
    b1 = [1]
    head = [0, 0, 1, 1]
    tail = [0, 1]
    head_tail = [0, 0, 1, 1, 0, 1]
    >>> a2, b2, tail1, tail2, head_tail2 = balanced_decomp(tail, open_to_close)

    Example
    -------
    >>> open_to_close = {'{': '}', '(': ')', '[': ']'}
    >>> sequence = '({[[]]})[[][]]'
    >>> a1, b1, head, tail, head_tail = balanced_decomp(sequence, open_to_close)
    >>> print('a1 = {!r}'.format(a1))
    >>> print('b1 = {!r}'.format(b1))
    >>> print('head = {!r}'.format(head))
    >>> print('tail = {!r}'.format(tail))
    >>> print('head_tail = {!r}'.format(head_tail))
    a1 = '('
    b1 = ')'
    head = '{[[]]}'
    tail = '[[][]]'
    head_tail = '{[[]]}[[][]]'
    >>> a2, b2, tail1, tail2, head_tail2 = balanced_decomp(tail, open_to_close)
    >>> print('a2 = {!r}'.format(a2))
    >>> print('b2 = {!r}'.format(b2))
    >>> print('tail1 = {!r}'.format(tail1))
    >>> print('tail2 = {!r}'.format(tail2))
    >>> print('head_tail2 = {!r}'.format(head_tail2))
    a2 = '['
    b2 = ']'
    tail1 = '[][]'
    tail2 = ''
    head_tail2 = '[][]'
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
    tail = sequence[head_stop + 1:]
    head_tail = head + tail
    return pop_open, pop_close, head, tail, head_tail


def generate_balance(sequence, open_to_close):
    """
    Iterates through a balanced sequence and reports if the sequence-so-far
    is balanced at that position or not.

    Parameters
    ----------
    sequence: List[Tuple] | str:
        an input balanced sequence

    open_to_close : Dict
        a mapping from opening to closing tokens in the balanced sequence

    Raises
    ------
    UnbalancedException - if the input sequence is not balanced

    Yields
    ------
    Tuple[bool, T]:
        boolean indicating if the sequence is balanced at this index,
        and the current token

    Example
    -------
    >>> open_to_close = {0: 1}
    >>> sequence = [0, 0, 0, 1, 1, 1]
    >>> gen = list(generate_balance(sequence, open_to_close))
    >>> for flag, token in gen:
    >>>     print('flag={:d}, token={}'.format(flag, token))

    Example
    -------
    >>> from netharn.initializers._nx_ext.demodata import random_balanced_sequence
    >>> sequence, open_to_close = random_balanced_sequence(4)
    >>> print('sequence = {!r}'.format(sequence))
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


def generate_all_decomp_prehash(seq, open_to_close, open_to_node):
    """
    Like :func:`generate_all_decomp` but additionally returns the
    precomputed hashes of the sequences.
    """
    all_decomp = {}
    stack = [seq]
    while stack:
        seq = stack.pop()
        if seq:
            # key = hash(seq)
            key = seq
            if key not in all_decomp:
                info = balanced_decomp_prehash(seq, open_to_close, open_to_node)
                head, tail, head_tail = info[2:5]
                all_decomp[key] = info
                stack.append(head_tail)
                stack.append(head)
                stack.append(tail)
    return all_decomp


def balanced_decomp_prehash(seq, open_to_close, open_to_node):
    """
    Like :func:`balanced_decomp` but additionally returns the
    precomputed hashes of the sequences.
    """
    pop_open, pop_close, head, tail, head_tail = balanced_decomp_unsafe(seq, open_to_close)
    head_key = hash(head)
    tail_key = hash(tail)
    head_tail_key = hash(head_tail)
    node = open_to_node[pop_open[0]]
    a = pop_open
    b = pop_close
    info = (node, seq, head, tail, head_tail, head_key, tail_key, head_tail_key, a, b)
    return info


def balanced_decomp_unsafe(sequence, open_to_close):
    """
    Same as :func:`balanced_decomp` but assumes that ``sequence`` is valid
    balanced sequence in order to execute faster.
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
    head_tail = head + tail
    return pop_open, pop_close, head, tail, head_tail


def generate_balance_unsafe(sequence, open_to_close):
    """
    Same as :func:`generate_balance` but assumes that ``sequence`` is valid
    balanced sequence in order to execute faster.
    """
    stacklen = 0
    for token in sequence:
        if token in open_to_close:
            stacklen += 1
        else:
            stacklen -= 1
        yield stacklen == 0, token
