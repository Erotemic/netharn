from netharn.initializers._nx_ext.balanced_sequence import UnbalancedException, IdentityDict  # NOQA
from netharn.initializers._nx_ext.balanced_sequence import generate_all_decomp, _cython_lcs_backend, _lcs_iter_simple_alt2, _lcs_iter_prehash2, _lcs_recurse, _lcs_iter_simple, _lcs_iter_simple_alt1, _lcs_iter_prehash  # NOQA


def _lcs_iter_simple_alt3(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node):
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
    >>> res = _lcs_iter_simple_alt3(full_seq1, full_seq2, open_to_close, node_affinity, open_to_node)
    >>> embeddings, val, delseq = res
    >>> print('embeddings = {!r}'.format(embeddings[0]))
    >>> print('delseq = {!r}'.format(delseq[0]))
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
    base_result = (0, best, ([], []))
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
                x, y, z = cand1
                z1, z2 = z
                z1 = z1 + [a1]
                z2 = z2 + [a2]
                z3 = (z1, z2)
                cand1 = (x, y, z3)
            else:
                # stack.append(key)
                stack.append(try_key)
                continue

            # Case 3: The current edge in sequence2 is deleted
            try_key = (seq1, head_tail2)
            if try_key in _results:
                cand2 = _results[try_key]
                x, y, z = cand2
                z1, z2 = z
                z1 = z1 + [a1]
                z2 = z2 + [a2]
                z3 = (z1, z2)
                cand2 = (x, y, z3)
            else:
                # stack.append(key)
                stack.append(try_key)
                continue

            # Case 1: The LCS involves this edge
            affinity = node_affinity(t1, t2)
            if affinity:
                try_key = (head1, head2)
                if try_key in _results:
                    pval_h, new_heads, delseq_h = _results[try_key]
                else:
                    # stack.append(key)
                    stack.append(try_key)
                    continue

                try_key = (tail1, tail2)
                if try_key in _results:
                    pval_t, new_tails, delseq_t = _results[try_key]
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

                h1, h2 = delseq_h
                t1, t2 = delseq_t

                delseq3 = (h1 + t1, h2 + t2)
                cand3 = (val3, res3, delseq3)
            else:
                cand3 = (-1, None)

            # We solved the frame
            _results[key] = max(cand1, cand2, cand3)
        stack.pop()

    val, best, delseq = _results[key0]
    found = (best, val, delseq)
    return found


def balanced_decomp2(sequence, open_to_close, start=0):
    gen = generate_balance2(sequence, open_to_close)
    for tup in gen:
        (bal_curr, tok_curr, idx1, idx2) = tup
        if idx2 == start:
            stop = idx1
            assert bal_curr
            break

    return start, stop
    # pop_open = sequence[0:1]
    # pop_close = sequence[head_stop:head_stop + 1]
    # head = sequence[1:head_stop]
    # tail = sequence[head_stop + 1:]
    # head_tail = head + tail
    # return pop_open, pop_close, head, tail, head_tail


def generate_balance2(sequence, open_to_close, start=0):
    """
    Alternate version that also returns index information

    Yields
    ------
    bool, T, int, int
        is balanced
        opening token
        opening token index
        current token index


    Example
    -------
    >>> open_to_close = {0: 1}
    >>> seq = sequence = [0, 0, 0, 1, 1, 1, 0, 1]
    >>> gen = list(generate_balance2(sequence, open_to_close))
    >>> for flag, token, idx1, idx2 in gen:
    >>>     print('flag={:d}, token={}, {}, {}'.format(flag, token, idx1, idx2))

    balanced_decomp2(sequence, open_to_close)
    """
    stack = []
    # Traversing the Expression
    for curr_idx, token in enumerate(sequence, start=start):

        if token in open_to_close:
            # Push opening elements onto the stack
            stack.append((token, curr_idx))
            open_idx = -1
        else:
            # Check that closing elements
            if not stack:
                raise UnbalancedException
            prev_open, open_idx = stack.pop()
            want_close = open_to_close[prev_open]

            if token != want_close:
                raise UnbalancedException

        # If the stack is empty the sequence is currently balanced
        currently_balanced = not bool(stack)
        yield currently_balanced, token, curr_idx, open_idx

    if stack:
        raise UnbalancedException


def generate_all_decomp2(full_seq, open_to_close, open_to_node=None):
    """
    Alternate version where we keep track of indices instead

    Example
    -------
    >>> full_seq = '0010010010111101'
    >>> open_to_close = {'0': '1'}
    >>> full_seq = '{[{}]}[()]'
    >>> open_to_close = {'[': ']', '{': '}', '(': ')'}
    >>> list(generate_balance2(full_seq, open_to_close))
    >>> all_decomp = generate_all_decomp2(full_seq, open_to_close)

    >>> from netharn.initializers._nx_ext import demodata
    >>> full_seq, open_to_close = demodata.random_balanced_sequence(5, mode='number')
    >>> all_decomp = generate_all_decomp2(full_seq, open_to_close)
    """
    if open_to_node is None:
        open_to_node = IdentityDict()
    all_decomp = {}

    start = 0
    stop = len(full_seq)
    deleted = []
    stack = [
        ('f', full_seq, start, stop, deleted)
    ]

    DEBUG = 1

    while stack:
        t, seq, seq_start, seq_stop, seq_del = stack.pop()
        if DEBUG:
            import ubelt as ub
            print('-----')
            print(list(full_seq))

            isdel = ['X' if b else ' ' for b in ub.boolmask(seq_del, len(full_seq))]
            sep = ' :  '
            pos = list(' ' * len(full_seq))
            pos[seq_start] = 'S'
            pos[seq_stop - 1] = 'T'
            prefix = ': '
            def padjoin(s):
                return sep.join(['{:>2}'.format(c) for c in s])
            print(prefix + padjoin(range(len(full_seq))))
            print(prefix + padjoin(full_seq) + ' <- full_seq')
            print(prefix + padjoin(isdel) + ' <- seq_del')
            print(prefix + padjoin(pos) + ' <- seq_start, seq_stop')

            val = seq_start, seq_stop, seq_del
            print('seq = {}, {!r}, {}'.format(t, seq, val))
            base = full_seq[seq_start:seq_stop]
            print('base = {!r}'.format(base))
            rel_pad_del = [idx - seq_start for idx in seq_del if idx >= seq_start]
            keep_idxs = sorted(set(range(len(base))) - set(rel_pad_del))
            newlist = [base[idx] for idx in keep_idxs]
            try:
                recon = ''.join(newlist)
            except TypeError:
                recon = tuple(newlist)
            print('recon = {!r}'.format(recon))
        if seq:
            rel_start, rel_stop = balanced_decomp2(seq, open_to_close)

            rel_head_start = rel_start + 1
            rel_head_stop = rel_stop
            rel_tail_start = rel_stop + 1
            rel_tail_stop = len(seq)
            if DEBUG > 1:
                print('rel_start = {!r}'.format(rel_start))
                print('rel_stop = {!r}'.format(rel_stop))
                print('rel_head_start = {!r}'.format(rel_head_start))
                print('rel_head_stop = {!r}'.format(rel_head_stop))
                print('rel_tail_start = {!r}'.format(rel_tail_start))
                print('rel_tail_stop = {!r}'.format(rel_tail_stop))

            rel_pad_del = [idx - seq_start for idx in seq_del if seq_start <= idx <= seq_stop]
            if DEBUG:
                print('rel_pad_del = {!r}'.format(rel_pad_del))

            # I think there is a cumsum way of doing this, I'm being dense atm
            # seq = '3' * 10
            # rel_pad_del = [4, 5, 9, 11]
            hack_map = list(range(1 + len(seq) + len(rel_pad_del)))
            for idx in sorted(rel_pad_del, reverse=True):
                del hack_map[idx]

            if DEBUG:
                print('hack_map = {!r}'.format(hack_map))

            # I believe it is the case that the deleted indexes will only be
            # able to cause a shift in the abs_tail_stop, the abs_tail_start,
            # abs_head_stop, and abs_head_start should never "conflict" with
            # the deleted indexes (I think).

            # num_del_after_tail_start = sum(abs_tail_start <= i <= seq_stop for i in seq_del)
            # print('num_del_after_tail_start = {!r}'.format(num_del_after_tail_start))
            # num_del_before_tail_start = sum(0 <= i <= rel_tail_stop for i in rel_pad_del)

            abs_head_start = hack_map[rel_head_start] + seq_start
            abs_head_stop = hack_map[rel_head_stop] + seq_start

            abs_tail_start = hack_map[rel_tail_start] + seq_start
            abs_tail_stop = hack_map[rel_tail_stop] + seq_start

            if DEBUG > 1:
                print('abs_head_start = {!r}'.format(abs_head_start))
                print('abs_head_stop = {!r}'.format(abs_head_stop))

                print('abs_tail_start = {!r}'.format(abs_tail_start))
                print('abs_tail_stop = {!r}'.format(abs_tail_stop))

            head_sl = slice(rel_head_start, rel_head_stop)
            tail_sl = slice(rel_tail_start, rel_tail_stop)

            head = seq[head_sl]
            tail = seq[tail_sl]
            head_tail = head + tail

            head_del = seq_del
            tail_del = seq_del

            if abs_head_stop == abs_head_start:
                # case where tail is empty (which head_tail doesnt matter
                # anyway but this is just a POC
                abs_head_tail_start = abs_tail_start
            else:
                abs_head_tail_start = abs_head_start

            if abs_tail_stop == abs_tail_start:
                # case where tail is empty (which head_tail doesnt matter
                # anyway but this is just a POC
                abs_head_tail_stop = abs_head_stop
            else:
                abs_head_tail_stop = abs_tail_stop

            abs_del_start = seq_start + rel_start
            abs_del_stop = seq_start + rel_stop

            # head_tail_del = [abs_del_start, abs_del_stop] + seq_del
            assert abs_del_start < abs_head_tail_start
            if abs_del_stop < abs_head_tail_stop:
                head_tail_del = [abs_del_stop] + seq_del
            else:
                head_tail_del = seq_del

            # seq[head_sl] + seq[tail_sl]

            # pop_open, pop_close, head, tail, head_tail = balanced_decomp2(seq, open_to_close)
            # node = open_to_node[pop_open[0]]
            all_decomp[seq] = (seq_start, seq_stop, seq_del)
            # (node, pop_open, pop_close, head, tail, head_tail)

            if abs_head_stop > len(full_seq):
                raise AssertionError
            if abs_tail_stop > len(full_seq):
                raise AssertionError
            if abs_head_tail_stop > len(full_seq):
                raise AssertionError

            if head:
                if DEBUG:
                    print('head = {!r}'.format(head))
                head_del = [i for i in head_del if abs_head_start <= i < abs_head_stop]
                stack.append(('h', head, abs_head_start, abs_head_stop, head_del))
            if tail:
                if DEBUG:
                    print('tail = {!r}'.format(tail))
                tail_del = [i for i in tail_del if abs_tail_start <= i < abs_tail_stop]
                stack.append(('t', tail, abs_tail_start, abs_tail_stop, tail_del))
            if tail and head:
                if DEBUG:
                    print('head_tail = {!r}'.format(head_tail))
                    print('head_tail_del = {!r}'.format(head_tail_del))
                head_tail_del = [i for i in head_tail_del if abs_head_tail_start <= i < abs_head_tail_stop]
                stack.append(('ht', head_tail, abs_head_tail_start, abs_head_tail_stop, head_tail_del))
            if DEBUG:
                assert seq == recon

    return all_decomp
