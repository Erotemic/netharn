"""
cythonize -a -i ~/code/netharn/netharn/initializers/_nx_extensions_cython_backend.pyx

        >>> from netharn.initializers import _nx_extensions_cython_backend 
        >>> import timerit
        >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
        >>> for timer in ti.reset('time'):
        >>>     with timer:
        >>>         list(_nx_extensions_cython_backend.generate_balance_unsafe_cython(sequence, open_to_close))

"""

def generate_balance_unsafe_cython(sequence, open_to_close):
    cdef tuple item
    cdef bint flag
    cdef int stacklen = 0
    for token in sequence:
        if token in open_to_close:
            stacklen += 1
        else:
            stacklen -= 1
        flag = stacklen == 0
        item = (flag, token)
        yield item


def balanced_decomp_unsafe2_cython(tuple sequence, dict open_to_close):
    cdef int stacklen = 1  # always +1 in the first iteration
    cdef int head_stop = 1

    tok_curr = sequence[0]
    want_close = open_to_close[tok_curr]

    # for tok_curr in sequence[1:]:
    for head_stop in range(1, len(sequence)):
        tok_curr = sequence[head_stop]
        stacklen += 1 if tok_curr in open_to_close else -1
        if stacklen == 0 and tok_curr == want_close:
            pop_close = sequence[head_stop:head_stop + 1]
            break

    pop_open = sequence[0:1]
    head = sequence[1:head_stop]
    tail = sequence[head_stop + 1:]
    return pop_open, pop_close, head, tail
        
