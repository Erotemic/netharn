

def test_all_implementations_are_same():
    """
    Tests several random sequences
    """
    from netharn.initializers._nx_ext import balanced_sequence
    from netharn.initializers._nx_ext import demodata
    from networkx.utils import create_py_random_state

    seed = 93024896892223032652928827097264
    rng = create_py_random_state(seed)

    maxsize = 20
    num_trials = 5

    for _ in range(num_trials):
        n1 = rng.randint(1, maxsize)
        n2 = rng.randint(1, maxsize)

        seq1, open_to_close = demodata.random_balanced_sequence(n1, seed=rng)
        seq2, open_to_close = demodata.random_balanced_sequence(n2, open_to_close=open_to_close, seed=rng)
        longest_common_balanced_sequence = balanced_sequence.longest_common_balanced_sequence

        # Note: the returned sequences may be different (maximum embeddings may not
        # be unique), but the values should all be the same.
        results = {}
        impls = balanced_sequence.available_impls_longest_common_balanced_sequence()
        for impl in impls:
            best, val = longest_common_balanced_sequence(
                seq1, seq2, open_to_close, node_affinity=None, impl=impl)
            results[impl] = val
