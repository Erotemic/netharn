import numpy as np

_SEED_MAX = (2 ** 32 - 1)


def ensure_rng(seed):
    """
    Creates a random number generator.

    Args:
        seed: if None, then the rng is unseeded. Otherwise the seed can be an
            integer or a RandomState class

    Example:
        >>> rng = ensure_rng(None)
        >>> ensure_rng(0).randint(0, 1000)
        684
        >>> ensure_rng(np.random.RandomState(1)).randint(0, 1000)
        37
    """
    if seed is None:
        rng = np.random.RandomState()
    elif isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed % _SEED_MAX)
    return rng


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_random all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
