# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import random
import itertools as it
import ubelt as ub  # NOQA
import torch

_SEED_MAX = (2 ** 32 - 1)


def seed_global(seed, offset=0):
    """
    Seeds the python, numpy, and torch global random states

    Args:
        seed (int): seed to use
        offset (int, optional): if specified, uses a different seed for each
            global random state separated by this offset.
    """
    random.seed((seed) % _SEED_MAX)
    np.random.seed((seed + offset) % _SEED_MAX)
    torch.random.manual_seed((seed + 2 * offset) % _SEED_MAX)
    torch.cuda.manual_seed_all((seed + 3 * offset) % _SEED_MAX)


def shuffle(items, rng=None):
    """
    Shuffles a list inplace and then returns it for convinience

    Args:
        items (list or ndarray): list to shuffl
        rng (RandomState or int): seed or random number gen

    Returns:
        list: this is the input, but returned for convinience

    Example:
        >>> list1 = [1, 2, 3, 4, 5, 6]
        >>> list2 = shuffle(list(list1), rng=1)
        >>> assert list1 != list2
        >>> result = str(list2)
        >>> print(result)
        [3, 2, 5, 1, 4, 6]
    """
    rng = ensure_rng(rng)
    rng.shuffle(items)
    return items


def random_combinations(items, size, num=None, rng=None):
    """
    Yields `num` combinations of length `size` from items in random order

    Args:
        items (?):
        size (?):
        num (None): (default = None)
        rng (RandomState):  random number generator(default = None)

    Yields:
        tuple: combo

    Example:
        >>> items = list(range(10))
        >>> size = 3
        >>> num = 5
        >>> rng = 0
        >>> combos = list(random_combinations(items, size, num, rng))
        >>> result = ('combos = %s' % (ub.repr2(combos),))
        >>> print(result)

    Example:
        >>> items = list(zip(range(10), range(10)))
        >>> size = 3
        >>> num = 5
        >>> rng = 0
        >>> combos = list(random_combinations(items, size, num, rng))
        >>> result = ('combos = %s' % (ub.repr2(combos),))
        >>> print(result)
    """
    import scipy.misc
    import numpy as np
    rng = ensure_rng(rng, api='python')
    num_ = np.inf if num is None else num
    # Ensure we dont request more than is possible
    n_max = int(scipy.misc.comb(len(items), size))
    num_ = min(n_max, num_)
    if num is not None and num_ > n_max // 2:
        # If num is too big just generate all combinations and shuffle them
        combos = list(it.combinations(items, size))
        rng.shuffle(combos)
        for combo in combos[:num]:
            yield combo
    else:
        # Otherwise yield randomly until we get something we havent seen
        items = list(items)
        combos = set()
        while len(combos) < num_:
            # combo = tuple(sorted(rng.choice(items, size, replace=False)))
            combo = tuple(sorted(rng.sample(items, size)))
            if combo not in combos:
                # TODO: store indices instead of combo values
                combos.add(combo)
                yield combo


def random_product(items, num=None, rng=None):
    """
    Yields `num` items from the cartesian product of items in a random order.

    Args:
        items (list of sequences): items to get caresian product of
            packed in a list or tuple.
            (note this deviates from api of it.product)

    Example:
        >>> items = [(1, 2, 3), (4, 5, 6, 7)]
        >>> rng = 0
        >>> list(random_product(items, rng=0))
        >>> list(random_product(items, num=3, rng=0))
    """
    rng = ensure_rng(rng, 'python')
    seen = set()
    items = [list(g) for g in items]
    max_num = np.prod(np.array(list(map(len, items))))
    if num is None:
        num = max_num
    if num > max_num:
        raise ValueError('num exceedes maximum number of products')

    # TODO: make this more efficient when num is large
    if num > max_num // 2:
        for prod in shuffle(list(it.product(*items)), rng=rng):
            yield prod
    else:
        while len(seen) < num:
            # combo = tuple(sorted(rng.choice(items, size, replace=False)))
            idxs = tuple(rng.randint(0, len(g) - 1) for g in items)
            if idxs not in seen:
                seen.add(idxs)
                prod = tuple(g[x] for g, x in zip(items, idxs))
                yield prod


def _npstate_to_pystate(npstate):
    """
    Convert state of a NumPy RandomState object to a state
    that can be used by Python's Random.

    References:
        https://stackoverflow.com/questions/44313620/converting-randomstate

    Example:
        >>> py_rng = random.Random(0)
        >>> np_rng = np.random.RandomState(seed=0)
        >>> npstate = np_rng.get_state()
        >>> pystate = _npstate_to_pystate(npstate)
        >>> py_rng.setstate(pystate)
        >>> assert np_rng.rand() == py_rng.random()
    """
    PY_VERSION = 3
    version, keys, pos, has_gauss, cached_gaussian_ = npstate
    keys_pos = tuple(map(int, keys)) + (int(pos),)
    cached_gaussian_ = cached_gaussian_ if has_gauss else None
    pystate = (PY_VERSION, keys_pos, cached_gaussian_)
    return pystate


def _pystate_to_npstate(pystate):
    """
    Convert state of a Python Random object to state usable
    by NumPy RandomState.

    References:
        https://stackoverflow.com/questions/44313620/converting-randomstate

    Example:
        >>> py_rng = random.Random(0)
        >>> np_rng = np.random.RandomState(seed=0)
        >>> pystate = py_rng.getstate()
        >>> npstate = _pystate_to_npstate(pystate)
        >>> np_rng.set_state(npstate)
        >>> assert np_rng.rand() == py_rng.random()
    """
    NP_VERSION = 'MT19937'
    version, keys_pos_, cached_gaussian_ = pystate
    keys, pos = keys_pos_[:-1], keys_pos_[-1]
    keys = np.array(keys, dtype=np.uint32)
    has_gauss = cached_gaussian_ is not None
    cached_gaussian = cached_gaussian_ if has_gauss else 0.0
    npstate = (NP_VERSION, keys, pos, has_gauss, cached_gaussian)
    return npstate


def ensure_rng(rng, api='numpy'):
    """
    Returns a random number generator

    Args:
        seed: if None, then deafults to the global rng.
            Otherwise the seed can be an integer or a RandomState class

    Example:
        >>> rng = ensure_rng(None)
        >>> ensure_rng(0).randint(0, 1000)
        684
        >>> ensure_rng(np.random.RandomState(1)).randint(0, 1000)
        37

    Example:
        >>> num = 4
        >>> print('--- Python as PYTHON ---')
        >>> py_rng = random.Random(0)
        >>> pp_nums = [py_rng.random() for _ in range(num)]
        >>> print(pp_nums)
        >>> print('--- Numpy as PYTHON ---')
        >>> np_rng = ensure_rng(random.Random(0), api='numpy')
        >>> np_nums = [np_rng.rand() for _ in range(num)]
        >>> print(np_nums)
        >>> print('--- Numpy as NUMPY---')
        >>> np_rng = np.random.RandomState(seed=0)
        >>> nn_nums = [np_rng.rand() for _ in range(num)]
        >>> print(nn_nums)
        >>> print('--- Python as NUMPY---')
        >>> py_rng = ensure_rng(np.random.RandomState(seed=0), api='python')
        >>> pn_nums = [py_rng.random() for _ in range(num)]
        >>> print(pn_nums)
        >>> assert np_nums == pp_nums
        >>> assert pn_nums == nn_nums

    Ignore:
        >>> np.random.seed(0)
        >>> np.random.randint(0, 10000)
        2732
        >>> np.random.seed(0)
        >>> np.random.mtrand._rand.randint(0, 10000)
        2732
        >>> np.random.seed(0)
        >>> nh.util.ensure_rng(None).randint(0, 10000)
        2732
        >>> np.random.randint(0, 10000)
        9845
        >>> nh.util.ensure_rng(None).randint(0, 10000)
        3264
    """
    if api == 'numpy':
        if rng is None:
            # Dont do this because it seeds using dev/urandom
            # rng = np.random.RandomState(seed=None)
            # This is the underlying random state of the np.random module
            rng = np.random.mtrand._rand
        elif isinstance(rng, int):
            rng = np.random.RandomState(seed=rng % _SEED_MAX)
        elif isinstance(rng, random.Random):
            # Convert python to numpy random state
            py_rng = rng
            pystate = py_rng.getstate()
            npstate = _pystate_to_npstate(pystate)
            rng = np_rng = np.random.RandomState(seed=0)
            np_rng.set_state(npstate)
    elif api == 'python':
        if rng is None:
            rng = random
        elif isinstance(rng, int):
            rng = random.Random(rng % _SEED_MAX)
        elif isinstance(rng, np.random.RandomState):
            # Convert numpy to python random state
            np_rng = rng
            npstate = np_rng.get_state()
            pystate = _npstate_to_pystate(npstate)
            rng = py_rng = random.Random(0)
            py_rng.setstate(pystate)
    else:
        raise KeyError('unknown rng api={}'.format(api))
    return rng


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_random all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
