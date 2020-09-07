"""
Helpers for creating random data for tests / benchmarks for the tree embedding
algorithms.
"""


def random_paths(
        size=10, max_depth=10, common=0, prefix_depth1=0, prefix_depth2=0,
        sep='/', labels=26, seed=None):
    """
    Returns two randomly created paths (as in directory structures) for use in
    testing and benchmarking :func:`maximum_common_path_embedding`.

    Parameters
    ----------
    size : int
        The number of independant random paths

    max_depth : int
        Maximum depth for the independant random paths

    common : int
        The number of shared common paths

    prefix_depth1: int
        Depth of the random prefix attacheded to first common paths

    prefix_depth2: int
        Depth of the random prefix attacheded to second common paths

    labels: int or collection
        Number of or collection of tokens that can be used as node labels

    sep: str
        path separator

    seed:
        Random state or seed

    Examples
    --------
    >>> paths1, paths2 = random_paths(
    >>>     size=5, max_depth=3, common=6,
    >>>     prefix_depth1=3, prefix_depth2=3, labels=2 ** 64,
    >>>     seed=0)
    >>> from netharn.initializers._nx_ext.path_embedding import paths_to_otree
    >>> from netharn.initializers._nx_ext.tree_embedding import tree_to_seq
    >>> tree = paths_to_otree(paths1)
    >>> seq, open_to_close, node_to_open = tree_to_seq(tree, mode='chr')
    >>> seq, open_to_close, node_to_open = tree_to_seq(tree, mode='number')
    >>> seq, open_to_close, node_to_open = tree_to_seq(tree, mode='tuple')
    >>> # xdoctest: +REQUIRES(module:ubelt)
    >>> import ubelt as ub
    >>> print('paths1 = {}'.format(ub.repr2(paths1, nl=1)))
    >>> print('paths2 = {}'.format(ub.repr2(paths2, nl=1)))
    """
    from networkx.utils import create_py_random_state
    rng = create_py_random_state(seed)

    if isinstance(labels, int):
        alphabet = list(map(chr, range(ord('a'), ord('z'))))

        def random_label():
            digit = rng.randint(0, labels)
            label = _convert_digit_base(digit, alphabet)
            return label
    else:
        from functools import partial
        random_label = partial(rng.choice, labels)

    def random_path(rng, max_depth):
        depth = rng.randint(1, max_depth)
        parts = [str(random_label()) for _ in range(depth)]
        path = sep.join(parts)
        return path

    # These paths might be shared (but usually not)
    iid_paths1 = {random_path(rng, max_depth) for _ in range(size)}
    iid_paths2 = {random_path(rng, max_depth) for _ in range(size)}

    # These paths will be shared
    common_paths = {random_path(rng, max_depth) for _ in range(common)}

    if prefix_depth1 > 0:
        prefix1 = random_path(rng, prefix_depth1)
        common1 = {sep.join([prefix1, suff]) for suff in common_paths}
    else:
        common1 = common_paths

    if prefix_depth2 > 0:
        prefix2 = random_path(rng, prefix_depth2)
        common2 = {sep.join([prefix2, suff]) for suff in common_paths}
    else:
        common2 = common_paths

    paths1 = sorted(common1 | iid_paths1)
    paths2 = sorted(common2 | iid_paths2)

    return paths1, paths2


def random_ordered_tree(n, seed=None):
    """
    Creates a random ordered tree

    TODO
    ----
    - [ ] Rename to random_ordered_directed_tree ?
    - [ ] Merge in with other data generators?

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    networkx.OrderedDiGraph

    Example
    -------
    >>> assert len(random_ordered_tree(n=1, seed=0).nodes) == 1
    >>> assert len(random_ordered_tree(n=2, seed=0).nodes) == 2
    >>> assert len(random_ordered_tree(n=3, seed=0).nodes) == 3
    >>> from netharn.initializers._nx_ext.tree_embedding import forest_str
    >>> print(forest_str(random_ordered_tree(n=5, seed=3)))
    └── 1
        ├── 4
        │   ├── 3
        │   └── 2
        └── 0
    """
    import networkx as nx
    from networkx.utils import create_py_random_state
    rng = create_py_random_state(seed)
    # Create a random undirected tree
    utree = nx.random_tree(n, seed=rng)
    # Use a random root node and dfs to define edge directions
    nodes = list(utree.nodes)
    source = rng.choice(nodes)
    edges = nx.dfs_edges(utree, source=source)
    # Populate the ordered graph
    otree = nx.OrderedDiGraph()
    otree.add_nodes_from(utree.nodes)
    otree.add_edges_from(edges)
    return otree


def random_balanced_sequence(n, seed=None, mode='chr', open_to_close=None):
    r"""
    Creates a random balanced sequence for testing / benchmarks

    Parameters
    ----------
    n : int
        A positive integer representing the number of nodes in the tree.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    open_to_close : dict | None
        if specified, updates existing open_to_close with tokens from this
        sequence.

    mode: str
        the type of sequence returned (see :func:`tree_to_seq` for details)

    Returns
    -------
    : tuple
        The first item is the sequence itself
        the second item is the open_to_close mappings.

    Example
    -------
    >>> # Demo the various sequence encodings that we might use
    >>> seq, open_to_close = random_balanced_sequence(2, seed=1, mode='tuple')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='chr')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='number')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(4, seed=1, mode='str')
    >>> print('seq = {!r}'.format(seq))
    >>> seq, open_to_close = random_balanced_sequence(10, seed=1, mode='paren')
    >>> print('seq = {!r}'.format(seq))
    seq = (('open', 0), ('open', 1), ('close', 1), ('close', 0))
    seq = '\x00\x02\x04\x06\x07\x05\x03\x01'
    seq = (1, 2, 3, 4, -4, -3, -2, -1)
    seq = ('2(', '1(', '0(', '3(', ')3', ')0', ')1', ')2')
    seq = '([[[]{{}}](){{[]}}])'
    """
    from networkx.utils import create_py_random_state
    from netharn.initializers._nx_ext.tree_embedding import tree_to_seq
    # Create a random otree and then convert it to a balanced sequence
    rng = create_py_random_state(seed)
    tree = random_ordered_tree(n, seed=rng)
    if mode == 'paren':
        pool = '[{('
        for node in tree.nodes:
            tree.nodes[node]['label'] = rng.choice(pool)
        seq, open_to_close, _ = tree_to_seq(
            tree, mode=mode, open_to_close=open_to_close, strhack=1)
    else:
        seq, open_to_close, _ = tree_to_seq(
            tree, mode=mode, open_to_close=open_to_close)
    return seq, open_to_close


def _convert_digit_base(digit, alphabet):
    """
    Parameters
    ----------
    digit : int
        number in base 10 to convert

    alphabet : list
        symbols of the conversion base
    """
    baselen = len(alphabet)
    x = digit
    if x == 0:
        return alphabet[0]
    sign = 1 if x > 0 else -1
    x *= sign
    digits = []
    while x:
        digits.append(alphabet[x % baselen])
        x //= baselen
    if sign < 0:
        digits.append('-')
    digits.reverse()
    newbase_str = ''.join(digits)
    return newbase_str
