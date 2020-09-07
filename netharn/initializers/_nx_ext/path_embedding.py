import networkx as nx
from .tree_embedding import maximum_common_ordered_tree_embedding


def maximum_common_path_embedding(paths1, paths2, sep='/', impl='iter-alt2', mode='chr'):
    """
    Finds the maximum path embedding common between two sets of paths

    Parameters
    ----------
    paths1, paths2: List[str]
        a list of paths

    sep: str
        path separator character

    impl: str
        backend runtime to use

    mode: str
        backend representation to use

    Returns
    -------
    :tuple
    corresponding lists subpaths1 and subpaths2 which are subsets of
    paths1 and path2 respectively

    Examples
    --------
    >>> paths1 = [
    >>>     '/usr/bin/python',
    >>>     '/usr/bin/python3.6.1',
    >>>     '/usr/lib/python3.6/dist-packages/networkx',
    >>>     '/usr/lib/python3.6/dist-packages/numpy',
    >>>     '/usr/include/python3.6/Python.h',
    >>> ]
    >>> paths2 = [
    >>>     '/usr/local/bin/python',
    >>>     '/usr/bin/python3.6.2',
    >>>     '/usr/local/lib/python3.6/dist-packages/networkx',
    >>>     '/usr/local/lib/python3.6/dist-packages/scipy',
    >>>     '/usr/local/include/python3.6/Python.h',
    >>> ]
    >>> subpaths1, subpaths2 = maximum_common_path_embedding(paths1, paths2)
    >>> import pprint
    >>> print('subpaths1 = {}'.format(pprint.pformat(subpaths1)))
    >>> print('subpaths2 = {}'.format(pprint.pformat(subpaths2)))
    subpaths1 = ['/usr/bin/python',
     '/usr/include/python3.6/Python.h',
     '/usr/lib/python3.6/dist-packages/networkx']
    subpaths2 = ['/usr/local/bin/python',
     '/usr/local/include/python3.6/Python.h',
     '/usr/local/lib/python3.6/dist-packages/networkx']
    """
    # the longest common balanced sequence problem
    def _affinity(node1, node2):
        score = 0
        for t1, t2 in zip(node1[::-1], node2[::-1]):
            if t1 == t2:
                score += 1
            else:
                break
        return score
    node_affinity = _affinity

    tree1 = paths_to_otree(paths1, sep=sep)
    tree2 = paths_to_otree(paths2, sep=sep)

    subtree1, subtree2 = maximum_common_ordered_tree_embedding(
            tree1, tree2, node_affinity=node_affinity, impl=impl, mode=mode)

    subpaths1 = [sep.join(node) for node in subtree1.nodes if subtree1.out_degree[node] == 0]
    subpaths2 = [sep.join(node) for node in subtree2.nodes if subtree2.out_degree[node] == 0]
    return subpaths1, subpaths2


def paths_to_otree(paths, sep='/'):
    """
    Generates an ordered tree from a list of path strings

    Parameters
    ----------
    paths: List[str]
        a list of paths

    sep : str
        path separation character. defaults to "/"

    Returns
    -------
    nx.OrderedDiGraph

    Example
    -------
    >>> from netharn.initializers._nx_ext.tree_embedding import forest_str
    >>> paths = [
    >>>     '/etc/ld.so.conf',
    >>>     '/usr/bin/python3.6',
    >>>     '/usr/include/python3.6/Python.h',
    >>>     '/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so',
    >>>     '/usr/local/bin/gnumake.h',
    >>>     '/usr/local/etc',
    >>>     '/usr/local/lib/python3.6/dist-packages/',
    >>> ]
    >>> otree = paths_to_otree(paths)
    >>> print(forest_str(otree))
    └── /
        ├── usr
        │   ├── local
        │   │   ├── lib
        │   │   │   └── python3.6
        │   │   │       └── dist-packages
        │   │   │           └──
        │   │   ├── etc
        │   │   └── bin
        │   │       └── gnumake.h
        │   ├── lib
        │   │   └── python3.6
        │   │       └── config-3.6m-x86_64-linux-gnu
        │   │           └── libpython3.6.so
        │   ├── include
        │   │   └── python3.6
        │   │       └── Python.h
        │   └── bin
        │       └── python3.6
        └── etc
            └── ld.so.conf
    """
    otree = nx.OrderedDiGraph()
    for path in sorted(paths):
        parts = tuple(path.split(sep))
        node_path = []
        for i in range(1, len(parts) + 1):
            node = parts[0:i]
            otree.add_node(node)
            otree.nodes[node]['label'] = node[-1]
            node_path.append(node)
        for u, v in zip(node_path[:-1], node_path[1:]):
            otree.add_edge(u, v)
    if ('',) in otree.nodes:
        otree.nodes[('',)]['label'] = sep
    return otree
