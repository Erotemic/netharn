from netharn.initializers._nx_ext.tree_embedding import (
    maximum_common_ordered_tree_embedding, forest_str)

from netharn.initializers._nx_ext.demodata import (
    random_ordered_tree
)
import networkx as nx
import pytest
from networkx.utils import create_py_random_state


def test_null_common_embedding():
    """
    The empty graph is not a tree and should raise an error
    """
    empty = nx.OrderedDiGraph()
    non_empty = random_ordered_tree(n=1)

    with pytest.raises(nx.NetworkXPointlessConcept):
        maximum_common_ordered_tree_embedding(empty, empty)

    with pytest.raises(nx.NetworkXPointlessConcept):
        maximum_common_ordered_tree_embedding(empty, non_empty)

    with pytest.raises(nx.NetworkXPointlessConcept):
        maximum_common_ordered_tree_embedding(non_empty, empty)


def test_self_common_embedding():
    """
    The common embedding of a tree with itself should always be itself
    """
    rng = create_py_random_state(85652972257)
    for n in range(1, 10):
        tree = random_ordered_tree(n=n, seed=rng)
        embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree, tree)
        assert tree.edges == embedding1.edges


def test_common_tree_embedding_small():
    tree1 = nx.OrderedDiGraph([(0, 1)])
    tree2 = nx.OrderedDiGraph([(0, 1), (1, 2)])
    print(forest_str(tree1))
    print(forest_str(tree2))

    embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2)
    print(forest_str(embedding1))
    print(forest_str(embedding2))


def test_common_tree_embedding_small2():
    tree1 = nx.OrderedDiGraph([(0, 1), (2, 3), (4, 5), (5, 6)])
    tree2 = nx.OrderedDiGraph([(0, 1), (1, 2), (0, 3)])
    print(forest_str(tree1))
    print(forest_str(tree2))

    embedding1, embedding2 = maximum_common_ordered_tree_embedding(tree1, tree2, node_affinity=None)
    print(forest_str(embedding1))
    print(forest_str(embedding2))


def test_all_implementations_are_same():
    """
    Tests several random sequences
    """
    from netharn.initializers._nx_ext import balanced_sequence
    from netharn.initializers._nx_ext import demodata
    from networkx.utils import create_py_random_state

    seed = 24658885408229410362279507020239
    rng = create_py_random_state(seed)

    maxsize = 20
    num_trials = 5

    for _ in range(num_trials):
        n1 = rng.randint(1, maxsize)
        n2 = rng.randint(1, maxsize)

        tree1 = demodata.random_ordered_tree(n1, seed=rng)
        tree2 = demodata.random_ordered_tree(n2, seed=rng)

        # Note: the returned sequences may be different (maximum embeddings may not
        # be unique), but the values should all be the same.
        results = {}
        impls = balanced_sequence.available_impls_longest_common_balanced_sequence()
        for impl in impls:
            # FIXME: do we need to rework the return value here?
            subtree1, subtree2 = maximum_common_ordered_tree_embedding(
                tree1, tree2, node_affinity=None, impl=impl)
            _check_common_embedding_invariants(tree1, tree2, subtree1, subtree2)
            results[impl] = len(subtree1.nodes)

        x = max(results.values())
        assert all(v == x for v in results.values())


def _check_embedding_invariants(tree, subtree):
    assert set(subtree.nodes).issubset(set(tree.nodes)), 'must have a node subset'
    assert len(subtree.edges) <= len(tree.edges)


def _check_common_embedding_invariants(tree1, tree2, subtree1, subtree2):
    """
    Validates that this solution satisfies properties of an embedding
    """
    _check_embedding_invariants(tree1, subtree1)
    _check_embedding_invariants(tree2, subtree2)
    assert len(subtree1.nodes) == len(subtree2.nodes)
