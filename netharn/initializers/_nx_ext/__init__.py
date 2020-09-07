"""
TEMPORARY FORK
--------------

CommandLine:
    sedr networkx.algorithms.isomorphism._embedding netharn.initializers._nx_ext
    sedr netharn.initializers._nx_ext netharn.initializers._nx_ext * True


Subpackages for helpers and such related to the ordered subtree embedding /
isomorphism problems.

Contains routines for solving balanced sequence and path subproblems. Only the
final graph-based API is exposed, but modification to the internals (is / will
be) available via keyword arguments.

balanced_sequence.py - core python implementations for the longest common
balanced sequence subproblem.

balanced_sequence_cython.pyx -
faster alternative implementsions for balanced_sequence.py

tree_embedding.py - defines reduction from tree problem to balanced sequence
problems.

path_embedding.py - defines reduction from path problem to tree problem (not
core, this is just useful for testing among other things).

demodata.py - Contains data for docstrings, benchmarks, and synthetic problems


Outstanding Issues
------------------
- [ ] Multiple implementations of the algorithm backend / data structure
  reduction, need to reduce the impelmentation and / or determine a better
  mechansim for allowing the user to switch between them.

- [ ] strhack is not a good API in `tree_to_seq`

- [ ] Should we return which edges were contracted in each tree to create the
  embeddings? That seems useful (but maybe not equivalent to the embeddings
  themselves?)

- [ ] How to deal with cython + networkx? Do we need to fix that skbuild with
  pypy?

- [ ] The open_to_node problem:
        Note, we may be able to simply use the position of each opening token
        as a proxy for unique tokens. Pass in an ordered list of nodes, then
        just use their indexes.


CommandLine
-----------
xdoctest -m netharn.initializers._nx_ext list
xdoctest -m netharn.initializers._nx_ext all

# Run all tests in this module
DPATH=$(python -c "
import os; import netharn.initializers._nx_ext as m;
print(os.path.dirname(m.__file__))")
pytest --xdoctest $DPATH --xdoc-analysis=dynamic

# The mkinit tool helps autogenerate explicit `__init__.py` files
mkinit ~/code/networkx/netharn.initializers._nx_ext/__init__.py -w
"""

__submodules__ = [
    'tree_embedding',
]

# from netharn.initializers._nx_ext import tree_embedding
from netharn.initializers._nx_ext.tree_embedding import (
    maximum_common_ordered_tree_embedding)

__all__ = ['maximum_common_ordered_tree_embedding']
