from netharn.initializers._nx_ext.path_embedding import (  # NOQA
    maximum_common_path_embedding)
# from netharn.initializers._nx_ext.tree_embedding import (  # NOQA
#     maximum_common_ordered_tree_embedding, tree_to_seq)
from netharn.initializers._nx_ext.demodata import random_paths
from netharn.initializers._nx_ext.demodata import random_ordered_tree  # NOQA
import operator


def bench_maximum_common_path_embedding():
    """
    xdoctest -m netharn.initializers._nx_ext.benchmarks bench_maximum_common_path_embedding
    """
    import itertools as it
    import ubelt as ub
    import timerit
    from netharn.initializers._nx_ext import balanced_sequence
    from netharn.initializers._nx_ext import path_embedding

    data_modes = []

    # Define which implementations we are going to test
    run_basis = {
        'mode': [
            'chr',
            # 'number'
            # 'tuple',  # by far the slowest
        ],
        'impl': balanced_sequence.available_impls_longest_common_balanced_sequence(),
    }

    # Define the properties of the random data we are going to test on
    data_basis = {
        'size': [20, 50],
        'max_depth': [8, 16],
        'common': [8, 16],
        'prefix_depth1': [0, 4],
        'prefix_depth2': [0, 4],
        # 'labels': [26 ** 1, 26 ** 8]
        'labels': [1, 26]
    }

    # run_basis['impl'] = set(run_basis['impl']) & {
    #     'iter-alt2-cython',
    #     'iter-prehash2-cython',
    #     'iter-prehash2',
    #     'iter-alt2',
    #     # 'iter-alt1',
    #     # 'iter-prehash',
    #     # 'iter',
    #     # 'recurse'
    # }

    # TODO: parametarize demo names
    # BENCH_MODE = None
    # BENCH_MODE = 'small'
    # BENCH_MODE = 'small2'
    # BENCH_MODE = 'recursion-error'
    BENCH_MODE = 'medium'
    # BENCH_MODE = 'large'

    if BENCH_MODE == 'small':
        data_basis = {
            'size': [30],
            'max_depth': [8, 2],
            'common': [2, 8],
            'prefix_depth1': [0, 4],
            'prefix_depth2': [0],
            'labels': [4]
        }
        run_basis['impl'] = set(run_basis['impl']) & {
            # 'iter-alt2-cython',
            'iter-prehash2-cython',
            'iter-prehash2',
            # 'iter-alt2',
            # 'iter',
            # 'recurse',
        }
        run_basis['impl'] = ub.oset(balanced_sequence.available_impls_longest_common_balanced_sequence()) - {
                'recurse',
        }
        # runparam_to_time = {
        #     ('chr', 'iter-prehash2-cython'): {'mean': 0.062, 'max': 0.157},
        #     ('chr', 'iter-prehash2')       : {'mean': 0.071, 'max': 0.185},
        # }

    if BENCH_MODE == 'small2':
        data_basis = {
            'size': [30],
            'max_depth': [8, 2],
            'common': [2, 8],
            'prefix_depth1': [0, 4],
            'prefix_depth2': [0],
            'labels': [4]
        }
        run_basis['impl'] = ub.oset(balanced_sequence.available_impls_longest_common_balanced_sequence()) - {
                'recurse',
        }
        run_basis['mode'] = ['number', 'chr']
        # runparam_to_time = {
        #     ('chr', 'iter-alt2-cython')       : {'mean': 0.036, 'max': 0.094},
        #     ('chr', 'iter-alt2')              : {'mean': 0.049, 'max': 0.125},
        #     ('chr', 'iter-alt1')              : {'mean': 0.050, 'max': 0.129},
        #     ('chr', 'iter-prehash2-cython')   : {'mean': 0.057, 'max': 0.146},
        #     ('number', 'iter-prehash2-cython'): {'mean': 0.057, 'max': 0.146},
        #     ('chr', 'iter')                   : {'mean': 0.064, 'max': 0.167},
        #     ('chr', 'iter-prehash2')          : {'mean': 0.066, 'max': 0.170},
        #     ('number', 'iter-prehash2')       : {'mean': 0.067, 'max': 0.176},
        #     ('chr', 'iter-prehash')           : {'mean': 0.073, 'max': 0.187},
        #     ('number', 'iter-prehash')        : {'mean': 0.074, 'max': 0.196},
        #     ('number', 'iter-alt1')           : {'mean': 0.126, 'max': 0.344},
        #     ('number', 'iter-alt2-cython')    : {'mean': 0.133, 'max': 0.363},
        #     ('number', 'iter')                : {'mean': 0.140, 'max': 0.386},
        #     ('number', 'iter-alt2')           : {'mean': 0.149, 'max': 0.408},
        # }

    if BENCH_MODE == 'medium':
        data_basis = {
            'size': [30, 40],
            'max_depth': [4, 8],
            'common': [8, 50],
            'prefix_depth1': [0, 4],
            'prefix_depth2': [2],
            'labels': [8, 1]
        }
        # Results
        # runparam_to_time = {
        #     ('chr', 'iter-alt2-cython')    : {'mean': 0.112, 'max': 0.467},
        #     ('chr', 'recurse')             : {'mean': 0.153, 'max': 0.648},
        #     ('chr', 'iter-alt2')           : {'mean': 0.155, 'max': 0.661},
        #     ('chr', 'iter-alt1')           : {'mean': 0.163, 'max': 0.707},
        #     ('chr', 'iter-prehash2-cython'): {'mean': 0.197, 'max': 0.849},
        #     ('chr', 'iter')                : {'mean': 0.216, 'max': 0.933},
        #     ('chr', 'iter-prehash2')       : {'mean': 0.225, 'max': 0.974},
        #     ('chr', 'iter-prehash')        : {'mean': 0.253, 'max': 1.097},
        # }

    if BENCH_MODE == 'large':
        data_basis = {
            'size': [30, 40],
            'max_depth': [4, 12],  # 64000
            'common': [8, 32],
            'prefix_depth1': [0, 4],
            'prefix_depth2': [2],
            'labels': [8]
        }
        run_basis['impl'] = balanced_sequence.available_impls_longest_common_balanced_sequence()
        # runparam_to_time = {
        #     ('chr', 'iter-alt2-cython')    : {'mean': 0.282, 'max': 0.923},
        #     ('chr', 'recurse')             : {'mean': 0.397, 'max': 1.297},
        #     ('chr', 'iter-alt2')           : {'mean': 0.409, 'max': 1.328},
        #     ('chr', 'iter-alt1')           : {'mean': 0.438, 'max': 1.428},
        #     ('chr', 'iter-prehash2-cython'): {'mean': 0.511, 'max': 1.668},
        #     ('chr', 'iter')                : {'mean': 0.580, 'max': 1.915},
        #     ('chr', 'iter-prehash2')       : {'mean': 0.605, 'max': 1.962},
        #     ('chr', 'iter-prehash')        : {'mean': 0.679, 'max': 2.211},
        # }

    elif BENCH_MODE == 'too-big':
        data_basis = {
            'size': [100],
            'max_depth': [8],
            'common': [80],
            'prefix_depth1': [4],
            'prefix_depth2': [2],
            'labels': [8]
        }
    if BENCH_MODE == 'recursion-error':
        data_basis = {
            'size': [0],
            'max_depth': [512],
            'common': [4],
            'prefix_depth1': [0],
            'prefix_depth2': [0],
            'labels': [256]
        }
        run_basis['impl'] = ub.oset(['recurse']) | ub.oset(balanced_sequence.available_impls_longest_common_balanced_sequence())
        # Results
        # complexity = 69.48
        # stats1 = {'depth': 395, 'n_edges': 1203, 'n_leafs': 4, 'n_nodes': 1207, 'npaths': 4}
        # stats2 = {'depth': 395, 'n_edges': 1203, 'n_leafs': 4, 'n_nodes': 1207, 'npaths': 4}
        # runparam_to_time = {
        #     ('chr', 'recurse')             : {'mean': NAN, 'max': NAN},
        #     ('chr', 'iter-alt2-cython')    : {'mean': 7.979, 'max': 7.979},
        #     ('chr', 'iter-alt2')           : {'mean': 11.307, 'max': 11.307},
        #     ('chr', 'iter-alt1')           : {'mean': 11.659, 'max': 11.659},
        #     ('chr', 'iter-prehash2-cython'): {'mean': 15.230, 'max': 15.230},
        #     ('chr', 'iter-prehash2')       : {'mean': 17.058, 'max': 17.058},
        #     ('chr', 'iter')                : {'mean': 18.377, 'max': 18.377},
        #     ('chr', 'iter-prehash')        : {'mean': 19.508, 'max': 19.508},
        # }

    data_modes = [
        dict(zip(data_basis.keys(), vals))
        for vals in it.product(*data_basis.values())]
    run_modes = [
        dict(zip(run_basis.keys(), vals))
        for vals in it.product(*run_basis.values())]

    print('len(data_modes) = {!r}'.format(len(data_modes)))
    print('len(run_modes) = {!r}'.format(len(run_modes)))
    print('total = {}'.format(len(data_modes) * len(run_modes)))

    seed = 0
    # if len(data_modes) < 10:
    #     for datakw in data_modes:
    #         _datakw = ub.dict_diff(datakw, {'complexity'})
    #         paths1, paths2 = random_paths(seed=seed, **datakw)
    #         print('paths1 = {}'.format(ub.repr2(paths1, nl=1)))
    #         print('paths2 = {}'.format(ub.repr2(paths2, nl=1)))
    #         print('---')
    for idx, datakw in enumerate(data_modes):
        print('datakw = {}'.format(ub.repr2(datakw, nl=1)))
        _datakw = ub.dict_diff(datakw, {'complexity'})
        paths1, paths2 = random_paths(seed=seed, **_datakw)
        tree1 = path_embedding.paths_to_otree(paths1)
        tree2 = path_embedding.paths_to_otree(paths2)
        stats1 = {
            'npaths': len(paths1),
            'n_nodes': len(tree1.nodes),
            'n_edges': len(tree1.edges),
            'n_leafs': len([n for n in tree1.nodes if len(tree1.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths1),
        }
        stats2 = {
            'npaths': len(paths2),
            'n_nodes': len(tree2.nodes),
            'n_edges': len(tree2.edges),
            'n_leafs': len([n for n in tree2.nodes if len(tree2.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths2),
        }
        complexity = (
            stats1['n_nodes'] * min(stats1['n_leafs'], stats1['depth']) *
            stats2['n_nodes'] * min(stats2['n_leafs'], stats2['depth'])) ** .25

        datakw['complexity'] = complexity
        print('datakw = {}'.format(ub.repr2(datakw, nl=0, precision=2)))

        if True:
            # idx + 4 > len(data_modes):
            print('stats1 = {}'.format(ub.repr2(stats1, nl=0)))
            print('stats2 = {}'.format(ub.repr2(stats2, nl=0)))
            # print('complexity = {:.2f}'.format(complexity))

    total = len(data_modes) * len(run_modes)
    print('len(data_modes) = {!r}'.format(len(data_modes)))
    print('len(run_modes) = {!r}'.format(len(run_modes)))
    print('total = {!r}'.format(total))
    seed = 0

    prog = ub.ProgIter(total=total, verbose=3)
    prog.begin()
    results = []
    ti = timerit.Timerit(1, bestof=1, verbose=1, unit='s')
    for datakw in data_modes:
        _datakw = ub.dict_diff(datakw, {'complexity'})
        paths1, paths2 = random_paths(seed=seed, **_datakw)
        print('---')
        prog.step(4)
        tree1 = path_embedding.paths_to_otree(paths1)
        tree2 = path_embedding.paths_to_otree(paths2)
        stats1 = {
            'npaths': len(paths1),
            'n_nodes': len(tree1.nodes),
            'n_edges': len(tree1.edges),
            'n_leafs': len([n for n in tree1.nodes if len(tree1.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths1),
        }
        stats2 = {
            'npaths': len(paths2),
            'n_nodes': len(tree2.nodes),
            'n_edges': len(tree2.edges),
            'n_leafs': len([n for n in tree2.nodes if len(tree2.succ[n]) == 0]),
            'depth': max(len(p.split('/')) for p in paths2),
        }
        complexity = (
            stats1['n_nodes'] * min(stats1['n_leafs'], stats1['depth']) *
            stats2['n_nodes'] * min(stats2['n_leafs'], stats2['depth'])) ** .25

        datakw['complexity'] = complexity
        print('datakw = {}'.format(ub.repr2(datakw, nl=0, precision=2)))

        if True:
            # idx + 4 > len(data_modes):
            print('stats1 = {}'.format(ub.repr2(stats1, nl=0)))
            print('stats2 = {}'.format(ub.repr2(stats2, nl=0)))
        for runkw in run_modes:
            paramkw = {**datakw, **runkw}
            run_key = ub.repr2(
                paramkw, sep='', itemsep='', kvsep='',
                explicit=1, nobr=1, nl=0, precision=1)
            try:
                for timer in ti.reset(run_key):
                    with timer:
                        maximum_common_path_embedding(paths1, paths2, **runkw)
            except RecursionError as ex:
                print('ex = {!r}'.format(ex))
                row = paramkw.copy()
                row['time'] = float('nan')
            else:
                row = paramkw.copy()
                row['time'] = ti.min()
            results.append(row)
    prog.end()

    print(ub.repr2(ub.sorted_vals(ti.measures['min']), nl=1, align=':', precision=6))

    import pandas as pd
    import kwarray
    df = pd.DataFrame.from_dict(results)

    dataparam_to_time = {}
    for mode, subdf in df.groupby(['complexity'] + list(data_basis.keys())):
        stats = kwarray.stats_dict(subdf['time'])
        stats.pop('min', None)
        stats.pop('std', None)
        stats.pop('shape', None)
        dataparam_to_time[mode] = stats
    dataparam_to_time = ub.sorted_vals(dataparam_to_time, key=lambda x: x['max'])
    print('dataparam_to_time = {}'.format(ub.repr2(dataparam_to_time, nl=1, precision=3, align=':')))
    print(list(data_basis.keys()))

    runparam_to_time = {}
    for mode, subdf in df.groupby(['mode', 'impl']):
        stats = kwarray.stats_dict(subdf['time'])
        stats.pop('min', None)
        stats.pop('std', None)
        stats.pop('shape', None)
        runparam_to_time[mode] = stats
    runparam_to_time = ub.sorted_vals(runparam_to_time, key=lambda x: x['max'])
    print('runparam_to_time = {}'.format(ub.repr2(runparam_to_time, nl=1, precision=3, align=':')))


def benchmark_balanced_sequence_single():
    from netharn.initializers._nx_ext import balanced_sequence
    from netharn.initializers._nx_ext import demodata
    import ubelt as ub
    mode = 'number'
    seq1, open_to_close = demodata.random_balanced_sequence(200, mode=mode)
    seq2, open_to_close = demodata.random_balanced_sequence(400, mode=mode, open_to_close=open_to_close)
    longest_common_balanced_sequence = balanced_sequence.longest_common_balanced_sequence
    impls = balanced_sequence.available_impls_longest_common_balanced_sequence()
    results = {}
    for impl in impls:
        with ub.Timer(impl):
            best, val = longest_common_balanced_sequence(
                seq1, seq2, open_to_close, node_affinity=None, impl=impl)
            results[impl] = val
    assert allsame(results.values())


def allsame(iterable, eq=operator.eq):
    """
    Determine if all items in a sequence are the same

    Args:
        iterable (Iterable[A]):
            items to determine if they are all the same

        eq (Callable[[A, A], bool], default=operator.eq):
            function used to test for equality

    Returns:
        bool: True if all items are equal, otherwise False

    Example:
        >>> allsame([1, 1, 1, 1])
        True
        >>> allsame([])
        True
        >>> allsame([0, 1])
        False
        >>> iterable = iter([0, 1, 1, 1])
        >>> next(iterable)
        >>> allsame(iterable)
        True
        >>> allsame(range(10))
        False
        >>> allsame(range(10), lambda a, b: True)
        True
    """
    iter_ = iter(iterable)
    try:
        first = next(iter_)
    except StopIteration:
        return True
    return all(eq(first, item) for item in iter_)
