#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BE VERY CAREFUL WHEN RUNNING THIS SCRIPT!!!
"""
from os.path import realpath
from os.path import basename
from os.path import dirname
from os.path import join, exists
import datetime
import glob
import numpy as np
import os
import parse
import ubelt as ub
import copy


def byte_str(num, unit='auto', precision=2):
    """
    Automatically chooses relevant unit (KB, MB, or GB) for displaying some
    number of bytes.

    Args:
        num (int): number of bytes
        unit (str): which unit to use, can be auto, B, KB, MB, GB, TB, PB, EB,
            ZB, or YB.

    References:
        https://en.wikipedia.org/wiki/Orders_of_magnitude_(data)

    Returns:
        str: string representing the number of bytes with appropriate units

    Example:
        >>> num_list = [1, 100, 1024,  1048576, 1073741824, 1099511627776]
        >>> result = ub.repr2(list(map(byte_str, num_list)), nl=0)
        >>> print(result)
        ['0.00 KB', '0.10 KB', '1.00 KB', '1.00 MB', '1.00 GB', '1.00 TB']
    """
    abs_num = abs(num)
    if unit == 'auto':
        if abs_num < 2.0 ** 10:
            unit = 'KB'
        elif abs_num < 2.0 ** 20:
            unit = 'KB'
        elif abs_num < 2.0 ** 30:
            unit = 'MB'
        elif abs_num < 2.0 ** 40:
            unit = 'GB'
        elif abs_num < 2.0 ** 50:
            unit = 'TB'
        elif abs_num < 2.0 ** 60:
            unit = 'PB'
        elif abs_num < 2.0 ** 70:
            unit = 'EB'
        elif abs_num < 2.0 ** 80:
            unit = 'ZB'
        else:
            unit = 'YB'
    if unit.lower().startswith('b'):
        num_unit = num
    elif unit.lower().startswith('k'):
        num_unit =  num / (2.0 ** 10)
    elif unit.lower().startswith('m'):
        num_unit =  num / (2.0 ** 20)
    elif unit.lower().startswith('g'):
        num_unit = num / (2.0 ** 30)
    elif unit.lower().startswith('t'):
        num_unit = num / (2.0 ** 40)
    elif unit.lower().startswith('p'):
        num_unit = num / (2.0 ** 50)
    elif unit.lower().startswith('e'):
        num_unit = num / (2.0 ** 60)
    elif unit.lower().startswith('z'):
        num_unit = num / (2.0 ** 70)
    elif unit.lower().startswith('y'):
        num_unit = num / (2.0 ** 80)
    else:
        raise ValueError('unknown num={!r} unit={!r}'.format(num, unit))
    return ub.repr2(num_unit, precision=precision) + ' ' + unit


def is_symlink_broken(path):
    """
    Check is a path is a broken symlink.

    Args:
        path (PathLike): path to check

    Returns:
        bool: True if the file is a broken symlink.

    Raises:
        TypeError: if the input is not a symbolic link

    References:
        https://stackoverflow.com/questions/20794/find-broken-symlinks-with-python

    Example:
        >>> test_dpath = ub.ensure_app_cache_dir('test')
        >>> real_fpath = ub.touch(join(test_dpath, 'real'))
        >>> link_fpath = ub.symlink(real_fpath, join(test_dpath, 'link'))
        >>> assert not is_symlink_broken(link_fpath)
        >>> ub.delete(real_fpath)
        >>> assert is_symlink_broken(link_fpath)
        >>> import pytest
        >>> with pytest.raises(TypeError):
        >>>     assert is_symlink_broken(test_dpath)
    """
    if os.path.islink(path):
        return not os.path.exists(os.readlink(path))
    else:
        raise TypeError('path={!r} is not a symbolic link'.format(path))


def get_file_info(fpath):
    from collections import OrderedDict
    from pwd import getpwuid
    statbuf = os.stat(fpath)
    owner = getpwuid(os.stat(fpath).st_uid).pw_name

    info = OrderedDict([
        # ('filesize', get_file_nBytes_str(fpath)),
        ('last_modified', statbuf.st_mtime),
        ('last_accessed', statbuf.st_atime),
        ('created', statbuf.st_ctime),
        ('owner', owner)
    ])
    return info


def _demodata_workdir():
    """
    Make a work directory with various types of sessions
    """
    workdir = ub.ensure_app_cache_dir('netharn/tests/sessions')
    def _demodata_toy_sesssion(workdir, name='demo_session', lr=1e-4):
        """
        workdir = ub.ensure_app_cache_dir('netharn/tests/sessions')
        workdir
        """
        # This will train a toy model with toy data using netharn
        import netharn as nh
        hyper = nh.HyperParams(**{
            'workdir'     : ub.ensure_app_cache_dir('netharn/tests/sessions'),
            'name'        : name,
            'xpu'         : nh.XPU.coerce('cpu'),
            'datasets'    : {'train': nh.data.ToyData2d(size=3, rng=0), 'vali': nh.data.ToyData2d(size=3, rng=0)},
            'loaders'     : {'batch_size': 64},
            'model'       : (nh.models.ToyNet2d, {}),
            'optimizer'   : (nh.optimizers.SGD, {'lr': lr}),
            'criterion'   : (nh.criterions.FocalLoss, {}),
            'initializer' : (nh.initializers.KaimingNormal, {}),
            'monitor'     : (nh.Monitor, {'max_epoch': 1}),
        })
        harn = nh.FitHarn(hyper)
        harn.preferences['use_tensorboard'] = False
        harn.preferences['timeout'] = 1
        harn.run()  # TODO: make this run faster if we don't need to rerun

    _demodata_toy_sesssion(workdir, name='demo_session1', lr=1e-3)
    _demodata_toy_sesssion(workdir, name='demo_session2', lr=1e-3)
    _demodata_toy_sesssion(workdir, name='demo_session3', lr=1e-3)
    _demodata_toy_sesssion(workdir, name='demo_session2', lr=1e-4)
    _demodata_toy_sesssion(workdir, name='demo_session3', lr=1e-4)
    _demodata_toy_sesssion(workdir, name='demo_session3', lr=1e-5)
    return workdir


def collect_sessions(workdir):
    """
    Netharn writes all training runs into a work directory under
    <workdir>/fit/runs/<hash>/<name>.  And makes symlinks in
    <workdir>/fit/name/<name>. This collects all sessions within a workdir that
    match the filter criteria.

    workdir = _demodata_workdir()
    all_sessions = collect_sessions(workdir)

    """
    run_dpath = join(workdir, 'fit', 'runs')
    training_dpaths = list(glob.glob(join(run_dpath, '*/*')))

    all_sessions = []
    for dpath in ub.ProgIter(training_dpaths, desc='collect sessions', freq=1):
        session = Session(dpath)
        all_sessions.append(session)
    return all_sessions


class Session(ub.NiceRepr):
    """
    NEW: object to maintain info / manipulate a specific training directory

    TODO:
        - [ ] Lazy properties
        - [ ] Better convinience methods
        - [ ] Log parsing
    """
    def __init__(session, dpath):
        session.dpath = dpath
        info, details = session_info(session.dpath)
        session.info = info
        session.details = details

    def __nice__(session):
        return repr(session.info)


def session_info(dpath):
    """
    Stats about a training session
    """
    info = {}
    snap_dpath = join(dpath, 'torch_snapshots')
    check_dpath = join(dpath, 'checkpoints')
    if exists(check_dpath):
        snapshots = os.listdir(check_dpath) if exists(check_dpath) else []
        snapshots = [join(check_dpath, fname) for fname in snapshots]
    elif exists(snap_dpath):
        # Old snapshot directory name
        snapshots = os.listdir(snap_dpath) if exists(snap_dpath) else []
        snapshots = [join(snap_dpath, fname) for fname in snapshots]
    else:
        snapshots = []
    dpath = realpath(dpath)

    if True:
        # Determine if we are pointed to by a "name" directory or not
        name = basename(dirname(dpath))
        info['name'] = name
        fitdir = dirname(dirname(dirname(dpath)))
        target = None
        name_dpath = join(fitdir, 'name', name)
        if exists(name_dpath):
            try:
                target = realpath(ub.util_links._readlink(name_dpath))
            except Exception:
                target = None
        else:
            nice_dpath = join(fitdir, 'nice', name)
            if exists(nice_dpath):
                try:
                    target = realpath(ub.util_links._readlink(nice_dpath))
                except Exception:
                    target = None
        info['linked'] = (target == dpath)

    best_snapshot_fpath = join(dpath, 'best_snapshot.pt')
    details = {}
    details['best_snapshot'] = best_snapshot_fpath if exists(best_snapshot_fpath) else None
    details['deployed'] = [p for p in glob.glob(join(dpath, '*.zip')) if not ub.util_links.islink(p)]
    details['snapshots'] = snapshots

    info['dpath'] = dpath
    info['has_deploy'] = bool(details['deployed'])
    info['has_best'] = bool(details['best_snapshot'])
    info['num_snapshots'] = len(snapshots)
    info['size'] = float(ub.cmd('du -s ' + dpath)['out'].split('\t')[0])
    if len(snapshots) > 0:
        contents = [join(dpath, c) for c in os.listdir(dpath)]
        timestamps = [get_file_info(c)['last_modified'] for c in contents if exists(c)]
        unixtime = max(timestamps)
        dt = datetime.datetime.fromtimestamp(unixtime)
        info['last_modified'] = dt
    return info, details


def _devcheck_remove_dead_runs(workdir, dry=True, dead_num_snap_thresh=10,
                               safe_num_days=7):
    """
    Look for directories in runs that have no / very few snapshots and no eval
    metrics that have a very old modified time and put them into a list as
    candidates for deletion.

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/netharn/dev'))
        from manage_snapshots import *  # NOQA
        from manage_snapshots import _devcheck_remove_dead_runs, _devcheck_manage_snapshots
        workdir = '.'
        import xdev
        globals().update(xdev.get_func_kwargs(_devcheck_remove_dead_runs))

        workdir = _demodata_workdir()
        _devcheck_remove_dead_runs(workdir)
    """
    print('Checking for dead / dangling sessions in your runs dir')

    run_dpath = join(workdir, 'fit', 'runs')
    all_sessions = collect_sessions(workdir)

    # Find if any run directory is empty
    now = datetime.datetime.now()
    long_time_ago = now - datetime.timedelta(days=safe_num_days)

    for session in all_sessions:
        info = session.info
        if not (info['has_deploy'] or info['num_snapshots'] or info['has_best']):
            info['decision'] = 'bad'
        elif info['num_snapshots'] < dead_num_snap_thresh:
            dt = info.get('last_modified', now)
            if dt < long_time_ago:
                info['decision'] = 'iffy'
            else:
                info['decision'] = 'good'
        else:
            info['decision'] = 'good'

    all_info = [s.info for s in all_sessions]

    nice_groups = ub.group_items(all_info, lambda x: x['name'])
    for name, group in nice_groups.items():
        print(' --- {} --- '.format(name))
        group = sorted(group, key=lambda x: x['size'])
        group_ = copy.deepcopy(group)
        for item in group_:
            item['dpath'] = '...' + item['dpath'][-20:]
            item.pop('last_modified', None)
            item['size'] = byte_str(item['size'])
        print(ub.repr2(group_, nl=1))

    # Partion your "name" sessions into broken and live symlinks.
    # For each live link remember what the real path is.
    broken_links = []
    nice_dpath = join(workdir, 'fit', 'nice')
    name_dpath = join(workdir, 'fit', 'name')
    dpaths = []
    if exists(name_dpath):
        dpaths += [join(name_dpath, d) for d in os.listdir(name_dpath)]
    if exists(nice_dpath):
        dpaths += [join(nice_dpath, d) for d in os.listdir(nice_dpath)]
    for dpath in dpaths:
        if is_symlink_broken(dpath):
            broken_links.append(dpath)

    empty_dpaths = []
    for dname in os.listdir(run_dpath):
        dpath = join(run_dpath, dname)
        if len(os.listdir(dpath)) == 0:
            empty_dpaths.append(dpath)

    decision_groups = ub.group_items(all_info, lambda x: x['decision'])

    print('Empty dpaths:  {:>4}'.format(len(empty_dpaths)))
    print('Broken links:  {:>4}'.format(len(broken_links)))
    for key in decision_groups.keys():
        group = decision_groups[key]
        size = byte_str(sum([s['size'] for s in group]))
        print('{:>4} sessions:  {:>4}, size={}'.format(key.capitalize(), len(group), size))

    if dry:
        print('DRY RUN. NOT DELETING ANYTHING')
    else:
        print('LIVE RUN. DELETING bad, empty, and broken.')
        print('NOT DELETING iffy and good sessions')

        # for p in iffy_sessions:
        #     ub.delete(p)
        for info in decision_groups.get('bad', []):
            ub.delete(info['dpath'])
        for p in empty_dpaths:
            ub.delete(p)
        for p in broken_links:
            os.unlink(p)


def _devcheck_manage_monitor(workdir, dry=True):

    all_sessions = collect_sessions(workdir)

    # Get all the images in the monitor directories
    # (this is a convention and not something netharn does by default)

    all_files = []
    # factor = 100
    max_keep = 300

    def _choose_action(file_infos):
        import kwarray
        file_infos = kwarray.shuffle(file_infos, rng=0)
        n_keep = max_keep
        # n_keep = (len(file_infos) // factor) + 1
        # n_keep = min(max_keep, n_keep)

        for info in file_infos[:n_keep]:
            info['action'] = 'keep'
        for info in file_infos[n_keep:]:
            info['action'] = 'delete'

    for session in ub.ProgIter(all_sessions, desc='checking monitor files'):
        dpaths = [
            join(session.dpath, 'monitor', 'train', 'batch'),
            join(session.dpath, 'monitor', 'vali', 'batch'),
            join(session.dpath, 'monitor', 'train'),
            join(session.dpath, 'monitor', 'vali'),
        ]
        exts = ['*.jpg', '*.png']
        for dpath in dpaths:
            for ext in exts:
                fpaths = list(glob.glob(join(dpath, ext)))
                file_infos = [{'size': os.stat(p).st_size, 'fpath': p}
                              for p in fpaths]
                _choose_action(file_infos)
                all_files.extend(file_infos)

    grouped_actions = ub.group_items(all_files, lambda x: x['action'])

    for key, group in grouped_actions.items():
        size = byte_str(sum([s['size'] for s in group]))
        print('{:>4} images:  {:>4}, size={}'.format(key.capitalize(), len(group), size))

    if dry:
        print('Dry run')
    else:
        delete = grouped_actions.get('delete', [])
        delete_fpaths = [item['fpath'] for item in delete]
        for p in ub.ProgIter(delete_fpaths, desc='deleting'):
            ub.delete(p)


def _devcheck_manage_snapshots(workdir, recent=5, factor=10, dry=True):
    """
    Sometimes netharn produces too many snapshots. The Monitor class attempts
    to prevent this, but its not perfect. So, sometimes you need to manually
    clean up. This code snippet serves as a template for doing so.

    I recommend using IPython to do this following this code as a guide.
    Unfortunately, I don't have a safe automated way of doing this yet.

    The basic code simply lists all snapshots that you have. Its then your job
    to find a huerstic to remove the ones you don't need.

    Note:
        # Idea for more automatic method

        In the future, we should use monitor to inspect the critical points of
        all metric curves and include any epoch that is at those cricial
        points. A cricial point is defined as one where there is a significant
        change in trajectory. Basically, we try to fit a low-degree polynomial
        or piecewise linear function to the metric curves, and we take the
        places where there is a significant change from a global perspective.

    # Specify your workdir
    workdir = ub.expandpath('~/work/voc_yolo2')
    """

    USE_RANGE_HUERISTIC = True

    all_sessions = collect_sessions(workdir)
    print('Checking sessions = {}'.format(ub.repr2(all_sessions, nl=1)))

    all_keep = []
    all_remove = []

    for session in all_sessions:
        snapshots = session.details['snapshots']
        epoch_to_snap = {}
        extra_types = {'prefix': parse.with_pattern('.*')(ub.identity)}
        for path in snapshots:
            parsed = parse.parse('{:prefix}_epoch_{num:d}.pt', path, extra_types)
            if parsed:
                epoch = int(parsed.named['num'])
                epoch_to_snap[epoch] = path
        existing_epochs = sorted(epoch_to_snap.keys())
        # print('existing_epochs = {}'.format(ub.repr2(existing_epochs)))
        toremove = []
        tokeep = []

        if USE_RANGE_HUERISTIC:
            # My Critieron is that I'm only going to keep the two latest and
            # I'll also keep an epoch in the range [0,50], [50,100], and
            # [100,150], and so on.
            existing_epochs = sorted(existing_epochs)
            dups = ub.find_duplicates(np.array(sorted(existing_epochs)) // factor, k=0)
            keep_idxs = [max(idxs) for _, idxs in dups.items()]
            keep = set(ub.take(existing_epochs, keep_idxs))

            keep.update(existing_epochs[-recent:])

            if existing_epochs and existing_epochs[0] != 0:
                keep.update(existing_epochs[0:1])

            kill = []
            for epoch, path in epoch_to_snap.items():
                if epoch in keep:
                    tokeep.append(path)
                else:
                    kill.append(epoch)
                    toremove.append(path)
            # print('toremove = {!r}'.format(toremove))
            print('keep = {!r}'.format(sorted(keep)))
            print('kill = {!r}'.format(sorted(kill)))

        print('Keep {}/{} from {}'.format(len(keep), len(existing_epochs), session.info['dpath']))
        all_keep += [tokeep]
        all_remove += [toremove]

    # print('all_keep = {}'.format(ub.repr2(all_keep, nl=2)))
    # print('all_remove = {}'.format(ub.repr2(all_remove, nl=2)))
    """
    pip install send2trash
    import send2trash
    send2trash.send2trash(path)
    """
    total = 0
    for path in ub.flatten(all_remove):
        total += os.path.getsize(path)

    if dry:
        print('Cleanup would delete {} snapshots and free {}'.format(len(all_remove), byte_str(total)))
        print('Use -f to confirm and force cleanup')
    else:
        print('About to free {}'.format(byte_str(total)))
        fpaths = list(ub.flatten(all_remove))
        for path in ub.ProgIter(fpaths, desc='deleting'):
            ub.delete(path)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog='manage_snapshots',
        description=ub.codeblock(
            '''
            Cleanup snapshots and dead runs produced by netharn
            ''')
    )
    parser.add_argument(*('-w', '--workdir'), type=str,
                        help='specify the workdir for your project', default=None)
    parser.add_argument(*('-f', '--force'), help='dry run',
                        action='store_false', dest='dry')
    # parser.add_argument(*('-n', '--dry'), help='dry run', action='store_true')
    parser.add_argument(*('--recent',), help='num recent to keep', type=int, default=100)
    parser.add_argument(*('--factor',), help='keep one every <factor> epochs', type=int, default=1)
    parser.add_argument('--mode', help='either runs or shapshots', default='snapshots')

    args, unknown = parser.parse_known_args()
    ns = args.__dict__.copy()
    print('ns = {!r}'.format(ns))

    mode = ns.pop('mode')
    ns['workdir'] = ub.expandpath(ns['workdir'])

    if mode == 'runs':
        _devcheck_remove_dead_runs(workdir=ns['workdir'], dry=ns['dry'])
    elif mode == 'snapshots':
        _devcheck_manage_snapshots(**ns)
    elif mode == 'monitor':
        _devcheck_manage_monitor(workdir=ns['workdir'], dry=ns['dry'])
    else:
        raise KeyError(mode)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/dev/manage_snapshots.py

        find . -iname "explit_checkpoints" -d

        python ~/code/netharn/dev/manage_snapshots.py --mode=snapshots --workdir=~/work/voc_yolo2/  --recent 2 --factor 40
        python ~/code/netharn/dev/manage_snapshots.py --mode=runs --workdir=~/work/voc_yolo2/
        python ~/code/netharn/dev/manage_snapshots.py --mode=monitor --workdir=~/work/voc_yolo2/
        python ~/code/netharn/dev/manage_snapshots.py --mode=monitor --workdir=. -f
        python ~/code/netharn/dev/manage_snapshots.py --mode=runs --workdir=.
        python ~/code/netharn/dev/manage_snapshots.py --mode=snapshots --workdir=. --recent 2 --factor 40 -f

    Notes:
        # Remove random files
        # https://superuser.com/questions/1186350/delete-all-but-1000-random-files-in-a-directory
        find . -type f -print0 | sort -zR | tail -zn +501 | xargs -0 rm



    """
    main()
