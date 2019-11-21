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


def _devcheck_remove_dead_runs(workdir, dry=True, dead_num_snap_thresh=10,
                               safe_num_days=7):
    """
    TODO:
         Look for directories in runs that have no / very few snapshots
         and no eval metrics that have a very old modified time and
         put them into a list as candidates for deletion

    """
    import ubelt as ub
    # workdir = ub.expandpath('~/work/foobar')

    print('Checking for dead / dangling sessions in your runs dir')

    # Find if any run directory is empty
    run_dpath = join(workdir, 'fit', 'runs')
    training_dpaths = list(glob.glob(join(run_dpath, '*/*')))

    infos = []
    for dpath in training_dpaths:
        info = session_info(dpath)
        infos.append(info)

    nice_groups = ub.group_items(infos, lambda x: x['nice'])

    for nice, group in nice_groups.items():
        print(' --- {} --- '.format(nice))
        group = sorted(group, key=lambda x: x['size'])
        import copy
        group_ = copy.deepcopy(group)
        for i in group_:
            i['dpath'] = '...' + i['dpath'][-20:]
            i.pop('last_modified')
            i['MB'] = i['size'] * 1e-3
        print(ub.repr2(group_, nl=1))

    # Partion your "nice" sessions into broken and live symlinks.
    # For each live link remember what the real path is.
    broken_links = []
    nice_dpath = join(workdir, 'fit', 'nice')
    for dname in os.listdir(nice_dpath):
        dpath = join(nice_dpath, dname)
        if is_symlink_broken(dpath):
            broken_links.append(dpath)

    empty_dpaths = []
    for dname in os.listdir(run_dpath):
        dpath = join(run_dpath, dname)
        if len(os.listdir(dpath)) == 0:
            empty_dpaths.append(dpath)

    bad_dpaths = []
    iffy_dpaths = []
    good_dpaths = []

    now = datetime.datetime.now()
    long_time_ago = now - datetime.timedelta(days=safe_num_days)

    for info in infos:
        if info['num_snapshots'] == 0:
            bad_dpaths.append(info['dpath'])
        elif info['num_snapshots'] < dead_num_snap_thresh:
            dt = info['last_modified']
            if dt < long_time_ago:
                iffy_dpaths.append(info['dpath'])
            else:
                good_dpaths.append(info['dpath'])
        else:
            good_dpaths.append(info['dpath'])

    if dry:
        print('Would leave {} good dpaths'.format(len(good_dpaths)))
        print('NOT DELETING {} iffy dpaths'.format(len(iffy_dpaths)))
        print('Would delete {} bad dpaths'.format(len(bad_dpaths)))
        print('Would delete {} broken links'.format(len(broken_links)))
        print('Would delete {} empty dpaths'.format(len(empty_dpaths)))
    else:
        print('Leaving {} good dpaths'.format(len(good_dpaths)))
        print('NOT DELETING {} iffy dpaths'.format(len(iffy_dpaths)))
        print('Deleting delete {} bad dpaths'.format(len(bad_dpaths)))
        print('Deleting delete {} broken links'.format(len(broken_links)))
        print('Deleting delete {} empty dpaths'.format(len(empty_dpaths)))
        # for p in iffy_dpaths:
        #     ub.delete(p)
        for p in bad_dpaths:
            ub.delete(p)
        for p in empty_dpaths:
            ub.delete(p)
        for p in broken_links:
            os.unlink(p)


def session_info(dpath):
    """
    Stats about a training session
    """
    info = {}
    snap_dpath = join(dpath, 'torch_snapshots')
    snapshots = os.listdir(snap_dpath) if exists(snap_dpath) else []
    dpath = realpath(dpath)

    if True:
        # Determine if we are pointed to by a nice directory or not
        nice = basename(dirname(dpath))
        info['nice'] = nice
        fitdir = dirname(dirname(dirname(dpath)))
        nice_dpath = join(fitdir, 'nice', nice)
        try:
            target = realpath(ub.util_links._readlink(nice_dpath))
        except Exception:
            target = None
        info['linked'] = (target == dpath)

    info['dpath'] = dpath
    info['num_snapshots'] = len(snapshots)
    info['size'] = float(ub.cmd('du -s ' + dpath)['out'].split('\t')[0])
    if len(snapshots) > 0:
        contents = [join(dpath, c) for c in os.listdir(dpath)]
        timestamps = [get_file_info(c)['last_modified'] for c in contents]
        unixtime = max(timestamps)
        dt = datetime.datetime.fromtimestamp(unixtime)
        info['last_modified'] = dt
    return info


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

    run_dpath = join(workdir, 'fit', 'runs')
    snapshot_dpaths = list(glob.glob(join(run_dpath, '**/torch_snapshots'), recursive=True))
    print('checking {} snapshot paths'.format(len(snapshot_dpaths)))

    all_keep = []
    all_remove = []

    for snapshot_dpath in snapshot_dpaths:
        snapshots = sorted(glob.glob(join(snapshot_dpath, '_epoch_*.pt')))
        epoch_to_snap = {
            int(parse.parse('{}_epoch_{num:d}.pt', path).named['num']): path
            for path in snapshots
        }
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

        print('Keep {}/{} from {}'.format(len(keep), len(existing_epochs), snapshot_dpath))
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

    total_mb = total / 2 ** 20
    if dry:
        print('Cleanup would delete {} snapshots and free {!r} MB'.format(len(all_remove), total_mb))
        print('Use -f to confirm and force cleanup')
    else:
        print('About to free {!r} MB'.format(total_mb))
        for path in ub.flatten(all_remove):
            ub.delete(path, verbose=True)


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
        print("A")
        _devcheck_manage_snapshots(**ns)
    else:
        raise KeyError(mode)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/dev/manage_snapshots.py

        python ~/code/netharn/dev/manage_snapshots.py --mode=snapshots --workdir=~/work/voc_yolo2/
        python ~/code/netharn/dev/manage_snapshots.py --mode=runs --workdir=~/work/voc_yolo2/

        python ~/code/netharn/dev/manage_snapshots.py --mode=snapshots --workdir=~/work/mc_harn3/ --recent 2 --factor 40
    """
    main()
