from os.path import join, exists
import datetime
import glob
import numpy as np
import os
import parse
import ubelt as ub


def _devcheck_remove_dead_runs(workdir):
    """
    TODO:
         Look for directories in runs that have no / very few snapshots
         and no eval metrics that have a very old modified time and
         put them into a list as candidates for deletion

    """
    import ubelt as ub
    workdir = ub.truepath('~/work/foobar')
    nice_dpath = join(workdir, 'fit', 'nice')

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

    bad_dpaths = []
    iffy_dpaths = []

    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)

    for dname in os.listdir(nice_dpath):
        dpath = join(nice_dpath, dname)

        contents = [join(dpath, c) for c in os.listdir(dpath)]
        timestamps = [get_file_info(c)['last_modified'] for c in contents]

        snap_dpath = join(dpath, 'torch_snapshots')
        if exists(snap_dpath):
            snapshots = os.listdir(snap_dpath)
            if len(snapshots) < 15:
                unixtime = max(timestamps)
                dt = datetime.datetime.fromtimestamp(unixtime)

                if dt < yesterday:
                    timefmt = '%Y/%m/%d %H:%M:%S'
                    print('---')
                    print(dt.strftime(timefmt))
                    print('dt = {!r}'.format(dt))
                    print('unixtime = {!r}'.format(unixtime))
                    dpath
                    print(len(snapshots))
                    print('dpath = {!r}'.format(dpath))
                    iffy_dpaths.append(dpath)

        else:
            bad_dpaths.append(dpath)

    if False:
        for p in iffy_dpaths:
            print('DELETE p = {!r}'.format(p))
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
    workdir = ub.truepath('~/work/voc_yolo2')
    """

    USE_RANGE_HUERISTIC = True

    run_dpath = join(workdir, 'fit', 'runs')
    snapshot_dpaths = list(glob.glob(join(run_dpath, '**/torch_snapshots')))

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

            print('keep = {!r}'.format(sorted(keep)))

            for epoch, path in epoch_to_snap.items():
                if epoch in keep:
                    tokeep.append(path)
                else:
                    toremove.append(path)

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
        print('Cleanup would free {!r} MB'.format(total_mb))
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
            Cleanup snapshots produced by netharn
            ''')
    )
    parser.add_argument(*('-w', '--workdir'), type=str,
                        help='specify the workdir for your project', default=None)
    parser.add_argument(*('-f', '--force'), help='dry run',
                        action='store_false', dest='dry')
    # parser.add_argument(*('-n', '--dry'), help='dry run', action='store_true')
    parser.add_argument(*('--recent',), help='num recent to keep', type=int, default=100)
    parser.add_argument(*('--factor',), help='keep one every <factor> epochs', type=int, default=1)

    args, unknown = parser.parse_known_args()
    ns = args.__dict__.copy()
    print('ns = {!r}'.format(ns))

    ns['workdir'] = ub.truepath(ns['workdir'])

    _devcheck_manage_snapshots(**ns)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/tools/manage_snapshots.py

        python ~/code/netharn/tools/manage_snapshots.py --workdir=~/work/voc_yolo2/
    """
    main()
