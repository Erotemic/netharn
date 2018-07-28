
def _devcheck_manage_snapshots(workdir):
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

    import glob
    from os.path import join
    import parse
    import ubelt as ub
    import numpy as np
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
        print('existing_epochs = {!r}'.format(existing_epochs))
        toremove = []
        tokeep = []

        if USE_RANGE_HUERISTIC:
            # My Critieron is that I'm only going to keep the two latest and
            # I'll also keep an epoch in the range [0,50], [50,100], and
            # [100,150], and so on.
            factor = 75
            existing_epochs = sorted(existing_epochs)
            dups = ub.find_duplicates(np.array(sorted(existing_epochs)) // factor, k=0)
            keep_idxs = [max(idxs) for _, idxs in dups.items()]
            keep = set(ub.take(existing_epochs, keep_idxs))
            keep.update(existing_epochs[-1:])

            if existing_epochs and existing_epochs[0] != 0:
                keep.update(existing_epochs[0:1])

            for epoch, path in epoch_to_snap.items():
                if epoch in keep:
                    tokeep.append(path)
                else:
                    toremove.append(path)
        all_keep += [tokeep]
        all_remove += [toremove]

    print('all_keep = {}'.format(ub.repr2(all_keep, nl=2)))
    print('all_remove = {}'.format(ub.repr2(all_remove)))

    """
    pip install send2trash
    import send2trash
    send2trash.send2trash(path)
    """
    import os
    total = 0
    for path in ub.flatten(all_remove):
        total += os.path.getsize(path)

    total_mb = total / 2 ** 20
    print('About to free {!r} MB'.format(total_mb))

    for path in ub.flatten(all_remove):
        ub.delete(path, verbose=True)
