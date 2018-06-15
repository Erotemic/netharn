import ubelt as ub


def read_tensorboard_scalars(train_dpath):
    """
    Reads all tensorboard scalar events in a directory.
    Caches them becuase reading events of interest from protobuf can be slow.
    """
    import glob
    import tqdm
    from os.path import join
    from tensorboard.backend.event_processing import event_accumulator
    event_paths = sorted(glob.glob(join(train_dpath, 'events.out.tfevents*')))
    # make a hash so we will re-read of we need to
    cfgstr = ub.hash_data(list(map(ub.hash_file, event_paths)))
    # cfgstr = ub.hash_data(list(map(basename, event_paths)))
    cacher = ub.Cacher('tb_scalars',
                       dpath=ub.ensuredir((train_dpath, '_cache')),
                       cfgstr=cfgstr)
    datas = cacher.tryload()
    if datas is None:
        datas = {}
        for p in tqdm.tqdm(list(reversed(event_paths)), desc='read tensorboard'):
            ea = event_accumulator.EventAccumulator(p)
            ea.Reload()
            for key in ea.scalars.Keys():
                if key not in datas:
                    datas[key] = {'xdata': [], 'ydata': [], 'wall': []}
                subdatas = datas[key]
                events = ea.scalars.Items(key)
                for e in events:
                    subdatas['xdata'].append(int(e.step))
                    subdatas['ydata'].append(float(e.value))
                    subdatas['wall'].append(float(e.wall_time))

        # Order all information by its wall time
        for key, subdatas in datas.items():
            sortx = ub.argsort(subdatas['wall'])
            for d, vals in subdatas.items():
                subdatas[d] = list(ub.take(vals, sortx))
        cacher.save(datas)
    return datas
