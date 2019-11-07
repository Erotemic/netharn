"""
Newest parts of the top-level API
"""
import ubelt as ub
import torch


class Optimizer(object):

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords 'optim'
        """
        import netharn as nh
        _update_defaults(config, kw)
        key = config['optim'] = config.get('optim', 'sgd').lower()
        if key == 'sgd':
            optim_ = (torch.optim.SGD, {
                'lr': config.get('lr', 1e-3),
                'weight_decay': config.get('decay', 0),
                'momentum': 0.9,
                'nesterov': True,
            })
        elif key == 'adam':
            optim_ = (torch.optim.Adam, {
                'lr': config.get('lr', 1e-3),
            })
        elif key == 'adamw':
            optim_ = (nh.optimizers.AdamW, {
                'lr': config.get('lr', 1e-3),
            })
        else:
            raise KeyError(key)
        return optim_


class Dynamics(object):

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords 'bstep'
        """
        _update_defaults(config, kw)
        return {
            # Controls how many batches to process before taking a step in the
            # gradient direction. Effectively simulates a batch_size that is
            # `bstep` times bigger.
            'batch_step': config.get('bstep', 1),
        }


class Scheduler(object):

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords 'optim'
        """
        import netharn as nh
        _update_defaults(config, kw)
        key = config['scheduler'] = config.get('scheduler', 'onecycle71').lower()
        lr = config['lr']
        if key.startswith('onecycle'):
            subkey = ''.join(key.split('onecycle')[1:])
            if subkey == '71':
                scheduler_ = (nh.schedulers.ListedScheduler, {
                    'points': {
                        'lr': {
                            0   : lr * 0.1,
                            35  : lr * 1.0,
                            70  : lr * 0.001,
                        },
                        'momentum': {
                            0   : 0.95,
                            35  : 0.85,
                            70  : 0.95,
                            71  : 0.999,
                        },
                    },
                })
        else:
            raise KeyError
        return scheduler_


def _update_defaults(config, kw):
    for k, v in kw.items():
        if k not in config:
            config[k] = v


class Loaders(object):

    @staticmethod
    def coerce(datasets, config={}, **kw):
        _update_defaults(config, kw)
        loaders = {}
        for key, dset in datasets.items():
            if hasattr(dset, 'make_loader'):
                loaders[key] = dset.make_loader(
                    batch_size=config['batch_size'],
                    num_workers=config['workers'], shuffle=(key == 'train'),
                    pin_memory=True)
            else:
                loaders[key] = torch.utils.data.DataLoader(
                    dset, batch_size=config['batch_size'], num_workers=config['workers'],
                    shuffle=(key == 'train'), pin_memory=True)
        return loaders


def configure_hacks(config={}, **kw):
    """
    Configures hacks to fix global settings in external modules

    Modules we currently hack:
        * cv2 - fix thread count
    """
    _update_defaults(config, kw)
    if config['workers'] > 0:
        import cv2
        cv2.setNumThreads(0)


def configure_workdir(config={}, **kw):
    _update_defaults(config, kw)
    if config['workdir'] is None:
        config['workdir'] = kw['workdir']
    workdir = config['workdir'] = ub.truepath(config['workdir'])
    ub.ensuredir(workdir)
    return workdir
