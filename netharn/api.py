"""
Newest parts of the top-level API
"""
import ubelt as ub
import torch


class Initializer(object):
    """
    Base class for all netharn initializers
    """
    def __call__(self, model, *args, **kwargs):
        return self.forward(model, *args, **kwargs)

    def forward(self, model):
        """
        Abstract function that does the initailization
        """
        raise NotImplementedError('implement me')

    def history(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        return None

    def get_initkw(self):
        """
        Initializer methods have histories which are short for algorithms and
        can be quite long for pretrained models
        """
        initkw = self.__dict__.copy()
        # info = {}
        # info['__name__'] = self.__class__.__name__
        # info['__module__'] = self.__class__.__module__
        # info['__initkw__'] = initkw
        return initkw

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts 'init', 'pretrained', 'pretrained_fpath', and 'noli'

        Args:
            config (dict):

        Returns:
            Tuple[nh.Initializer, dict]: initializer_ = initializer_cls, kw

        Examples:
            >>> import netharn as nh
            >>> print(ub.repr2(nh.Initializer.coerce({'init': 'noop'})))
            (
                <class 'netharn.initializers.core.NoOp'>,
                {},
            )
            >>> config = {
            ...     'init': 'pretrained',
            ...     'pretrained_fpath': '/fit/nice/untitled'
            ... }
            >>> print(ub.repr2(nh.Initializer.coerce(config)))
            (
                <class 'netharn.initializers.pretrained.Pretrained'>,
                {'fpath': '/fit/nice/untitled'},
            )
            >>> print(ub.repr2(nh.Initializer.coerce({'init': 'kaiming_normal'})))
            (
                <class 'netharn.initializers.core.KaimingNormal'>,
                {'param': 0},
            )
        """
        import netharn as nh
        config = _update_defaults(config, kw)

        pretrained_fpath = config.get('pretrained_fpath', config.get('pretrained', None))
        init = config.get('initializer', config.get('init', None))

        config['init'] = init
        config['pretrained_fpath'] = pretrained_fpath
        config['pretrained'] = pretrained_fpath

        if pretrained_fpath is not None:
            config['init'] = 'pretrained'

        # ---
        initializer_ = None
        if config['init'] == 'kaiming_normal':
            initializer_ = (nh.initializers.KaimingNormal, {
                # initialization params should depend on your choice of
                # nonlinearity in your model. See the Kaiming Paper for details.
                'param': 1e-2 if config.get('noli', 'relu') == 'leaky_relu' else 0,
            })
        elif config['init'] == 'noop':
            initializer_ = (nh.initializers.NoOp, {})
        elif config['init'] == 'pretrained':
            initializer_ = (nh.initializers.Pretrained, {
                'fpath': ub.truepath(config['pretrained_fpath'])})
        elif config['init'] == 'cls':
            # Indicate that the model will initialize itself
            # We have to trust that the user does the right thing here.
            pass
        else:
            raise KeyError(config['init'])
        return initializer_


class Optimizer(object):

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords:
            optimizer / optim
            learning_rate / lr
            weight_decay / decay
        """
        import netharn as nh
        _update_defaults(config, kw)
        key = config.get('optimizer', config.get('optim', 'sgd')).lower()
        lr = config.get('learning_rate', config.get('lr', 3e-3))
        decay = config.get('weight_decay', config.get('decay', 0))
        if key == 'sgd':
            optim_ = (torch.optim.SGD, {
                'lr': lr,
                'weight_decay': decay,
                'momentum': 0.9,
                'nesterov': True,
            })
        elif key == 'adam':
            optim_ = (torch.optim.Adam, {
                'lr': lr,
                'weight_decay': decay,
            })
        elif key == 'adamw':
            optim_ = (nh.optimizers.AdamW, {
                'lr': lr,
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

    def step_batch(self, bx=None):
        raise NotImplementedError

    def step_epoch(self, epoch=None):
        raise NotImplementedError

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords:
            scheduler / schedule
            learning_rate / lr

            for scheduler == exponential:
                gamma
                stepsize
        """
        import netharn as nh
        config = _update_defaults(config, kw)
        key = config.get('scheduler', config.get('schedule', 'onecycle70')).lower()
        lr = config.get('learning_rate', config.get('lr', 3e-3))
        if key.startswith('onecycle'):
            subkey = ''.join(key.split('onecycle')[1:])
            subkey = int(subkey)
            scheduler_ = (nh.schedulers.ListedScheduler, {
                'points': {
                    'lr': {
                        subkey * 0   : lr * 0.1,
                        subkey // 2  : lr * 1.0,
                        subkey * 1   : lr * 0.01,
                        subkey + 1   : lr * 0.0001,
                    },
                    'momentum': {
                        subkey * 0   : 0.95,
                        subkey // 2  : 0.85,
                        subkey * 1   : 0.98,
                        subkey + 1   : 0.999,
                    },
                },
            })
        elif key.startswith('step'):
            subkey = ''.join(key.split('step')[1:])
            subkey = int(subkey)
            scheduler_ = (nh.schedulers.ListedScheduler, {
                'points': {
                    'lr': {
                        subkey * 0   : lr * 1.0,
                        subkey * 1   : lr * 0.1,
                        subkey * 1.5 : lr * 0.01,
                        subkey * 2.0 : lr * 0.001,
                    },
                    # 'momentum': {
                    #     subkey * 0   : 0.95,
                    #     subkey * 1   : 0.98,
                    #     subkey * 1.5 : 0.99,
                    #     subkey * 2.0 : 0.999,
                    # },
                },
                'interpolation': 'left'
            })
        elif key.lower() == 'ReduceLROnPlateau'.lower():
            scheduler_ = (torch.optim.lr_scheduler.ReduceLROnPlateau, {})
        elif key.lower() == 'Exponential'.lower():
            scheduler_ = (nh.schedulers.Exponential, {
                'gamma': config.get('gamma', 0.1),
                'stepsize': config.get('stepsize', 100),
            })
        else:
            raise KeyError
        return scheduler_


class Loaders(object):

    @staticmethod
    def coerce(datasets, config={}, **kw):
        config = _update_defaults(config, kw)
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


class Criterion(object):

    @staticmethod
    def coerce(config={}, **kw):
        """
        Accepts keywords:
            criterion / loss - one of: (
                contrastive, focal, triplet, cross_entropy, mse)
        """
        raise NotImplementedError


def configure_hacks(config={}, **kw):
    """
    Configures hacks to fix global settings in external modules

    Args:
        config (dict): exected to contain they key "workers" with an
           integer value equal to the number of dataloader processes.
        **kw: can also be used to specify config items

    Modules we currently hack:
        * cv2 - fix thread count
    """
    config = _update_defaults(config, kw)
    if config['workers'] > 0:
        import cv2
        cv2.setNumThreads(0)


def configure_workdir(config={}, **kw):
    config = _update_defaults(config, kw)
    if config['workdir'] is None:
        config['workdir'] = kw['workdir']
    workdir = config['workdir'] = ub.truepath(config['workdir'])
    ub.ensuredir(workdir)
    return workdir


def _update_defaults(config, kw):
    config = dict(config)
    for k, v in kw.items():
        if k not in config:
            config[k] = v
    return config
