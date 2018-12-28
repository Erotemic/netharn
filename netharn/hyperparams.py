# -*- coding: utf-8 -*-
"""
Torch version of hyperparams

TODO:
    [ ] - need to extract relavent params from loaders
    [ ] - need to extract relavent params from datasets
    [ ] - ensure monitor is handled gracefully
    [ ] - prevent non-relevant params from being used in the hash

CommandLine:
    python ~/code/netharn/netharn/hyperparams.py __doc__

Example:
    >>> import netharn as nh
    >>> datasets = {
    >>>     'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
    >>>     'vali': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
    >>> }
    >>> hyper = nh.HyperParams(**{
    >>>     # --- Data First
    >>>     'datasets'    : datasets,
    >>>     'nice'        : 'demo',
    >>>     'loaders'     : {'batch_size': 64},
    >>>     'xpu'         : nh.XPU.cast('auto'),
    >>>     # --- Algorithm Second
    >>>     'model'       : (nh.models.ToyNet2d, {}),
    >>>     'optimizer'   : (nh.optimizers.SGD, {
    >>>         'lr': 0.0001
    >>>     }),
    >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
    >>>     #'criterion'   : (nh.criterions.FocalLoss, {}),
    >>>     'initializer' : (nh.initializers.KaimingNormal, {
    >>>         'param': 0,
    >>>     }),
    >>>     'scheduler'   : (nh.schedulers.ListedLR, {
    >>>         'points': {0: .0001, 2: .01, 5: .015, 6: .005, 9: .001},
    >>>     }),
    >>>     'monitor'     : (nh.Monitor, {
    >>>         'max_epoch': 10
    >>>     }),
    >>> })
    >>> print(ub.repr2(hyper.get_initkw()))
    >>> print(ub.repr2(hyper.hyper_id()))


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import platform
import warnings
from os.path import join
from os.path import normpath
from os.path import sys
import numpy as np
import ubelt as ub
import torch
import six
from netharn import util
from netharn import initializers
from netharn import device
from collections import OrderedDict
# from netharn import criterions
from torch.optim.optimizer import required
import torch.utils.data as torch_data


try:
    import imgaug
    Augmenter = imgaug.augmenters.meta.Augmenter
except ImportError:
    imgaug = None


def _hash_data(data):
    return ub.hash_data(data, hasher='sha512', base='abc', types=True)


def _rectify_class(lookup, arg, kw):
    if arg is None:
        return None, {}
    if lookup is None:
        lookup = ub.identity

    if isinstance(arg, tuple):
        cls = lookup(arg[0])
        try:
            kw2 = arg[1]
        except Exception:
            print('lookup = {!r}'.format(lookup))
            print('arg = {!r}'.format(arg))
            raise
    else:
        cls = lookup(arg)
        kw2 = {}

    cls_kw = _class_default_params(cls).copy()
    cls_kw.update(kw2)
    for key in cls_kw:
        if key in kw:
            cls_kw[key] = kw.pop(key)
    return cls, cls_kw


def _class_default_params(cls):
    """
    cls = torch.optim.Adam
    """
    if six.PY2:
        import funcsigs
        sig = funcsigs.signature(cls)
    else:
        import inspect
        sig = inspect.signature(cls)
    default_params = {
        k: p.default
        for k, p in sig.parameters.items()
        if p.default is not p.empty
    }
    return default_params


def _rectify_criterion(arg, kw):
    if arg is None:
        # arg = 'CrossEntropyLoss'
        return None, None

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                # criterions.ContrastiveLoss,
                torch.nn.CrossEntropyLoss,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


def _rectify_optimizer(arg, kw):
    if arg is None:
        arg = 'SGD'
        if kw is None:
            kw = {}
        kw = kw.copy()
        if 'lr' not in kw:
            kw['lr'] = .001

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                torch.optim.Adam,
                torch.optim.SGD,
            ]
            cls = {c.__name__.lower(): c for c in options}[arg.lower()]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)

    for k, v in kw2.items():
        if v is required:
            raise ValueError('Must specify {} for {}'.format(k, cls))

    return cls, kw2


def _rectify_lr_scheduler(arg, kw):
    if arg is None:
        return None, None
        # arg = 'Constant'

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                torch.optim.lr_scheduler.LambdaLR,
                torch.optim.lr_scheduler.StepLR,
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ExponentialLR,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


def _rectify_initializer(arg, kw):
    if arg is None:
        arg = 'NoOp'
        # arg = 'CrossEntropyLoss'
        # return None, None

    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
                initializers.KaimingNormal,
                initializers.NoOp,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


def _rectify_monitor(arg, kw):
    def _lookup(arg):
        if isinstance(arg, six.string_types):
            options = [
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls
    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


def _rectify_dynamics(arg, kw):
    """
    Special params that control the dynamics of learning at the harness level
    at point that doesnt correspond to a decoupled class component.
    """
    if arg is None:
        arg = {}
    arg = arg.copy()
    dynamics = {
        # batch_step simulates larger batch sizes
        'batch_step': arg.pop('batch_step', 1),
        # Clips gradients
        'grad_norm_max': arg.pop('grad_norm_max', None),
    }
    if not isinstance(dynamics['batch_step'], int):
        raise ValueError('batch_step must be an integer')
    if arg:
        raise KeyError('UNKNOWN dynamics: {}'.format(arg))
    return dynamics


def _rectify_model(arg, kw):
    if arg is None:
        return None, None

    def _lookup_model(arg):
        import torchvision
        if isinstance(arg, six.string_types):
            options = [
                torchvision.models.AlexNet,
                torchvision.models.DenseNet,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup_model, arg, kw)
    return cls, kw2


def _rectify_loaders(arg, kw):
    """
    Loaders are handled slightly differently than other classes
    We construct them eagerly (if they are not already constructed)
    """
    if arg is None:
        arg = {}

    loaders = None

    if isinstance(arg, dict):
        # Check if all args are already data loaders
        # if isinstance(arg.get('train', None), torch_data.DataLoader):
        if len(arg) and all(isinstance(v, torch_data.DataLoader) for v in arg.values()):
            # loaders were custom specified
            loaders = arg
            # TODO: extract relevant loader params efficiently
            cls = None
            if 'train' in loaders:
                kw2 = {
                    'batch_size': loaders['train'].batch_sampler.batch_size,
                }
            else:
                kw2 = {}
        else:
            # loaders is kwargs for `torch_data.DataLoader`
            arg = (torch_data.DataLoader, arg)
            cls, kw2 = _rectify_class(None, arg, kw)
    else:
        raise ValueError('Loaders should be a dict')

    kwnice = ub.dict_subset(kw2, ['batch_size'], default=None)
    return loaders, cls, kw2, kwnice


class HyperParams(object):
    """
    Holds hyperparams relavent to training strategy

    The idea is that you tell it what is relevant FOR YOU, and then it makes
    you nice ids based on that. If you give if enough info it also allows you
    to use the training harness.

    CommandLine:
        python -m netharn.hyperparams HyperParams

    Example:
        >>> from netharn.hyperparams import *
        >>> hyper = HyperParams(
        >>>     criterion=('CrossEntropyLoss', {
        >>>         'weight': torch.FloatTensor([0, 2, 1]),
        >>>     }),
        >>>     optimizer=(torch.optim.SGD, {
        >>>         'nesterov': True, 'weight_decay': .0005,
        >>>         'momentum': 0.9, 'lr': .001,
        >>>     }),
        >>>     scheduler=('ReduceLROnPlateau', {}),
        >>> )
        >>> # xdoctest: +IGNORE_WANT
        >>> print(hyper.hyper_id())
        NoOp,SGD,dampening=0,lr=0.001,momentum=0.9,nesterov=True,weight_decay=0.0005,ReduceLROnPlateau,cooldown=0,eps=1e-08,factor=0.1,min_lr=0,mode=min,patience=10,threshold=0.0001,threshold_mode=rel,verbose=False,CrossEntropyLoss,ignore_index=-100,reduce=None,reduction=mean,size_average=None,weight=[0.0,2.0,1.0],DataLoader,batch_size=1,Dynamics,batch_step=1,grad_norm_max=None
    """

    def __init__(hyper,
                 # ----
                 datasets=None,
                 nice=None,
                 workdir=None,
                 xpu=None,
                 loaders=None,
                 # ----
                 model=None,
                 criterion=None,
                 optimizer=None,
                 initializer=None,
                 scheduler=None,
                 # ---
                 dynamics=None,
                 monitor=None,
                 augment=None,
                 other=None,  # incorporated into the hash
                 extra=None,  # ignored when computing the hash
                 ):
        kwargs = {}

        hyper.datasets = datasets
        hyper.nice = nice
        hyper.workdir = workdir
        hyper.xpu = xpu

        loaders, cls, kw, kwnice = _rectify_loaders(loaders, kwargs)
        hyper.loaders = loaders
        hyper.loader_cls = cls
        hyper.loader_params = kw
        hyper.loader_params_nice = kwnice

        cls, kw = _rectify_model(model, kwargs)
        hyper.model_cls = cls
        hyper.model_params = kw

        cls, kw = _rectify_optimizer(optimizer, kwargs)
        hyper.optimizer_cls = cls
        hyper.optimizer_params = kw

        cls, kw = _rectify_lr_scheduler(scheduler, kwargs)
        hyper.scheduler_cls = cls
        hyper.scheduler_params = kw

        cls, kw = _rectify_criterion(criterion, kwargs)
        hyper.criterion_cls = cls
        hyper.criterion_params = kw

        cls, kw = _rectify_initializer(initializer, kwargs)
        hyper.initializer_cls = cls
        hyper.initializer_params = kw

        cls, kw = _rectify_monitor(monitor, kwargs)
        hyper.monitor_cls = cls
        hyper.monitor_params = kw

        hyper.dynamics = _rectify_dynamics(dynamics, kw)

        hyper.augment = augment
        hyper.other = other
        hyper.extra = extra

    def make_model(hyper):
        """ Instanciate the model defined by the hyperparams """
        model = hyper.model_cls(**hyper.model_params)
        return model

    def make_optimizer(hyper, parameters):
        """ Instanciate the optimizer defined by the hyperparams """
        # What happens if we want to group parameters
        optimizer = hyper.optimizer_cls(parameters, **hyper.optimizer_params)
        return optimizer

    def make_scheduler(hyper, optimizer):
        """ Instanciate the lr scheduler defined by the hyperparams """
        if hyper.scheduler_cls is None:
            return None
        kw = hyper.scheduler_params.copy()
        kw['optimizer'] = optimizer
        scheduler = hyper.scheduler_cls(**kw)
        return scheduler

    def make_initializer(hyper):
        """ Instanciate the initializer defined by the hyperparams """
        initializer = hyper.initializer_cls(**hyper.initializer_params)
        return initializer

    def make_criterion(hyper):
        """ Instanciate the criterion defined by the hyperparams """
        if hyper.criterion_cls is None:
            return None
        criterion = hyper.criterion_cls(**hyper.criterion_params)
        return criterion

    def make_loaders(hyper):
        if hyper.loaders is not None:
            return hyper.loaders
        else:
            loaders = {
                key: torch_data.DataLoader(dset, **hyper.loader_params)
                for key, dset in hyper.datasets.items()
            }
        return loaders

    def make_xpu(hyper):
        """ Instanciate the criterion defined by the hyperparams """
        xpu = device.XPU.cast(hyper.xpu)
        return xpu

    def make_monitor(hyper):
        """ Instanciate the monitor defined by the hyperparams """
        if hyper.monitor_cls is None:
            return None
        monitor = hyper.monitor_cls(**hyper.monitor_params)
        return monitor

    def other_id(hyper):
        """
            >>> from netharn.hyperparams import *
            >>> hyper = HyperParams(other={'augment': True, 'n_classes': 10, 'n_channels': 5})
            >>> hyper.hyper_id()
        """
        otherid = util.make_short_idstr(hyper.other, precision=4)
        return otherid

    def get_initkw(hyper):
        """
        Make list of class / params relevant to reproducing an experiment

        CommandLine:
            python ~/code/netharn/netharn/hyperparams.py HyperParams.get_initkw

        Example:
            >>> from netharn.hyperparams import *
            >>> hyper = HyperParams(
            >>>     criterion='CrossEntropyLoss',
            >>>     optimizer='Adam',
            >>>     loaders={'batch_size': 64},
            >>> )
            >>> print(ub.repr2(hyper.get_initkw()))
        """
        initkw = OrderedDict()
        def _append_part(key, cls, params):
            """
            append an id-string derived from the class and params.
            TODO: what if we have an instance and not a cls/params tuple?
            """
            if cls is None:
                initkw[key] = None
            else:
                d = OrderedDict()
                for k, v in sorted(params.items()):
                    # if k in total:
                    #     raise KeyError(k)
                    if isinstance(v, torch.Tensor):
                        v = v.numpy()
                    if isinstance(v, np.ndarray):
                        if v.dtype.kind == 'f':
                            try:
                                v = list(map(float, v))
                            except Exception:
                                v = v.tolist()
                        else:
                            raise NotImplementedError()
                    d[k] = v
                    # total[k] = v
                if isinstance(cls, six.string_types):
                    type_str = cls
                else:
                    modname = cls.__module__
                    type_str = modname + '.' + cls.__name__
                # param_str = util.make_idstr(d)
                initkw[key] = (type_str, d)

        _append_part('model', hyper.model_cls, hyper.model_params)
        _append_part('initializer', hyper.initializer_cls, hyper.initializer_params)
        _append_part('optimizer', hyper.optimizer_cls, hyper.optimizer_params)
        _append_part('scheduler', hyper.scheduler_cls, hyper.scheduler_params)
        _append_part('criterion', hyper.criterion_cls, hyper.criterion_params)

        # TODO: should other be included in initkw? I think it should.
        # probably should also include monitor, xpu, nice

        # Loader is a bit hacked
        _append_part('loader', hyper.loader_cls, hyper.loader_params_nice)
        _append_part('dynamics', 'Dynamics', hyper.dynamics)

        return initkw

    def augment_json(hyper):
        """
        Get augmentation info in json format

        Example:
            >>> from netharn.hyperparams import *
            >>> import imgaug
            >>> augment = imgaug.augmenters.Affine()
            >>> hyper = HyperParams(augment=augment)
            >>> info = hyper.augment_json()
            >>> assert info['__class__'] == 'Affine'
            >>> hyper = HyperParams(augment=OrderedDict())
            >>> assert hyper.augment_json() == {}
        """
        if hyper.augment is None:
            return None
        elif imgaug is not None and isinstance(hyper.augment, imgaug.augmenters.Augmenter):
            from netharn.data.transforms.augmenter_base import ParamatarizedAugmenter
            augment_json = ParamatarizedAugmenter._json_id(hyper.augment)
        elif isinstance(hyper.augment, six.string_types):
            return hyper.augment
        # Some classes in imgaug (e.g. Sequence) inherit from list,
        # so we have to check for Augmenter before we check for list type
        # if isinstance(hyper.augment, (dict, list)):
        elif isinstance(hyper.augment, OrderedDict):
            # dicts are specified in json format
            try:
                # hashable data should be loosely json-compatible
                _hash_data(hyper.augment)
            except TypeError:
                raise TypeError(
                    'NOT IN ORDERED JSON FORMAT hyper.augment={}'.format(
                        hyper.augment))
            augment_json = hyper.augment
        else:
            raise TypeError('Specify augment in json format')
        return augment_json

    def input_id(hyper, short=False, hashed=False):
        pass

    def _parts_id(hyper, parts, short=False, hashed=False):
        id_parts = []
        for key, value in parts.items():
            if value is None:
                continue
            clsname, params = value
            type_str = clsname.split('.')[-1]
            id_parts.append(type_str)

            # Precidence of specifications (from lowest to highest)
            # SF=single flag, EF=explicit flag
            # SF-short, SF-hash, EF-short EF-hash
            request_short = short is True
            request_hash = hashed is True
            if (ub.iterable(short) and key in short):
                request_hash = False
                request_short = True
            if (ub.iterable(hashed) and key in hashed):
                request_hash = True
                request_short = False

            if request_hash:
                param_str = util.make_idstr(params)
                param_str = _hash_data(param_str)[0:6]
            elif request_short:
                param_str = util.make_short_idstr(params)
            else:
                param_str = util.make_idstr(params)

            if param_str:
                id_parts.append(param_str)
        idstr = ','.join(id_parts)
        return idstr

    def hyper_id(hyper, short=False, hashed=False):
        """
        Identification string that uniquely determined by training hyper.
        Suitable for hashing.

        CommandLine:
            python -m netharn.hyperparams HyperParams.hyper_id

        Example:
            >>> from netharn.hyperparams import *
            >>> hyper = HyperParams(criterion='CrossEntropyLoss', other={'n_classes': 10, 'n_channels': 5})
            >>> print(hyper.hyper_id())
            >>> print(hyper.hyper_id(short=['optimizer']))
            >>> print(hyper.hyper_id(short=['optimizer'], hashed=True))
            >>> print(hyper.hyper_id(short=['optimizer', 'criterion'], hashed=['criterion']))
            >>> print(hyper.hyper_id(hashed=True))
        """
        parts = hyper.get_initkw()
        return hyper._parts_id(parts, short, hashed)

    def train_info(hyper, train_dpath=None):
        """
        Create json metadata that details enough information such that it would
        be possible for a human to reproduce the experiment.

        Example:
            >>> import netharn as nh
            >>> datasets = {
            >>>     'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
            >>>     'vali': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
            >>> }
            >>> hyper = nh.hyperparams.HyperParams(**{
            >>>     # --- Data First
            >>>     'datasets'    : datasets,
            >>>     'nice'        : 'demo',
            >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/demo'),
            >>>     'loaders'     : {'batch_size': 64},
            >>>     'xpu'         : nh.XPU.cast('auto'),
            >>>     # --- Algorithm Second
            >>>     'model'       : (nh.models.ToyNet2d, {}),
            >>>     'optimizer'   : (nh.optimizers.SGD, {
            >>>         'lr': 0.001
            >>>     }),
            >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
            >>>     #'criterion'   : (nh.criterions.FocalLoss, {}),
            >>>     'initializer' : (nh.initializers.KaimingNormal, {
            >>>         'param': 0,
            >>>     }),
            >>>     'scheduler'   : (nh.schedulers.ListedLR, {
            >>>         'step_points': {0: .001, 2: .01, 5: .015, 6: .005, 9: .001},
            >>>         'interpolate': True,
            >>>     }),
            >>>     'monitor'     : (nh.Monitor, {
            >>>         'max_epoch': 10
            >>>     }),
            >>> })
            >>> info = hyper.train_info()
            >>> print(ub.repr2(info))
        """
        given_explicit_train_dpath = train_dpath is not None
        # TODO: needs MASSIVE cleanup and organization

        # TODO: if pretrained is another netharn model, then we should read that
        # train_info if it exists and append it to a running list of train_info

        if hyper.model_cls is None:
            # import utool
            # utool.embed()
            raise ValueError('model_cls is None')
        # arch = hyper.model_cls.__name__

        train_dset = hyper.datasets.get('train', None)
        if train_dset is not None and hasattr(train_dset, 'input_id'):
            input_id = train_dset.input_id
            if callable(input_id):
                input_id = input_id()
        else:
            warnings.warn(
                'FitHarn cannot track the training dataset state because '
                'harn.datasets["train"] is missing the "input_id" attribute.'
            )
            input_id = 'none'

        def _hash_data(data):
            return ub.hash_data(data, hasher='sha512', base='abc', types=True)

        train_hyper_id_long = hyper.hyper_id()
        short = True
        hashed = True
        train_hyper_id_brief = hyper.hyper_id(short=short, hashed=hashed)
        train_hyper_hashid = _hash_data(train_hyper_id_long)[:8]

        # TODO: hash this to some degree
        other_id = hyper.other_id()

        augment_json = hyper.augment_json()

        aug_brief = 'AU' + _hash_data(augment_json)[0:6]
        # extra_hash = _hash_data([hyper.centering])[0:6]

        train_id = '{}_{}_{}_{}'.format(
            _hash_data(input_id)[:6], train_hyper_id_brief,
            aug_brief, other_id)

        # Gather all information about this run into a single hash
        train_hashid = _hash_data(train_id)[0:8]

        nice = hyper.nice

        nice_dpath = None
        if not given_explicit_train_dpath:
            # setup a cannonical and a linked symlink dir
            train_dpath = normpath(
                    join(hyper.workdir, 'fit', 'runs', nice, train_hashid))
            # also setup a "nice" custom name, which may conflict, but oh well
            if nice:
                try:
                    nice_dpath = normpath(
                            join(hyper.workdir, 'fit', 'nice', nice))
                except Exception:
                    print('hyper.workdir = {!r}'.format(hyper.workdir))
                    print('hyper.nice = {!r}'.format(hyper.nice))
                    raise

        # make temporary initializer so we can infer the history
        temp_initializer = hyper.make_initializer()
        init_history = temp_initializer.history()

        train_info =  ub.odict([
            ('train_hashid', train_hashid),

            ('train_id', train_id),

            ('workdir', hyper.workdir),

            ('aug_brief', aug_brief),

            ('input_id', input_id),

            ('other_id', other_id),

            ('hyper', hyper.get_initkw()),

            ('train_hyper_id_long', train_hyper_id_long),
            ('train_hyper_id_brief', train_hyper_id_brief),
            ('train_hyper_hashid', train_hyper_hashid),
            ('init_history', init_history),
            ('init_history_hashid', _hash_data(util.make_idstr(init_history))),

            ('nice', hyper.nice),

            ('old_train_dpath', normpath(
                join(hyper.workdir, 'fit', 'runs', train_hashid))),

            ('train_dpath', train_dpath),
            # ('link_dpath', link_dpath),
            ('nice_dpath', nice_dpath),

            ('given_explicit_train_dpath', given_explicit_train_dpath),

            # TODO, add in n_classes if applicable
            # TODO, add in centering if applicable
            # ('centering', hyper.centering),

            ('other', hyper.other),

            # HACKED IN
            ('augment', hyper.augment_json()),

            ('extra', hyper.extra),

            ('argv', sys.argv),
            ('hostname', platform.node()),
        ])
        return train_info

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.hyperparams
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
