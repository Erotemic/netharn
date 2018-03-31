# -*- coding: utf-8 -*-
"""
Torch version of hyperparams
"""
import numpy as np
import ubelt as ub
import torch
import six
from clab import util
from clab import criterions
from clab import nninit
from torch.optim.optimizer import required


def _rectify_class(lookup, arg, kw):
    if arg is None:
        return None, {}

    if isinstance(arg, tuple):
        cls = lookup(arg[0])
        kw2 = arg[1]
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
                criterions.CrossEntropyLoss2D,
                criterions.ContrastiveLoss,
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
                nninit.HeNormal,
                nninit.NoOp,
            ]
            cls = {c.__name__: c for c in options}[arg]
        else:
            cls = arg
        return cls

    cls, kw2 = _rectify_class(_lookup, arg, kw)
    return cls, kw2


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


class HyperParams(object):
    """
    Holds hyperparams relavent to training strategy

    The idea is that you tell it what is relevant FOR YOU, and then it makes
    you nice ids based on that. If you give if enough info it also allows you
    to use the training harness.

    CommandLine:
        python -m clab.hyperparams HyperParams

    Example:
        >>> from clab.hyperparams import *
        >>> hyper = HyperParams(
        >>>     criterion=('CrossEntropyLoss2D', {
        >>>         'weight': [0, 2, 1],
        >>>     }),
        >>>     optimizer=(torch.optim.SGD, {
        >>>         'nesterov': True, 'weight_decay': .0005,
        >>>         'momentum': 0.9, 'lr': .001,
        >>>     }),
        >>>     scheduler=('ReduceLROnPlateau', {}),
        >>> )
        >>> print(hyper.hyper_id())
    """

    def __init__(hyper, criterion=None, optimizer=None, scheduler=None,
                 model=None, other=None, initializer=None,

                 # TODO: give hyper info about the inputs
                 centering=None, augment=None, train=None, vali=None,
                 **kwargs):

        cls, kw = _rectify_model(model, kwargs)
        hyper.model_cls = cls
        hyper.model_params = kw

        cls, kw = _rectify_optimizer(optimizer, kwargs)
        hyper.optimizer_cls = cls
        hyper.optimizer_params = kw
        # hyper.optimizer_params.pop('lr', None)  # hack

        cls, kw = _rectify_lr_scheduler(scheduler, kwargs)
        hyper.scheduler_cls = cls
        hyper.scheduler_params = kw

        # What if multiple criterions are used?
        cls, kw = _rectify_criterion(criterion, kwargs)
        hyper.criterion_cls = cls
        hyper.criterion_params = kw

        cls, kw = _rectify_initializer(initializer, kwargs)
        hyper.initializer_cls = cls
        hyper.initializer_params = kw

        # set an identifier based on the input train dataset
        hyper.input_ids = {}  # TODO

        hyper.train = train
        hyper.vali = vali

        hyper.augment = augment
        hyper.centering = centering

        if len(kwargs) > 0:
            raise ValueError('Unused kwargs {}'.format(list(kwargs.keys())))

        hyper.other = other
    # def _normalize(hyper):
    #     """
    #     normalize for hashid generation
    #     """
    #     weight = hyper.criterion_params.get('weight', None)
    #     if weight is not None:
    #         weight = list(map(float, weight))
    #         hyper.criterion_params['weight'] = weight

    def make_model(hyper):
        """ Instanciate the model defined by the hyperparams """
        model = hyper.model_cls(**hyper.model_params)
        return model

    def make_optimizer(hyper, parameters):
        """ Instanciate the optimizer defined by the hyperparams """
        optimizer = hyper.optimizer_cls(parameters, **hyper.optimizer_params)
        return optimizer

    def make_scheduler(hyper, optimizer):
        """ Instanciate the lr scheduler defined by the hyperparams """
        scheduler = hyper.scheduler_cls(optimizer, **hyper.scheduler_params)
        return scheduler

    def make_initializer(hyper):
        """ Instanciate the initializer defined by the hyperparams """
        initializer = hyper.initializer_cls(**hyper.initializer_params)
        return initializer

    def make_criterion(hyper):
        """ Instanciate the criterion defined by the hyperparams """
        # NOTE: for some problems a crition may not be defined here
        criterion = hyper.criterion_cls(**hyper.criterion_params)
        return criterion

    # def model_id(hyper, brief=False):
    #     """
    #     CommandLine:
    #         python -m clab.hyperparams HyperParams.model_id

    #     Example:
    #         >>> from clab.hyperparams import *
    #         >>> hyper = HyperParams(model='DenseNet', optimizer=('SGD', dict(lr=.001)))
    #         >>> print(hyper.model_id())
    #         >>> hyper = HyperParams(model='AlexNet', optimizer=('SGD', dict(lr=.001)))
    #         >>> print(hyper.model_id())
    #         >>> print(hyper.hyper_id())
    #         >>> hyper = HyperParams(model='AlexNet', optimizer=('SGD', dict(lr=.001)), scheduler='ReduceLROnPlateau')
    #         >>> print(hyper.hyper_id())
    #     """
    #     arch = hyper.model_cls.__name__
    #     # TODO: add model as a hyperparam specification
    #     # archkw = _class_default_params(hyper.model_cls)
    #     # archkw.update(hyper.model_params)
    #     archkw = hyper.model_params
    #     if brief:
    #         arch_id = arch + ',' + util.hash_data(util.make_idstr(archkw))[0:8]
    #     else:
    #         # arch_id = arch + ',' + util.make_idstr(archkw)
    #         arch_id = arch + ',' + util.make_short_idstr(archkw)
    #     return arch_id

    def other_id(hyper):
        """
            >>> from clab.hyperparams import *
            >>> hyper = HyperParams(other={'augment': True, 'n_classes': 10, 'n_channels': 5})
            >>> hyper.hyper_id()
        """
        otherid = util.make_short_idstr(hyper.other, precision=4)
        return otherid

    def augment_json(hyper):
        """
        Get augmentation info in json format

        Example:
            >>> from clab.hyperparams import *
            >>> import imgaug as ia
            >>> import imgaug.augmenters as iaa
            >>> import imgaug
            >>> augment = imgaug.augmenters.Affine()
            >>> hyper = HyperParams(augment=augment)
            >>> info = hyper.augment_json()
            >>> assert info['__class__'] == 'Affine'
            >>> hyper = HyperParams(augment=ub.odict())
            >>> assert hyper.augment_json() == {}
            >>> #####
            >>> augmentors = [
            >>>     iaa.Fliplr(p=.5),
            >>>     iaa.Flipud(p=.5),
            >>>     iaa.Affine(
            >>>         scale={"x": (1.0, 1.01), "y": (1.0, 1.01)},
            >>>         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            >>>         rotate=(-15, 15),
            >>>         shear=(-7, 7),
            >>>         order=[0, 1, 3],
            >>>         cval=(0, 255),
            >>>         mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            >>>         # Note: currently requires imgaug master version
            >>>         backend='cv2',
            >>>     ),
            >>>     iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
            >>>     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
            >>> ]
            >>> augment = iaa.Sequential(augmentors)
            >>> hyper = HyperParams(augment=augment)
            >>> info = hyper.augment_json()
        """
        if hyper.augment is None:
            return None
        elif isinstance(hyper.augment, str):
            return hyper.augment
        # if isinstance(hyper.augment, (dict, list)):  # cant check for list because Seq inherits from it
        elif isinstance(hyper.augment, ub.odict):
            # already specified in json format
            try:
                ub.hash_data(hyper.augment)
            except TypeError:
                raise TypeError('NOT IN ORDERED JSON FORMAT hyper.augment={}'.format(hyper.augment))
            augment_json = hyper.augment
        else:
            try:
                import imgaug

                def imgaug_json_id(aug):
                    # TODO: submit a PR to imgaug that registers parameters
                    # with classes
                    if isinstance(aug, tuple):
                        return [imgaug_json_id(item) for item in aug]
                    elif isinstance(aug, imgaug.parameters.StochasticParameter):
                        return str(aug)
                    else:
                        try:
                            info = ub.odict()
                            info['__class__'] = aug.__class__.__name__
                            params = aug.get_parameters()
                            if params:
                                info['params'] = [imgaug_json_id(p) for p in params]
                            if isinstance(aug, list):
                                children = aug[:]
                                children = [imgaug_json_id(c) for c in children]
                                info['children'] = children
                            return info
                        except Exception:
                            # imgaug is weird and buggy
                            return str(aug)
            except ImportError:
                augment_json = str(hyper.augment)
            else:
                if isinstance(hyper.augment, imgaug.augmenters.meta.Augmenter):
                    augment_json = imgaug_json_id(hyper.augment)
                else:
                    raise TypeError('Specify augment in json format')
                try:
                    ub.hash_data(augment_json)
                except TypeError:
                    print('FAILED TO PRODUCE JSON FORMAT for hyper.augment={}'.format(hyper.augment))
                    raise
        return augment_json
        # print(ub.repr2(augment))

    def get_initkw(hyper):
        initkw = ub.odict()
        def _append_part(key, cls, params):
            """
            append an id-string derived from the class and params.
            TODO: what if we have an instance and not a cls/params tuple?
            """
            if cls is None:
                initkw[key] = None
                return
            d = ub.odict()
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
            modname = cls.__module__
            type_str = modname + '.' + cls.__name__
            # param_str = util.make_idstr(d)
            initkw[key] = (type_str, d)
        _append_part('model', hyper.model_cls, hyper.model_params)
        _append_part('initializer', hyper.initializer_cls, hyper.initializer_params)
        _append_part('optimizer', hyper.optimizer_cls, hyper.optimizer_params)
        _append_part('scheduler', hyper.scheduler_cls, hyper.scheduler_params)
        _append_part('criterion', hyper.criterion_cls, hyper.criterion_params)
        return initkw

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
                param_str = util.hash_data(param_str)[0:6]
            elif request_short:
                param_str = util.make_short_idstr(params)
            else:
                param_str = util.make_idstr(params)

            if param_str:
                id_parts.append(param_str)
        idstr = ','.join(id_parts)
        return idstr

    # def other_id2(hyper, short=False, hashed=False):
    #     """
    #     Example:
    #         >>> from clab.hyperparams import *
    #         >>> hyper = HyperParams(criterion='CrossEntropyLoss', aug='foobar', train='idfsds')
    #         >>> hyper.other_id2(hashed=True)
    #     """
    #     parts = ub.odict([
    #         ('aug', ('aug', hyper.augment)),
    #         ('train', ('train', hyper.train)),
    #         ('vali', ('vali', hyper.vali)),
    #     ])
    #     idstr = hyper._parts_id(parts, short, hashed)
    #     return idstr

    def hyper_id(hyper, short=False, hashed=False):
        """
        Identification string that uniquely determined by training hyper.
        Suitable for hashing.

        CommandLine:
            python -m clab.hyperparams HyperParams.hyper_id

        Example:
            >>> from clab.hyperparams import *
            >>> hyper = HyperParams(criterion='CrossEntropyLoss', other={'n_classes': 10, 'n_channels': 5})
            >>> print(hyper.hyper_id())
            >>> print(hyper.hyper_id(short=['optimizer']))
            >>> print(hyper.hyper_id(short=['optimizer'], hashed=True))
            >>> print(hyper.hyper_id(short=['optimizer', 'criterion'], hashed=['criterion']))
            >>> print(hyper.hyper_id(hashed=True))
        """
        # hyper._normalize()

        parts = hyper.get_initkw()
        return hyper._parts_id(parts, short, hashed)

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.hyperparams
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
