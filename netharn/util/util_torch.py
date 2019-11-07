# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
import six
import ubelt as ub


class ModuleMixin(object):
    """
    Adds convenience functions to a torch module
    """
    def number_of_parameters(self, trainable=True):
        return number_of_parameters(self, trainable)

    def _device_dict(self):
        return {key: item.device for key, item in self.state_dict().items()}

    def devices(self):
        """
        Returns all devices this module state is mounted on
        """
        state_devices = self._device_dict()
        return set(state_devices.values())


def number_of_parameters(model, trainable=True):
    """
    Returns number of trainable parameters in a torch module

    Example:
        >>> import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> number_of_parameters(model)
        824
    """
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params


class grad_context(object):
    """
    Context manager for controlling if autograd is enabled.
    """
    def __init__(self, flag):
        if tuple(map(int, torch.__version__.split('.')[0:2])) < (0, 4):
            self.prev = None
            self.flag = flag
        else:
            self.prev = torch.is_grad_enabled()
            self.flag = flag

    def __enter__(self):
        if self.prev is not None:
            torch.set_grad_enabled(self.flag)

    def __exit__(self, *args):
        if self.prev is not None:
            torch.set_grad_enabled(self.prev)
            return False


def _get_method_func(method):
    func = method.im_func if six.PY2 else method.__func__
    return func


def _get_method_base_class(method):
    """
    Finds the class in which a particular method function was defined.

    CommandLine:
        xdoctest -m netharn.util.util_torch _get_method_base_class

    Example:
        >>> method = torch.nn.BatchNorm2d(1).forward
        >>> print(_get_method_base_class(method))
        <class 'torch.nn.modules.batchnorm._BatchNorm'>
    """
    import sys
    if six.PY2:
        from qualname import qualname as find_qualname
        qualname = find_qualname(method)
    else:
        qualname = method.__qualname__
    module = sys.modules[method.__module__]
    base_name = qualname.split('.')[0]
    base_cls = getattr(module, base_name)
    return base_cls


class IgnoreLayerContext(object):
    """
    Context manager that modifies (monkey-patches) models to temporarily
    remove the forward pass for particular layers.

    Args:
        model (torch.nn.Module): model to modify

        category (type): the module class to be ignored

        enabled (bool, default=True): if True this context manager is enabled
            otherwise it does nothing (i.e. the specified layers will not be
            ignored).

    Example:
        >>> input = torch.rand(1, 1, 10, 10)
        >>> model = torch.nn.BatchNorm2d(1)
        >>> output1 = model(input)
        >>> with IgnoreLayerContext(model, torch.nn.BatchNorm2d):
        ...     output2 = model(input)
        >>> output3 = model(input)
        >>> assert torch.all(output3 == output1)
        >>> assert torch.all(output2 == input)

    Ignore:
        >>> # Test issue with data parallel
        >>> from netharn.util.util_torch import *
        >>> import torch
        >>> import netharn as nh
        >>> layer = raw_model = torch.nn.BatchNorm2d(1)
        >>> raw_inputs = torch.rand(8, 1, 10, 10)
        >>> xpu = nh.XPU.coerce([0,1])
        >>> model = xpu.mount(raw_model)
        >>> inputs = xpu.move(raw_inputs)
        >>> output1 = model(inputs)
        >>> with nh.util.IgnoreLayerContext(model, torch.nn.BatchNorm2d):
        ...     print('model.module.forward = {!r}'.format(model.module.forward))
        ...     output2 = model(inputs)
        >>> output3 = model(inputs)
        >>> assert torch.all(output3 == output1)
        >>> assert torch.all(output2 == inputs)
        >>> # ------------
        >>> raw_model = torch.nn.BatchNorm2d(1)
        >>> raw_inputs = torch.rand(8, 1, 10, 10)
        >>> model = xpu.mount(raw_model)
        >>> inputs = xpu.move(raw_inputs)
        >>> output1 = model(inputs)
        >>> self = nh.util.IgnoreLayerContext(model, torch.nn.BatchNorm2d)
        >>> self.__enter__()
        >>> output2 = model(inputs)
        >>> self.__exit__()
        >>> print('CAN WE DO THIS?')
        >>> output3 = model(inputs)
        >>> # ------------
        >>> xpu = nh.XPU.coerce([0,1])
        >>> devices = [torch.device(type='cuda', index=i) for i in [0, 1]]
        >>> replicas = torch.nn.parallel.replicate(xpu.move(raw_model), devices)
        >>> [r.forward for r in replicas]
        >>> print([r.weight for r in replicas])
        >>> r = replicas[1]
        >>> # ------
        >>> assert torch.all(output3 == output1)
        >>> assert torch.all(output2 == inputs)
    """
    def __init__(self, model, category=None, enabled=True):
        self.model = model
        self.category = category
        self.prev_state = None

        self._PATCH_CLASS = True  # are we patching the instance or class?

    def __enter__(self):
        self.prev_state = {}
        def _noop_forward(self, inputs, *args, **kwargs):
            return inputs
        _noop_forward._patched = True

        for name, layer in trainable_layers(self.model, names=True):
            needs_filter = False
            if self.category is not None:
                needs_filter |= isinstance(layer, self.category)

            if needs_filter:
                if self._PATCH_CLASS:
                    func = _get_method_func(layer.forward)
                    already_patched = not getattr(func, '_patched', False)
                    if already_patched:
                        # Patch the entire class if it wasn't already
                        base_cls = _get_method_base_class(layer.forward)
                        assert 'forward' in base_cls.__dict__
                        # print('PATCH FORWARD IN base_cls = {!r}'.format(base_cls))
                        # print('base_cls = {!r}'.format(base_cls))
                        self.prev_state[name] = (base_cls, base_cls.forward)
                        base_cls.forward = _noop_forward
                else:
                    self.prev_state[name] = layer.forward
                    ub.inject_method(layer, _noop_forward, name='forward')
        return self

    def __exit__(self, *args):
        if self.prev_state:
            if self._PATCH_CLASS:
                # Unpatch all patched classes
                for name, state in self.prev_state.items():
                    base_cls, orig = state
                    base_cls.forward = orig
            else:
                for name, layer in trainable_layers(self.model, names=True):
                    if name in self.prev_state:
                        # Unset the instance attribute that overrides the default
                        # class function attribute. Note that we cannot simply
                        # reset the forward attribute to its old value because that
                        # will still leave an entry in the layer.__dict__ that
                        # previously wasn't there. Having the forward method
                        # populated in layer.__dict__ causes issues with data
                        # parallel.
                        del layer.__dict__['forward']


class BatchNormContext(object):
    """
    Sets batch norm state of `model` to `enabled` within the context manager.

    Args:
        model (torch.nn.Module): model to modify

        training (bool, default=False):
            if True training of batch norm layers is enabled otherwise it is
            disabled. This is useful for batches of size 1.
    """
    def __init__(self, model, training=True, **kw):
        self.model = model
        if kw:
            import warnings
            warnings.warn('the enabled kwarg is depricated')
            training = kw.pop('enabled', training)
            if len(kw):
                raise ValueError('Unsupported kwargs: {}'.format(list(kw)))
        self.training = training
        self.prev_train_state = None

    def __enter__(self):
        self.prev_train_state = {}
        for name, layer in trainable_layers(self.model, names=True):
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                self.prev_train_state[name] = layer.training
                layer.training = self.training
        return self

    def __exit__(self, *args):
        if self.prev_train_state:
            for name, layer in trainable_layers(self.model, names=True):
                if name in self.prev_train_state:
                    layer.training = self.prev_train_state[name]


DisableBatchNorm = BatchNormContext


def trainable_layers(model, names=False):
    """
    Example:
        >>> import torchvision
        >>> model = torchvision.models.AlexNet()
        >>> list(trainable_layers(model, names=True))
    """
    if names:
        stack = [('', '', model)]
        while stack:
            prefix, basename, item = stack.pop()
            name = '.'.join([p for p in [prefix, basename] if p])
            if isinstance(item, torch.nn.modules.conv._ConvNd):
                yield name, item
            elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
                yield name, item
            elif hasattr(item, 'reset_parameters'):
                yield name, item

            child_prefix = name
            for child_basename, child_item in list(item.named_children())[::-1]:
                stack.append((child_prefix, child_basename, child_item))
    else:
        queue = [model]
        while queue:
            item = queue.pop(0)
            # TODO: need to put all trainable layer types here
            # (I think this is just everything with reset_parameters)
            if isinstance(item, torch.nn.modules.conv._ConvNd):
                yield item
            elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
                yield item
            elif hasattr(item, 'reset_parameters'):
                yield item
            # if isinstance(input, torch.nn.modules.Linear):
            #     yield item
            # if isinstance(input, torch.nn.modules.Bilinear):
            #     yield item
            # if isinstance(input, torch.nn.modules.Embedding):
            #     yield item
            # if isinstance(input, torch.nn.modules.EmbeddingBag):
            #     yield item
            for child in item.children():
                queue.append(child)


def one_hot_embedding(labels, num_classes, dtype=None):
    """
    Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].

    References:
        https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4

    CommandLine:
        python -m netharn.loss one_hot_embedding

    Example:
        >>> # each element in target has to have 0 <= value < C
        >>> labels = torch.LongTensor([0, 0, 1, 4, 2, 3])
        >>> num_classes = max(labels) + 1
        >>> t = one_hot_embedding(labels, num_classes)
        >>> assert all(row[y] == 1 for row, y in zip(t.numpy(), labels.numpy()))
        >>> import ubelt as ub
        >>> print(ub.repr2(t.numpy().tolist()))
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
        >>> t2 = one_hot_embedding(labels.numpy(), num_classes)
        >>> assert np.all(t2 == t.numpy())
        >>> if torch.cuda.is_available():
        >>>     t3 = one_hot_embedding(labels.to(0), num_classes)
        >>>     assert np.all(t3.cpu().numpy() == t.numpy())
    """
    if isinstance(labels, np.ndarray):
        dtype = dtype or np.float
        y = np.eye(num_classes, dtype=dtype)
        y_onehot = y[labels]
    else:  # if torch.is_tensor(labels):
        dtype = dtype or torch.float
        y = torch.eye(num_classes, device=labels.device, dtype=dtype)
        y_onehot = y[labels]
    return y_onehot


def one_hot_lookup(probs, labels):
    """
    Return probbility of a particular label (usually true labels) for each item

    Each item in labels corresonds to a row in probs. Returns the index
    specified at each row.

    Example:
        >>> probs = np.array([
        >>>     [0, 1, 2],
        >>>     [3, 4, 5],
        >>>     [6, 7, 8],
        >>>     [9, 10, 11],
        >>> ])
        >>> labels = np.array([0, 1, 2, 1])
        >>> one_hot_lookup(probs, labels)
        array([ 0,  4,  8, 10])
    """
    return probs[np.eye(probs.shape[1], dtype=np.bool)[labels]]
