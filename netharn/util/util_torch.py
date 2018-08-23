# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch


class ModuleMixin(object):
    """
    Adds convenince functions to a torch module
    """
    def number_of_parameters(self, trainable=True):
        return number_of_parameters(self, trainable)


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


class DisableBatchNorm(object):
    def __init__(self, model, enabled=True):
        self.model = model
        self.enabled = enabled
        self.previous_state = None

    def __enter__(self):
        if self.enabled:
            self.previous_state = {}
            for name, layer in trainable_layers(self.model, names=True):
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    self.previous_state[name] = layer.training
                    layer.training = False
        return self

    def __exit__(self, *args):
        if self.previous_state:
            for name, layer in trainable_layers(self.model, names=True):
                if name in self.previous_state:
                    layer.training = self.previous_state[name]


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
