import numpy as np
import ubelt as ub
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
