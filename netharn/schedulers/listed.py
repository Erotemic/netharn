import numpy as np
import torch.optim.lr_scheduler


class _LRScheduler2(torch.optim.lr_scheduler._LRScheduler):
    """ Add a bit of extra functionality to the base torch LR scheduler """

    def current_lrs(self):
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        return lrs


class ListedLR(_LRScheduler2):
    """
    Simple scheduler that simply sets the LR based on the epoch.

    Allows for hard-coded schedules for quick prototyping. Good for reproducing
    papers, but bad for experimentation.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_points (dict): Mapping from epoch number to a learning rate
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> from clab.lr_scheduler import *
        >>> import torchvision
        >>> model = torchvision.models.SqueezeNet()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0)
        >>> step_points = {0: .01, 1: .02, 2: .1, 6: .05, 9: .025}
        >>> self = ListedLR(optimizer, step_points)
        >>> lrs = [self._get_epoch_lr(epoch) for epoch in range(-1, 11)]
        >>> import ubelt as ub
        >>> print(list(ub.flatten(lrs)))
        [0, 0.01, 0.02, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]
        >>> assert self.current_lrs() == [0.01]
        >>> assert self.base_lrs() == [0]
    """

    def __init__(self, optimizer, step_points, last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(step_points, dict):
            raise TypeError(step_points)
        self.step_points = step_points
        self.last_epoch = last_epoch
        # epochs where the lr changes
        self.key_epochs = sorted(self.step_points.keys())
        super(ListedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_epoch_lr(self.last_epoch)

    # def step(self):
    #     print('STEPING = {!r}'.format(self))
    #     super().step()

    def _get_epoch_lr(self, epoch):
        """ return lr based on the epoch """
        key_epochs = self.key_epochs
        step_points = self.step_points
        base_lrs = self.base_lrs

        if epoch in key_epochs:
            key_epoch = epoch
        else:
            idx = np.searchsorted(key_epochs, epoch, 'left') - 1
            key_epoch = key_epochs[idx]
        if epoch < key_epoch:
            epoch_lrs = base_lrs
        else:
            new_lr = step_points[key_epoch]
            epoch_lrs = [new_lr for _ in base_lrs]
        return epoch_lrs
