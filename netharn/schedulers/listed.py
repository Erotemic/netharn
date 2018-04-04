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
        points (dict): Mapping from epoch number to a learning rate
        last_epoch (int): The index of last epoch. Default: -1.

    CommandLine:
        python ~/code/netharn/netharn/schedulers/listed.py ListedLR

    Example:
        >>> # Assuming optimizer has two groups.
        >>> import ubelt as ub
        >>> import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0)
        >>> points = {0: .01, 2: .02, 3: .1, 6: .05, 9: .025}
        >>> self = ListedLR(optimizer, points)
        >>> lrs = [self._get_epoch_lr(epoch) for epoch in range(0, 11)]
        >>> print(list(ub.flatten(lrs)))
        [0.01, 0.01, 0.02, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]
        >>> assert self.current_lrs() == [0.01]
        >>> self = ListedLR(optimizer, points, interpolate=True)
        >>> lrs = [self._get_epoch_lr(epoch) for epoch in range(0, 11)]
        >>> print(ub.repr2(list(ub.flatten(lrs)), precision=3, nl=0))
        [0.010, 0.015, 0.020, 0.100, 0.083, 0.067, 0.050, 0.042, 0.033, 0.025, 0.025]
    """

    def __init__(self, optimizer, points, interpolate=False,
                 last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(points, dict):
            raise TypeError(points)
        self.interpolate = interpolate
        self.points = points
        self.last_epoch = last_epoch
        # epochs where the lr changes
        self.key_epochs = sorted(self.points.keys())
        super(ListedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_epoch_lr(self.last_epoch)

    def _get_epoch_lr(self, epoch):
        """ return lr based on the epoch """
        key_epochs = self.key_epochs
        points = self.points
        base_lrs = self.base_lrs

        if epoch < 0:
            epoch = 0

        if epoch in key_epochs:
            prev_key_epoch = epoch
            next_key_epoch = epoch
        else:
            idx = np.searchsorted(key_epochs, epoch, 'left') - 1
            prev_key_epoch = key_epochs[idx]
            if idx < len(key_epochs) - 1:
                next_key_epoch = key_epochs[idx + 1]
            else:
                next_key_epoch = prev_key_epoch

        if self.interpolate:
            if next_key_epoch == prev_key_epoch:
                new_lr = points[next_key_epoch]
            else:
                prev_lr = points[next_key_epoch]
                next_lr = points[prev_key_epoch]

                alpha = (epoch - prev_key_epoch) / (next_key_epoch - prev_key_epoch)

                new_lr = alpha * prev_lr + (1 - alpha) * next_lr

            epoch_lrs = [new_lr for _ in base_lrs]
        else:
            if epoch < prev_key_epoch:
                epoch_lrs = base_lrs
            else:
                new_lr = points[prev_key_epoch]
                epoch_lrs = [new_lr for _ in base_lrs]
        return epoch_lrs

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.schedulers.listed all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
