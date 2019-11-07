import torch.optim.lr_scheduler


class CommonMixin:

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    # def get_lr(self):
    #     raise NotImplementedError

    def current_lrs(self):
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        return lrs


class TorchNetharnScheduler(CommonMixin, torch.optim.lr_scheduler._LRScheduler):
    """
    Fixes call to epoch 0 twice

    See:
        https://github.com/pytorch/pytorch/issues/8837
    """
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, torch.optim.lr_scheduler.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch
        self.step(last_epoch)  # The major change is to remove the +1

    @property
    def epoch(self):
        return self.last_epoch + 1

    def step(self, epoch=None):
        # epoch is really last epoch
        if epoch is None:
            epoch = self.epoch
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    # def get_lr(self):
    #     raise NotImplementedError

    # def step(self, epoch=None):
    #     if epoch is None:
    #         epoch = self.last_epoch + 1
    #     self.last_epoch = epoch
    #     for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
    #         param_group['lr'] = lr


class NetharnScheduler(CommonMixin):
    pass


class YOLOScheduler(NetharnScheduler):
    """
    Scheduler that changs learning rates on a per-ITERATION level
    """
    def __init__(self, optimizer, last_epoch=-1, batch_num=0, dset_size=1, base_lr=0.001):
        super().__init__(self)
        self.base_lr = base_lr
        self.epoch = last_epoch + 1
        self.batch_num = batch_num
        self.dset_size = dset_size
        self.optimizer = optimizer

    def epoch(self):
        return self.batch_num / self.dset_size

    def get_lr(self):
        """ Return the current LR """
        if self.batch_num < 1000:
            progress = (self.batch_num / self.dset_size) * 0.1
        lr = self.base_lr * progress ** 4
        return lr

    @property
    def batch_per_epoch(self):
        return self.dset_size / self.batch_num

    def step_batch(self, batch_num=None):

        self.batch_num = batch_num
        return
        pass

    def step_epoch(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        self.epoch += 1
        self.batch_num = int(self.batch_per_epoch) * self.epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
