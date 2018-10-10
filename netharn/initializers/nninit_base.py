from torch.autograd import Variable
import numpy as np
import torch
import ubelt as ub
from netharn.util.util_torch import trainable_layers


class _BaseInitializer(object):
    """
    """
    def __call__(self, model, *args, **kwargs):
        self.forward(model, *args, **kwargs)

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


class NoOp(_BaseInitializer):
    """
    Example:
        >>> from netharn.initializers import *
        >>> self = NoOp()
        >>> #info = self.history()
        >>> #assert info['__name__'] == 'NoOp'
    """
    def forward(self, model):
        return


def apply_initializer(input, func, funckw):
    """
    Args:
        input: can be a model, layer, or tensor

    Example:
        >>> from torch import nn
        >>> class DummyNet(nn.Module):
        >>>     def __init__(self, n_channels=1, n_classes=10):
        >>>         super(DummyNet, self).__init__()
        >>>         self.conv = nn.Conv2d(n_channels, 10, kernel_size=5)
        >>>         self.norm = nn.BatchNorm2d(10)
        >>> model = DummyNet()
        >>> func = nn.init.constant_
        >>> apply_initializer(model, func, {'val': 42})
        >>> assert np.all(model.conv.weight.detach().numpy() == 42)
        >>> assert np.all(model.conv.bias.detach().numpy() == 0), 'bias is always init to zero'
        >>> assert np.all(model.norm.bias.detach().numpy() == 0), 'bias is always init to zero'
        >>> assert np.all(model.norm.weight.detach().numpy() == 1)
        >>> assert np.all(model.norm.running_mean.detach().numpy() == 0.0)
        >>> assert np.all(model.norm.running_var.detach().numpy() == 1.0)
    """
    if getattr(input, 'bias', None) is not None:
        # zero all biases
        input.bias.data.zero_()

    if isinstance(input, (Variable, torch.Tensor)):
        # assert False, ('input is tensor? does this make sense?')
        func(input, **funckw)
        # data = input
    elif isinstance(input, (torch.nn.modules.conv._ConvNd)):
        func(input.weight, **funckw)
    elif isinstance(input, torch.nn.modules.batchnorm._BatchNorm):
        input.reset_parameters()
        # always initialize batch norm weights to 1
        torch.nn.init.constant_(input.weight, 1.0)
        torch.nn.init.constant_(input.bias, 0.0)
    # elif isinstance(input, torch.nn.modules.Linear):
    #     input.reset_parameters()
    elif hasattr(input, 'reset_parameters'):
        input.reset_parameters()
    else:
        # input is a torch module
        model = input
        for item in trainable_layers(model):
            apply_initializer(item, func, funckw)


def load_partial_state(model, model_state_dict, initializer=None,
                       ignore_unset=False, verbose=2):
    """
    CommandLine:
        python -m netharn.initializers.nninit_base load_partial_state

    Example:
        >>> import netharn as nh
        >>> self1 = nh.models.ToyNet2d(input_channels=1, num_classes=10)
        >>> self2 = nh.models.ToyNet2d(input_channels=3, num_classes=2)
        >>> model_state_dict = self1.state_dict()
        >>> load_partial_state(self2, model_state_dict)

    Example:
        >>> import netharn as nh
        >>> xpu = nh.XPU(None)
        >>> self1 = nh.models.ToyNet2d()
        >>> self2 = xpu.mount(self1)
        >>> load_partial_state(self2, self1.state_dict())
        >>> load_partial_state(self1, self2.state_dict())
    """
    self_state = model.state_dict()

    def _fix_keys(model_state_dict):
        """
        Hack around DataParallel wrapper. If there is nothing in common between
        the two models check to see if prepending 'module.' to other keys fixes
        it.
        """
        other_keys = set(model_state_dict)
        self_keys = set(self_state)

        if not other_keys.intersection(self_keys):
            prefix = 'module.'
            def smap(f, ss):
                return set(map(f, ss))
            def fix1(k):
                return prefix + k
            def fix2(k):
                if k.startswith(prefix):
                    return k[len(prefix):]
            if smap(fix1, other_keys).intersection(self_keys):
                model_state_dict = ub.map_keys(fix1, model_state_dict)
            elif smap(fix2, other_keys).intersection(self_keys):
                model_state_dict = ub.map_keys(fix2, model_state_dict)

        return model_state_dict

    other_state = _fix_keys(model_state_dict)

    self_unset_keys = set(self_state.keys())  # will end up as keys in our that were not set
    other_unused_keys = set(other_state.keys())  # will end up as keys in the other model that were not used

    seen_keys = ub.ddict(set)

    for key, other_value in other_state.items():
        if key not in self_state:
            print('Skipping {} because it does not exist'.format(key))
            seen_keys['skipped'].add(key)
        else:
            self_value = self_state[key]
            if other_value.size() == self_value.size():
                self_state[key] = other_value
                self_unset_keys.remove(key)
                other_unused_keys.remove(key)
                seen_keys['full_add'].add(key)
            elif len(other_value.size()) == len(self_value.size()):
                if key.endswith('bias'):
                    print('Skipping {} due to incompatable size'.format(key))
                    print(' * self  = {!r}'.format(self_value.size()))
                    print(' * other = {!r}'.format(other_value.size()))
                    seen_keys['skipped'].add(key)
                else:
                    if initializer is None:
                        print('Skipping {} due to incompatable size and no default initializer'.format(key))
                        print(' * self  = {!r}'.format(self_value.size()))
                        print(' * other = {!r}'.format(other_value.size()))
                        seen_keys['skipped'].add(key)
                    else:
                        print('Partially add {} with incompatable size'.format(key))
                        print(' * self  = {!r}'.format(self_value.size()))
                        print(' * other = {!r}'.format(other_value.size()))
                        # Initialize all weights in case any are unspecified
                        if initializer is not None:
                            initializer(self_state[key])

                        # Transfer as much as possible
                        min_size = np.minimum(self_state[key].shape, other_value.shape)
                        sl = tuple([slice(0, s) for s in min_size])
                        self_state[key][sl] = other_value[sl]

                        # if shock_partial:
                        #     # Shock weights because we are doing something weird
                        #     # might help the network recover in case this is
                        #     # not a good idea
                        #     shock(self_state[key], func=initializer)
                        self_unset_keys.remove(key)
                        other_unused_keys.remove(key)
                        seen_keys['partial_add'].add(key)
            else:
                print('Skipping {} due to incompatable size'.format(key))
                print(' * self  = {!r}'.format(self_value.size()))
                print(' * other = {!r}'.format(other_value.size()))
                seen_keys['skipped'].add(key)

    if ignore_unset is True:
        self_unset_keys = []
    elif ignore_unset:
        self_unset_keys = list(ub.oset(self_unset_keys) - set(ignore_unset))

    if self_unset_keys or other_unused_keys:
        print('Seen Keys: {}'.format(ub.repr2(seen_keys, nl=2)))

        print('Self Unset Keys: {}'.format(ub.repr2(self_unset_keys, nl=1)))

        print('Other Unused keys: {}'.format(ub.repr2(other_unused_keys, nl=1)))
        if initializer:
            print('Initializing unused keys using {}'.format(initializer))
            for key in self_unset_keys:
                if key.endswith('.bias'):
                    self_state[key].fill_(0)
                else:
                    initializer(self_state[key])
    else:
        if verbose > 1:
            print('Pretrained weights are a perfect fit')
    model.load_state_dict(self_state)

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.initializers.nninit_base all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
