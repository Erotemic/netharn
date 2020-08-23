import numpy as np
import torch
import ubelt as ub


def trainable_layers(model, names=False):
    """
    Returns all layers containing trainable parameters

    Notes:
        It may be better to simply use model.named_parameters() instead in most
        situation. This is useful when you need the classes that contains the
        parameters instead of the parameters themselves.

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


def apply_initializer(input, func, funckw):
    """
    Recursively initializes the input using a torch.nn.init function.

    If the input is a model, then only known layer types are initialized.

    Args:
        input (Tensor | Module): can be a model, layer, or tensor
        func (callable): initialization function
        funckw (dict):

    Example:
        >>> from torch import nn
        >>> import torch
        >>> class DummyNet(nn.Module):
        >>>     def __init__(self, n_channels=1, n_classes=10):
        >>>         super(DummyNet, self).__init__()
        >>>         self.conv = nn.Conv2d(n_channels, 10, kernel_size=5)
        >>>         self.norm = nn.BatchNorm2d(10)
        >>>         self.param = torch.nn.Parameter(torch.rand(3))
        >>> self = DummyNet()
        >>> func = nn.init.kaiming_normal_
        >>> apply_initializer(self, func, {})
        >>> func = nn.init.constant_
        >>> apply_initializer(self, func, {'val': 42})
        >>> assert np.all(self.conv.weight.detach().numpy() == 42)
        >>> assert np.all(self.conv.bias.detach().numpy() == 0), 'bias is always init to zero'
        >>> assert np.all(self.norm.bias.detach().numpy() == 0), 'bias is always init to zero'
        >>> assert np.all(self.norm.weight.detach().numpy() == 1)
        >>> assert np.all(self.norm.running_mean.detach().numpy() == 0.0)
        >>> assert np.all(self.norm.running_var.detach().numpy() == 1.0)
    """
    if getattr(input, 'bias', None) is not None:
        # print('zero input bias')
        # zero all biases
        input.bias.data.zero_()

    if isinstance(input, (torch.Tensor)):
        # assert False, ('input is tensor? does this make sense?')
        # print('input is tensor')
        func(input, **funckw)
        # data = input
    elif isinstance(input, (torch.nn.modules.conv._ConvNd)):
        # print('input is convnd')
        func(input.weight, **funckw)
    # elif isinstance(input, (torch.nn.modules.linear.Linear)):
    #     func(input.weight, **funckw)
    elif isinstance(input, torch.nn.modules.batchnorm._BatchNorm):
        # Use default batch norm
        input.reset_parameters()
    # elif isinstance(input, torch.nn.modules.Linear):
    #     input.reset_parameters()
    elif hasattr(input, 'reset_parameters'):
        # print('unknown input type fallback on reset_params')
        input.reset_parameters()
    else:
        # input is a torch module
        model = input
        # print('recurse input')
        layers = list(trainable_layers(model))
        # print('layers = {!r}'.format(layers))
        for item in layers:
            apply_initializer(item, func, funckw)


def load_partial_state(model, model_state_dict, leftover=None,
                       ignore_unset=False, verbose=2,
                       mangle=True, association=None,
                       initializer=None):
    """
    CommandLine:
        python -m netharn.initializers.nninit_base load_partial_state

    Args:
        model (torch.nn.Module): module to initialize

        model_state_dict (dict): state dict we wish to transfer

        leftover (callable): fallback method for initializing incompatible
             areas, if none then those areas are left as-is.

        association (str): controls how we search for the association between
            the two model states. Can be strict, module-hack, prefix-hack, or
            embedding.  Default is: prefix-hack.

        mangle (bool, default=True): If True, mangles tensors that have the
            same key, but different shapes forcing them to fit. This might
            destroy information when forcing a a larger tensor into a smaller
            tensor, or leave extra uninitialized room when a small tensor is
            placed in a larger one. Note be careful when mangling a
            classification layer if class indexes are not aligned.

        verbose (int): verbosity level

    Returns:
        Dict: info - summary of actions taken

    TODO:
        - [ ] Allow user to specify how incompatible layers are handled.

    Notes:

        Have you ever had the scenario where

        Has anyone ever had a problem where you had a torch model with a state
        dict with keys that looked like: `mymodel.detector.layer1.conv.weight`,
        but you had a pretrained weight file with keys that looked like:
        `module.layer1.conv.weight`?

        The latest version of
        `netharn.initializers.functional.load_patial_state` can handle this by
        solving a maximum-common-subtree-isomorphism problem. This computes the
        largest possible mapping between the two state dictionaries that share
        consistent suffixes.

        >>> # This means you can load an off-the-shelf unmodified pretrained resnet50
        >>> # where the keys might look something like this:
        >>> resnet_keys = {
        >>>     'conv1.weight',
        >>>     'layer1.0.conv1.weight',
        >>>     'layer1.0.conv2.weight',
        >>>     'layer1.0.conv3.weight',
        >>>     'layer1.0.downsample.0.weight',
        >>>     'layer2.0.conv1.weight',
        >>>     'layer2.0.conv2.weight',
        >>>     'layer2.0.conv3.weight',
        >>>     'layer3.0.conv1.weight',
        >>>     'layer4.0.conv1.weight',
        >>>     'fc.weight',
        >>>     'fc.bias',
        >>> }
        >>> #
        >>> # And perhaps you have a model that has a state dict where keys
        >>> # look like this:
        >>> model_keys = {
        >>>     'preproc.conv1.weight'
        >>>     'backbone.layer1.0.conv1.weight',
        >>>     'backbone.layer1.0.conv2.weight',
        >>>     'backbone.layer1.0.conv3.weight',
        >>>     'backbone.layer1.0.downsample.0.weight',
        >>>     'backbone.layer2.0.conv1.weight',
        >>>     'backbone.layer2.0.conv2.weight',
        >>>     'backbone.layer2.0.conv3.weight',
        >>>     'backbone.layer3.0.conv1.weight',
        >>>     'backbone.layer4.0.conv1.weight',
        >>>     'head.conv1'
        >>>     'head.conv2'
        >>>     'head.fc.weight'
        >>>     'head.fc.bias'
        >>> }
        >>> #
        >>> # We can compute a partial mapping between them
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(resnet_keys, model_keys)
        >>> print(ub.repr2(ub.dzip(subpaths1, subpaths2)))
        {
            'layer1.0.conv2.weight':        'backbone.layer1.0.conv2.weight',
            'layer1.0.conv3.weight':        'backbone.layer1.0.conv3.weight',
            'layer1.0.downsample.0.weight': 'backbone.layer1.0.downsample.0.weight',
            'layer2.0.conv1.weight':        'backbone.layer2.0.conv1.weight',
            'layer2.0.conv2.weight':        'backbone.layer2.0.conv2.weight',
            'layer2.0.conv3.weight':        'backbone.layer2.0.conv3.weight',
            'layer3.0.conv1.weight':        'backbone.layer3.0.conv1.weight',
            'layer4.0.conv1.weight':        'backbone.layer4.0.conv1.weight',
        }

        Also, if the sizes of the tensor don't quite fit, they will be
        mangled, i.e. "shoved-in" as best as possible.


    Example:
        >>> import netharn as nh
        >>> self1 = nh.models.ToyNet2d(input_channels=1, num_classes=10)
        >>> self2 = nh.models.ToyNet2d(input_channels=3, num_classes=2)
        >>> self1.hack_param1 = torch.nn.Parameter(torch.rand(1))
        >>> self2.hack_param1 = torch.nn.Parameter(torch.rand(3))
        >>> self2.hack_param2 = torch.nn.Parameter(torch.rand(3))
        >>> model_state_dict = self1.state_dict()
        >>> load_partial_state(self2, model_state_dict)
        >>> load_partial_state(self2, model_state_dict, leftover=torch.nn.init.kaiming_normal_)

    Example:
        >>> import netharn as nh
        >>> xpu = nh.XPU(None)
        >>> self1 = nh.models.ToyNet2d()
        >>> self2 = xpu.mount(self1)
        >>> load_partial_state(self2, self1.state_dict())
        >>> load_partial_state(self1, self2.state_dict())
        >>> # Add extra nonsense to state-dict
        >>> extra_state_dict = {'extra.' + k: v for k, v in self1.state_dict().items()}
        >>> extra_state_dict['stats'] = ub.peek(extra_state_dict.values()).clone()
        >>> model = self2
        >>> model_state_dict = extra_state_dict
        >>> load_partial_state(self2, extra_state_dict)

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> from netharn.initializers.functional import *  # NOQA
        >>> import torchvision
        >>> import torch
        >>> resnet50 = torchvision.models.resnet50()
        >>> class CustomModel(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.module = resnet50
        >>>         self.extra = torch.nn.Linear(1, 1)
        >>> model = CustomModel()
        >>> model_state_dict = resnet50.state_dict()
        >>> model_state_dict2 = {'prefix.' + k: v for k, v in model_state_dict.items()}
        >>> import ubelt as ub
        >>> with ub.Timer(verbose=2, label='strict'):
        >>>     load_partial_state(model, model_state_dict, association='strict', verbose=0)
        >>> with ub.Timer(verbose=2, label='prefix-hack'):
        >>>     load_partial_state(model, model_state_dict, association='prefix-hack', verbose=0)
        >>> with ub.Timer(verbose=2, label='module-hack'):
        >>>     load_partial_state(model, model_state_dict, association='module-hack', verbose=0)
        >>> with ub.Timer(verbose=2, label='embedding'):
        >>>     load_partial_state(model, model_state_dict, association='embedding', verbose=0)

        >>> load_partial_state(model, model_state_dict, association='prefix-hack', verbose=1)
        >>> load_partial_state(model, model_state_dict, association='module-hack', verbose=1)

    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/functional.py load_partial_state:2 --slow

    """
    if association is None:
        # association = 'module-hack'  # old default
        association = 'prefix-hack'  # new default

    if initializer is not None:
        import warnings
        warnings.warn('initializer is deprecated use leftover')
        leftover = initializer

    self_state = model.state_dict()

    def _fix_keys(model_state_dict):
        """
        Hack around DataParallel wrapper. If there is nothing in common between
        the two models check to see if prepending 'module.' to other keys fixes
        it.
        """
        other_keys = set(model_state_dict)
        self_keys = set(self_state)
        common_keys = other_keys.intersection(self_keys)
        if not common_keys:
            if association == 'strict':
                pass
            elif association == 'module-hack':
                # If there are no common keys try a hack
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
            elif association == 'prefix-hack':
                import functools
                def add_prefix(k, prefix):
                    return prefix + k
                def remove_prefix(k, prefix):
                    if k.startswith(prefix):
                        return k[len(prefix):]
                # set1 = other_keys
                # target_set2 = self_keys
                found = _best_prefix_transform(other_keys, self_keys)
                if found is not None:
                    for action, prefix in found['transform']:
                        if action == 'add':
                            func = functools.partial(add_prefix, prefix=prefix)
                        elif action == 'remove':
                            func = functools.partial(remove_prefix, prefix=prefix)
                        else:
                            raise AssertionError
                        model_state_dict = ub.map_keys(func, model_state_dict)
            elif association == 'embedding':
                if verbose > 1:
                    print('Using subpath embedding assocation, may take some time')
                # I believe this is the correct way to solve the problem
                paths1 = sorted(other_keys)
                paths2 = sorted(self_state)
                subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2)
                mapping = ub.dzip(subpaths1, subpaths2)
                if verbose > 1:
                    print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
                model_state_dict = ub.map_keys(lambda k: mapping.get(k, k), model_state_dict)
            else:
                raise KeyError(association)
        return model_state_dict

    other_state = _fix_keys(model_state_dict)

    self_unset_keys = set(self_state.keys())  # will end up as keys in our that were not set
    other_unused_keys = set(other_state.keys())  # will end up as keys in the other model that were not used

    seen_keys = ub.ddict(set)

    for key, other_value in other_state.items():
        if key not in self_state:
            if verbose > 0:
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
                if not mangle:
                    if verbose > 0:
                        print('Skipping {} due to incompatable size and mangle=False'.format(key))
                        print(' * self  = {!r}'.format(self_value.size()))
                        print(' * other = {!r}'.format(other_value.size()))
                    seen_keys['skipped'].add(key)
                elif key.endswith('bias'):
                    if verbose > 0:
                        print('Skipping {} due to incompatable size'.format(key))
                        print(' * self  = {!r}'.format(self_value.size()))
                        print(' * other = {!r}'.format(other_value.size()))
                    seen_keys['skipped'].add(key)
                else:
                    if leftover is None:
                        if verbose > 0:
                            print('Skipping {} due to incompatable size and no default initializer'.format(key))
                            print(' * self  = {!r}'.format(self_value.size()))
                            print(' * other = {!r}'.format(other_value.size()))
                        seen_keys['skipped'].add(key)
                    else:
                        if verbose > 0:
                            print('Partially add {} with incompatable size'.format(key))
                            print(' * self  = {!r}'.format(self_value.size()))
                            print(' * other = {!r}'.format(other_value.size()))
                        # Initialize all weights in case any are unspecified
                        if leftover is None:
                            try:
                                leftover(self_state[key])
                            except Exception:
                                if verbose > 0:
                                    print('Unable to init {} with {}'.format(key, leftover))

                        # Transfer as much as possible
                        min_size = np.minimum(self_state[key].shape,
                                              other_value.shape)
                        sl = tuple([slice(0, s) for s in min_size])
                        self_state[key][sl] = other_value[sl]

                        # if shock_partial:
                        #     # Shock weights because we are doing something weird
                        #     # might help the network recover in case this is
                        #     # not a good idea
                        #     shock(self_state[key], func=leftover)
                        self_unset_keys.remove(key)
                        other_unused_keys.remove(key)

                        if self_state[key].numel() < other_value.numel():
                            seen_keys['partial_add_some'].add(key)
                        else:
                            seen_keys['partial_add_all'].add(key)
            else:
                if verbose > 0:
                    print('Skipping {} due to incompatable size'.format(key))
                    print(' * self  = {!r}'.format(self_value.size()))
                    print(' * other = {!r}'.format(other_value.size()))
                seen_keys['skipped'].add(key)

    if ignore_unset is True:
        self_unset_keys = []
    elif ignore_unset:
        self_unset_keys = list(ub.oset(self_unset_keys) - set(ignore_unset))

    if (self_unset_keys or other_unused_keys or
         seen_keys['partial_add_some'] or seen_keys['partial_add_all']):
        if verbose > 0:
            if seen_keys:
                print('Pretrained weights are a partial fit')
            else:
                print('Pretrained weights do not fit!')
        if verbose > 1:
            print('Seen Keys: {}'.format(ub.repr2(seen_keys, nl=2)))
            print('Self Unset Keys: {}'.format(ub.repr2(self_unset_keys, nl=1)))
            print('Other Unused keys: {}'.format(ub.repr2(other_unused_keys, nl=1)))
        if leftover:
            if verbose > 0:
                print('Initializing unused keys using {}'.format(leftover))
            for key in self_unset_keys:
                if key.endswith('.num_batches_tracked'):
                    pass  # ignore num_batches_tracked
                elif key.endswith('.bias'):
                    self_state[key].fill_(0)
                else:
                    try:
                        leftover(self_state[key])
                    except Exception:
                        if verbose > 0:
                            print('Unable to init {} with {}'.format(key, leftover))

    else:
        if verbose > 0:
            print('Pretrained weights are a perfect fit')
    model.load_state_dict(self_state)

    info = {
        'seen': seen_keys,
        'self_unset': self_unset_keys,
        'other_unused': other_unused_keys
    }
    return info


def _best_prefix_transform(set1, target_set2):
    """
    Find a way to transform prefixes of items in set1 to match target_set2

    Example:
        >>> set1 = {'mod.f.0.w',
        >>>         'mod.f.1.b',
        >>>         'mod.f.1.n',
        >>>         'mod.f.1.rm',
        >>>         'mod.f.1.rv',}
        >>> #
        >>> target_set2 = {
        >>>      'bar.foo.extra.f.1.b',
        >>>      'bar.foo.extra.f.1.n',
        >>>      'bar.foo.extra.f.1.w',
        >>>      'bar.foo.extra.f.3.w',
        >>> }
        >>> _best_prefix_transform(set1, target_set2)
        >>> target_set2.add('JUNK')
        >>> _best_prefix_transform(set1, target_set2)
    """

    # probably an efficient way to do this with a trie

    # NOTE: In general this is a graph-isomorphism problem or a  maximum common
    # subgraph problem. However, we can look only at the special case of
    # "maximum common subtrees". Given two directory structures (as trees)
    # we find the common bits.
    # https://perso.ensta-paris.fr/~diam/ro/online/viggo_wwwcompendium/node168.html
    # We can approximate to O(log log n / log^2 n)
    # Can get algorithm from maximum independent set
    # https://arxiv.org/abs/1602.07210

    # The most efficient algorithm here would be for solving
    # "Maximum common labeled subtrees"
    # APX-hard for unordered trees, but polytime solveable for ordered trees
    # For directory structures we can induce an order, and hense obtain a
    # polytime solution
    # #
    # On the Maximum Common Embedded Subtree Problem for Ordered Trees
    # https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf

    from os.path import commonprefix
    prefixes1 = commonprefix(list(set1)).split('.')
    prefixes2 = commonprefix(list(target_set2)).split('.')

    # Remove the trailing prefixes that are the same
    num_same = 0
    for i in range(1, min(len(prefixes1), len(prefixes2))):
        if prefixes1[-i] == prefixes2[-i]:
            num_same = i
        else:
            break
    prefixes1 = prefixes1[:-num_same]
    prefixes2 = prefixes2[:-num_same]

    ALLOW_FUZZY = 1
    if ALLOW_FUZZY and len(prefixes2) == 0:
        # SUPER HACK FOR CASE WHERE THERE IS JUST ONE SPOILER ELEMENT IN THE
        # TARGET SET. THE ALGORITHM NEEDS TO BE RETHOUGHT FOR THAT CASE
        possible_prefixes = [k.split('.') for k in target_set2]
        prefix_hist = ub.ddict(lambda: 0)
        for item in possible_prefixes:
            for i in range(1, len(item)):
                prefix_hist[tuple(item[0:i])] += 1
        prefixes2 = ['.'.join(ub.argmax(prefix_hist))]

    def add_prefix(items, prefix):
        return {prefix + k for k in items}
    def remove_prefix(items, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k for k in items}

    import itertools as it
    found_cand = []
    for i1, i2 in it.product(range(len(prefixes1) + 1), range(len(prefixes2) + 1)):
        if i1 == 0 and i2 == 0:
            continue
        # Very inefficient, we should be able to do better
        prefix1 = '.'.join(prefixes1[:i1])
        prefix2 = '.'.join(prefixes2[:i2])
        if prefix1:
            prefix1 = prefix1 + '.'
        if prefix2:
            prefix2 = prefix2 + '.'

        # We are allowed to remove a prefix from a set, add the other
        # prefix to the set, or remove and then add.
        set1_cand1 = remove_prefix(set1, prefix1)
        set1_cand2 = add_prefix(set1, prefix2)
        set1_cand3 = add_prefix(set1_cand1, prefix2)

        common1 = set1_cand1 & target_set2
        common2 = set1_cand2 & target_set2
        common3 = set1_cand3 & target_set2
        if common1:
            found_cand.append({
                'transform': [('remove', prefix1)],
                'value': len(common1),
            })
        if common2:
            found_cand.append({
                'transform': [('add', prefix2)],
                'value': len(common2),
            })
        if common3:
            found_cand.append({
                'transform': [('remove', prefix1), ('add', prefix2)],
                'value': len(common3),
            })
    if len(found_cand):
        found = max(found_cand, key=lambda x: x['value'])
    else:
        found = None
    return found


def maximum_common_ordered_subpaths(paths1, paths2, sep='.'):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/functional.py maximum_common_ordered_subpaths:0 --profile && cat profile_output.txt
        xdoctest -m /home/joncrall/code/netharn/netharn/initializers/functional.py maximum_common_ordered_subpaths:0

    Example:
        >>> import torchvision
        >>> resnet50 = torchvision.models.resnet50()
        >>> paths1 = sorted(resnet50.state_dict().keys())[0:100]
        >>> paths2 = ['prefix.' + k for k in paths1]
        >>> paths2.append('extra_key')
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))

    Example:
        >>> rng = None
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(rng)
        >>> def random_paths(rng, max_depth=10):
        >>>     depth = rng.randint(1, max_depth)
        >>>     parts = list(map(chr, rng.randint(ord('a'), ord('z'), size=depth)))
        >>>     path = '.'.join(parts)
        >>>     return path
        >>> n = 50
        >>> paths1 = sorted({random_paths(rng) for _ in range(n)})
        >>> paths2 = sorted({random_paths(rng) for _ in range(n)})
        >>> paths1 = paths1 + ['a.' + k for k in paths2[0:n // 3]]
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))

    Example:
        >>> from netharn.initializers.functional import *  # NOQA
        >>> paths1 = [
        >>>     'stats',
        >>>     'z.mod.f.0.w',
        >>>     'a.z.mod.f.0.b',
        >>>     'z.mod.f.1.b',
        >>>     'z.mod.f.1.n',
        >>>     'z.mod.f.1.m',
        >>>     'z.mod.f.1.v',
        >>>     'z.mod.f.2.m',
        >>>     'z.mod.z.q'
        >>> ]
        >>> # paths1 = ['mod']
        >>> #
        >>> paths2 = [
        >>>     'stats',
        >>>     'bar.f.0.w',
        >>>     'bar.foo.extra.z.q',
        >>>     'bar.foo.extra',
        >>>     'bar.foo.extra.f.1.b',
        >>>     'bar.foo.extra.f.1.n',
        >>>     'bar.foo.extra.f.1.w',
        >>>     'bar.foo.extra.f.3.z',  # FIXME we need to handle label comparision operators
        >>>     # I think we allow labels to match if they have the same suffix
        >>> ]
        >>> sep = '.'
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))


    Example:
        >>> sep = '.'
        >>> paths1 = ['a.b']
        >>> paths2 = ['a.b']
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
        >>> paths1 = ['c.a.b']
        >>> paths2 = ['a.b']
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
        >>> paths1 = ['c.a.b', 'c.a.e', 'c.a.q']
        >>> paths2 = ['a.b', 'c.e', 'c.a', 'a.q']
        >>> subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2, sep)
        >>> mapping = ub.dzip(subpaths1, subpaths2)
        >>> print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
    """
    import networkx as nx

    # the longest common balanced sequence problem
    def _affinity(tok1, tok2):
        score = 0
        for t1, t2 in zip(tok1[::-1], tok2[::-1]):
            if t1 == t2:
                score += 1
            else:
                break
        return score
        # return tok1[-1] == tok2[-1]
    node_affinity = _affinity
    # import operator
    # eq = operator.eq

    def paths_to_tree(paths):
        tree = nx.OrderedDiGraph()
        for path in sorted(paths):
            parts = tuple(path.split(sep))
            node_path = []
            for i in range(1, len(parts) + 1):
                node = parts[0:i]
                tree.add_node(node)
                tree.nodes[node]['label'] = node[-1]
                node_path.append(node)
            for u, v in ub.iter_window(node_path, 2):
                tree.add_edge(u, v)
        return tree

    tree1 = paths_to_tree(paths1)
    tree2 = paths_to_tree(paths2)

    # _print_forest(tree1)
    # _print_forest(tree2)

    # if 0:
    #     DiGM = isomorphism.DiGraphMatcher(tree1, tree2)
    #     DiGM.is_isomorphic()
    #     list(DiGM.subgraph_isomorphisms_iter())

    from netharn.initializers import _nx_extensions
    subtree1, subtree2 = _nx_extensions.maximum_common_ordered_tree_embedding(tree1, tree2, node_affinity=node_affinity)
    # subtree1, subtree2 = _nx_extensions.maximum_common_ordered_subtree_isomorphism(tree1, tree2, node_affinity=node_affinity)

    subpaths1 = [sep.join(node) for node in subtree1.nodes if subtree1.out_degree[node] == 0]
    subpaths2 = [sep.join(node) for node in subtree2.nodes if subtree2.out_degree[node] == 0]
    return subpaths1, subpaths2
