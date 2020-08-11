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
                       mangle=True, initializer=None):
    """
    CommandLine:
        python -m netharn.initializers.nninit_base load_partial_state

    Args:
        model (torch.nn.Module): module to initialize

        model_state_dict (dict): state dict we wish to transfer

        leftover (callable): fallback method for initializing incompatible
             areas, if none then those areas are left as-is.

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
    """
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

            OLD_WAY = 0
            if OLD_WAY:
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
            elif 0:
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
            else:
                # I believe this is the correct way to solve the problem
                paths1 = sorted(other_keys)
                paths2 = sorted(self_state)
                subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2)
                mapping = ub.dzip(subpaths1, subpaths2)
                print('mapping = {}'.format(ub.repr2(mapping, nl=1)))
                model_state_dict = ub.map_keys(lambda k: mapping.get(k, k), model_state_dict)
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


def maximum_common_ordered_subpaths(paths1, paths2):
    """
    paths1 = [
        'stats',
        'z.mod.f.0.w',
        'a.z.mod.f.0.b',
        'z.mod.f.1.b',
        'z.mod.f.1.n',
        'z.mod.f.1.m',
        'z.mod.f.1.v',
        'z.mod.f.2.m',
        'z.mod.z.q'
    ]
    # paths1 = ['mod']

    paths2 = [
        'stats',
        'bar.f.0.w',
        'bar.foo.extra.z.q',
        'bar.foo.extra.f.0.w',
        'bar.foo.extra.f.1.b',
        'bar.foo.extra.f.1.n',
        'bar.foo.extra.f.1.w',
        'bar.foo.extra.f.3.z',  # FIXME we need to handle label comparision operators
        # I think we allow labels to match if they have the same suffix
    ]

    subpaths1, subpaths2 = maximum_common_ordered_subpaths(paths1, paths2)
    mapping = ub.dzip(subpaths1, subpaths2)
    print('mapping = {}'.format(ub.repr2(mapping, nl=1)))

    print([n for n in tree1.nodes if tree1.in_degree[n] > 1])
    print([n for n in tree2.nodes if tree2.in_degree[n] > 1])
    _print_forest(tree1)
    _print_forest(tree2)

    # for n in subtree1.nodes:
    #     subtree1.nodes[n]['label'] = n[-1]
    _print_forest(subtree1)
    _print_forest(subtree2)

    tree1_remain = tree1.copy()
    tree1_remain.remove_nodes_from(subtree1.nodes)
    _print_forest(tree1_remain)
    """
    # the longest common balanced sequence problem
    def _matchable(tok1, tok2):
        return tok1.value[-1] == tok2.value[-1]

    sep = '.'

    import networkx as nx
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

    eq = _matchable
    subtree1, subtree2 = maximum_common_ordered_subtree(tree1, tree2, eq=eq)

    subpaths1 = [sep.join(node) for node in subtree1.nodes if subtree1.out_degree[node] == 0]
    subpaths2 = [sep.join(node) for node in subtree2.nodes if subtree2.out_degree[node] == 0]
    return subpaths1, subpaths2


def maximum_common_ordered_subtree(tree1, tree2, eq=None):
    """
    Finds the maximum common subtree between two ordered trees.

    Implements algorithm described in [1]_.

    References:
        On the Maximum Common Embedded Subtree Problem for Ordered Trees
        https://pdfs.semanticscholar.org/0b6e/061af02353f7d9b887f9a378be70be64d165.pdf

    # import netharn as nh
    # nh.util.shortest_unique_suffixes(paths1 + paths2, sep='.')
    # nh.util.shortest_unique_suffixes(paths1)
    # + paths2, sep='.')

    # the longest common balanced sequence problem
    def _matchable(tok1, tok2):
        return tok1.value[-1] == tok2.value[-1]

    eq = _matchable
    print([n for n in tree1.nodes if tree1.in_degree[n] > 1])
    print([n for n in tree2.nodes if tree2.in_degree[n] > 1])
    _print_forest(tree1)
    _print_forest(tree2)
    subtree1, subtree2 = maximum_common_ordered_subtree(tree1, tree2, eq=eq)
    # for n in subtree1.nodes:
    #     subtree1.nodes[n]['label'] = n[-1]
    _print_forest(subtree1)
    _print_forest(subtree2)

    tree1_remain = tree1.copy()
    tree1_remain.remove_nodes_from(subtree1.nodes)
    _print_forest(tree1_remain)

    tree = tree1
    """
    from collections import namedtuple
    Token = namedtuple('Token', ['action', 'value'])
    def tree_to_balanced_sequence(tree, open_to_close=None, toks=None):
        import networkx as nx
        # mapping between opening and closing tokens
        sources = [n for n in tree.nodes if tree.in_degree[n] == 0]
        sequence = []

        if open_to_close is None:
            open_to_close = {}
        if toks is None:
            toks = {}

        for source in sources:
            for u, v, etype in nx.dfs_labeled_edges(tree, source=source):
                if etype == 'forward':
                    # u has been visited by v has not
                    if v not in toks:
                        if 1:
                            # TODO: token encoding scheme where subdirectories
                            # are matchable via a custom operation.
                            # open_tok = '<{}>'.format(v)
                            # close_tok = '</{}>'.format(v)
                            open_tok = Token('open', v)
                            close_tok = Token('close', v)
                        else:
                            open_tok = len(toks)
                            close_tok = -open_tok
                        toks[v] = open_tok
                        open_to_close[open_tok] = close_tok
                    open_tok = toks[v]
                    sequence.append(open_tok)
                elif etype == 'reverse':
                    # Both u and v are visited and the edge is in the tree
                    close_tok = open_to_close[toks[v]]
                    sequence.append(close_tok)
                else:
                    raise KeyError(etype)
        sequence = tuple(sequence)
        return sequence, open_to_close, toks

    def generate_balance(sequence, open_to_close):
        """
        open_to_close = {0: 1}
        sequence = [0, 0, 0, 1, 1, 1]
        gen = generate_balance(sequence, open_to_close)
        for flag, token in gen:
            print('flag={:d}, token={}'.format(flag, token))
        """
        stack = []

        class UnbalancedException(Exception):
            pass

        # Traversing the Expression
        for token in sequence:

            if token in open_to_close:
                # Push opening elements onto the stack
                stack.append(token)
            else:
                if not stack:
                    raise UnbalancedException
                prev_open = stack.pop()
                want_close = open_to_close[prev_open]

                if token != want_close:
                    raise UnbalancedException

            # If the stack is empty the sequence is currently balanced
            currently_balanced = not bool(stack)
            yield currently_balanced, token

        if stack:
            raise UnbalancedException

    def head_tail(sequence, open_to_close):
        """
        open_to_close = {0: 1}
        sequence = [0, 0, 0, 1, 1, 1, 0, 1]
        open_to_close = {'{': '}', '(': ')', '[': ']'}
        sequence = '({[[]]})[[][]]'
        a1, b1, head, tail = head_tail(sequence, open_to_close)
        a2, b2, tail1, tail2 = head_tail(tail, open_to_close)
        """
        gen = generate_balance(sequence, open_to_close)

        bal_curr, tok_curr = next(gen)
        pop_open = sequence[0:1]
        want_close = open_to_close[tok_curr]

        head_stop = 1
        for head_stop, (bal_curr, tok_curr) in enumerate(gen, start=1):
            if tok_curr is None:
                break
            elif bal_curr and tok_curr == want_close:
                pop_close = sequence[head_stop:head_stop + 1]
                break
        head = sequence[1:head_stop]
        if __debug__:
            list(gen)  # exhaust the generator to check we are balanced
        tail = sequence[head_stop + 1:]
        return pop_open, pop_close, head, tail

    def longest_common_balanced_sequence(seq1, seq2, open_to_close, eq=None):
        """
        open_to_close = {'0': '1'}
        seq1 = '0010010010111100001011011011'
        seq2 = '001000101101110001000100101110111011'

        open_to_close = {'(': ')'}
        seq1 = '(()(()(()())))(((()())())())'
        seq2 = '(()((()())()))((()((()(()()))()))())'
        longest_common_balanced_sequence(seq1, seq2, open_to_close)

        open_to_close = {'0': '1'}
        seq1 = '0010010010111100001011011011'
        seq2 = '001000101101110001000100101110111011'
        longest_common_balanced_sequence(seq1, seq2, open_to_close)

        open_to_close = {'0': '1'}
        seq1 = '001101'
        seq2 = '00110011'
        seq1 = '001101'
        seq2 = '00110011'
        longest_common_balanced_sequence(seq1, seq2, open_to_close)

        open_to_close = {'{': '}', '(': ')', '[': ']'}
        seq1 = '{}{[]}[{}]'
        seq2 = '({}[{{}}])'
        def comp(a, b):
            return True
        eq = comp
        best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close, eq=eq)
        subseq1, subseq2 = best
        """
        if eq is None:
            import operator
            eq = operator.eq

        _memo = {}
        def _lcs(seq1, seq2):
            if not seq1:
                return (seq1, seq1), 0
            elif not seq2:
                return (seq2, seq2), 0
            else:
                # if len(seq2) < len(seq1):
                #     seq1, seq2 = seq2, seq1
                key = (seq1, seq2)
                if key in _memo:
                    return _memo[key]

                a1, b1, head1, tail1 = head_tail(seq1, open_to_close)
                a2, b2, head2, tail2 = head_tail(seq2, open_to_close)

                candidates = {}

                # Case 1: The LCS involves this edge
                if eq(a1[0], a2[0]):
                    # TODO: need to return the correspondence between the
                    # matches and the original nodes.
                    new_heads, pval_h = _lcs(head1, head2)
                    new_tails, pval_t = _lcs(tail1, tail2)

                    new_head1, new_head2 = new_heads
                    new_tail1, new_tail2 = new_tails

                    subseq1 = a1 + new_head1 + b1 + new_tail1   # fixme
                    subseq2 = a2 + new_head2 + b2 + new_tail2   # fixme
                    cand1 = (subseq1, subseq2)
                    # cand1 = a2 + part1 + b2 + part2   # fixme
                    candidates[cand1] = pval_h + pval_t + 1

                # Case 2: The current edge in sequence1 is deleted
                cand2, val2 = _lcs(head1 + tail1, seq2)
                candidates[cand2] = val2

                # Case 3: The current edge in sequence2 is deleted
                cand3, val3 = _lcs(seq1, head2 + tail2)
                candidates[cand3] = val3

                best = ub.argmax(candidates)
                value = candidates[best]
                # print('key={!r}, best={!r}, value={!r}'.format(key, best, value))
                _memo[key] = (best, value)
                return best, value

        best = _lcs(seq1, seq2)
        return best

    def seq_to_tree(subseq, open_to_close, toks):
        import networkx as nx
        open_to_tok = ub.invert_dict(toks)
        subtree = nx.OrderedDiGraph()
        stack = []
        for token in subseq:
            if token in open_to_close:
                node = open_to_tok[token]
                if stack:
                    parent = open_to_tok[stack[-1]]
                    subtree.add_edge(parent, node)
                else:
                    subtree.add_node(node)
                stack.append(token)
            else:
                if not stack:
                    raise Exception
                prev_open = stack.pop()
                want_close = open_to_close[prev_open]
                if token != want_close:
                    raise Exception
        return subtree

    # Convert the trees to balanced sequences
    sequence1, open_to_close, toks = tree_to_balanced_sequence(tree1, open_to_close=None, toks=None)
    sequence2, open_to_close, toks = tree_to_balanced_sequence(tree2, open_to_close, toks)
    seq1 = sequence1
    seq2 = sequence2

    # Solve the longest common balanced sequence problem
    best, value = longest_common_balanced_sequence(seq1, seq2, open_to_close, eq=eq)
    subseq1, subseq2 = best

    # Convert the subsequence back into a tree
    subtree1 = seq_to_tree(subseq1, open_to_close, toks)
    subtree2 = seq_to_tree(subseq2, open_to_close, toks)
    return subtree1, subtree2


def _print_forest(graph):
    """
    Nice ascii representation of a forest

    Ignore:
        graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        _print_forest(graph)

        graph = CategoryTree.demo('coco').graph
        _print_forest(graph)
    """
    import networkx as nx
    assert nx.is_forest(graph)
    from kwcoco.category_tree import to_directed_nested_tuples
    encoding = to_directed_nested_tuples(graph)
    def _recurse(encoding, indent=''):
        for idx, item in enumerate(encoding):
            node, data, children = item
            if idx == len(encoding) - 1:
                this_prefix = indent + '└── '
                next_prefix = indent + '    '
            else:
                this_prefix = indent + '├── '
                next_prefix = indent + '│   '
            label = graph.nodes[node].get('label', node)
            print(this_prefix + str(label))
            _recurse(children, indent=next_prefix)
    _recurse(encoding)
