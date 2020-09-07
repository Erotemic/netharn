from netharn.initializers._nx_ext.path_embedding import maximum_common_path_embedding
from netharn.initializers._nx_ext.demodata import random_paths


def test_not_compatable():
    paths1 = [
        'foo/bar'
    ]
    paths2 = [
        'baz/biz'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert len(embedding1) == 0
    assert len(embedding2) == 0


def test_compatable():
    paths1 = [
        'root/suffix1'
    ]
    paths2 = [
        'root/suffix2'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['root']
    assert embedding2 == ['root']

    paths1 = [
        'root/suffix1'
    ]
    paths2 = [
        'root'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['root']
    assert embedding2 == ['root']


def test_prefixed():
    paths1 = [
        'prefix1/root/suffix1'
    ]
    paths2 = [
        'root/suffix2'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['prefix1/root']
    assert embedding2 == ['root']

    paths1 = [
        'prefix1/root/suffix1'
    ]
    paths2 = [
        'prefix1/root/suffix2'
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == ['prefix1/root']
    assert embedding2 == ['prefix1/root']


def test_simple1():
    paths1 = [
        'root/file1',
        'root/file2',
        'root/file3',
    ]
    paths2 = [
        'prefix1/root/file1',
        'prefix1/root/file2',
        'root/file3',
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == paths1
    assert embedding2 == paths2

    paths1 = [
        'root/file1',
        'root/file2',
        'root/file3',
    ]
    paths2 = [
        'prefix1/root/file1',
        'prefix1/root/file2',
        'prefix2/root/file3',
        'prefix2/root/file4',
    ]
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)
    assert embedding1 == paths1


def test_random1():
    paths1, paths2 = random_paths(10, seed=321)
    embedding1, embedding2 = maximum_common_path_embedding(paths1, paths2)


def _demodata_resnet_module_state(arch):
    """
    Construct paths corresponding to resnet convnet state keys to
    simulate a real world use case for path-embeddings.

    Ignore
    ------
    # Check to make sure the demodata agrees with real data
    import torchvision
    paths_true = list(torchvision.models.resnet50().state_dict().keys())
    paths_demo = _demodata_resnet_module_state('resnet50')
    print(ub.hzcat([ub.repr2(paths_true, nl=2), ub.repr2(paths_demo)]))
    assert paths_demo == paths_true

    paths_true = list(torchvision.models.resnet18().state_dict().keys())
    paths_demo = _demodata_resnet_module_state('resnet18')
    print(ub.hzcat([ub.repr2(paths_true, nl=2), ub.repr2(paths_demo)]))
    assert paths_demo == paths_true

    paths_true = list(torchvision.models.resnet152().state_dict().keys())
    paths_demo = _demodata_resnet_module_state('resnet152')
    print(ub.hzcat([ub.repr2(paths_true, nl=2), ub.repr2(paths_demo)]))
    assert paths_demo == paths_true
    """
    if arch == 'resnet18':
        block_type = 'basic'
        layer_blocks = [2, 2, 2, 2]
    elif arch == 'resnet50':
        block_type = 'bottleneck'
        layer_blocks = [3, 4, 6, 3]
    elif arch == 'resnet152':
        block_type = 'bottleneck'
        layer_blocks = [3, 8, 36, 3]
    else:
        raise KeyError(arch)
    paths = []
    paths += [
        'conv1.weight',
        'bn1.weight',
        'bn1.bias',
        'bn1.running_mean',
        'bn1.running_var',
        'bn1.num_batches_tracked',
    ]
    if block_type == 'bottleneck':
        num_convs = 3
    elif block_type == 'basic':
        num_convs = 2
    else:
        raise KeyError(block_type)

    for layer_idx, nblocks in enumerate(layer_blocks, start=1):
        for block_idx in range(0, nblocks):
            prefix = 'layer{}.{}.'.format(layer_idx, block_idx)

            for conv_idx in range(1, num_convs + 1):
                paths += [
                    prefix + 'conv{}.weight'.format(conv_idx),
                    prefix + 'bn{}.weight'.format(conv_idx),
                    prefix + 'bn{}.bias'.format(conv_idx),
                    prefix + 'bn{}.running_mean'.format(conv_idx),
                    prefix + 'bn{}.running_var'.format(conv_idx),
                    prefix + 'bn{}.num_batches_tracked'.format(conv_idx),
                ]
            if block_idx == 0 and layer_idx > 0:
                if block_type != 'basic' or layer_idx > 1:
                    paths += [
                        prefix + 'downsample.0.weight',
                        prefix + 'downsample.1.weight',
                        prefix + 'downsample.1.bias',
                        prefix + 'downsample.1.running_mean',
                        prefix + 'downsample.1.running_var',
                        prefix + 'downsample.1.num_batches_tracked',
                    ]
    paths += [
        'fc.weight',
        'fc.bias',
    ]
    return paths


def test_realworld_case1():
    """
    import torchvision
    paths1 = list(torchvision.models.resnet50().state_dict().keys())

    print(ub.hzcat(['paths1 = {}'.format(ub.repr2(paths1, nl=2)), ub.repr2(paths)]))
    len(paths1)
    """
    # times: resnet18:  0.16 seconds
    # times: resnet50:  0.93 seconds
    # times: resnet152: 9.83 seconds
    paths1 = _demodata_resnet_module_state('resnet50')
    paths2 = ['module.' + p for p in paths1]
    # import ubelt as ub
    # with ub.Timer('test-real-world-case'):
    embedding1, embedding2 = maximum_common_path_embedding(
            paths1, paths2, sep='.')
    assert [p[len('module.'):] for p in embedding2] == embedding1


def test_realworld_case2():
    """
    import torchvision
    paths1 = list(torchvision.models.resnet152().state_dict().keys())
    print('paths1 = {}'.format(ub.repr2(paths1, nl=2)))
    """
    backbone = _demodata_resnet_module_state('resnet18')

    # Detector strips of prefix and suffix of the backbone net
    subpaths = ['detector.' + p for p in backbone[6:-2]]
    paths1 = [
        'detector.conv1.weight',
        'detector.bn1.weight',
        'detector.bn1.bias',
    ] + subpaths + [
        'detector.head1.conv1.weight',
        'detector.head1.conv2.weight',
        'detector.head1.conv3.weight',
        'detector.head1.fc.weight',
        'detector.head1.fc.bias',
        'detector.head2.conv1.weight',
        'detector.head2.conv2.weight',
        'detector.head2.conv3.weight',
        'detector.head2.fc.weight',
        'detector.head2.fc.bias',
    ]

    paths2 = ['module.' + p for p in backbone]

    # import ubelt as ub
    # with ub.Timer('test-real-world-case'):
    embedding1, embedding2 = maximum_common_path_embedding(
            paths1, paths2, sep='.')

    mapping = dict(zip(embedding1, embedding2))

    # Note in the embedding case there may be superfluous assignments
    # but they can either be discarded in post-processing or they wont
    # be in the solution if we use isomorphisms instead of embeddings
    assert len(subpaths) < len(mapping), (
        'all subpaths should be in the mapping')

    non_common1 = set(paths1) - set(embedding1)
    non_common2 = set(paths2) - set(embedding2)

    assert non_common2 == {
            'module.bn1.num_batches_tracked',
            'module.bn1.running_mean',
            'module.bn1.running_var',
            }

    assert non_common1 == {
        'detector.conv1.weight',
        'detector.head1.conv1.weight',
        'detector.head1.conv2.weight',
        'detector.head1.conv3.weight',
        'detector.head1.fc.bias',
        'detector.head1.fc.weight',
        'detector.head2.conv2.weight',
        'detector.head2.conv3.weight',
    }
    # print('non_common1 = {}'.format(ub.repr2(non_common1, nl=1)))
    # print('non_common2 = {}'.format(ub.repr2(non_common2, nl=1)))
    # assert [p[len('module.'):] for p in embedding2] == embedding1
