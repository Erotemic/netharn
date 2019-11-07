from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import ubelt as ub
import numpy as np


def test_convT_rf():
    """
    CommandLine:
        xdoctest -m ~/code/netharn/tests/test_receptive_feild.py test_convT_rf
    """
    # Test that we always invert whatever weird crazy thing we do
    import netharn as nh
    rng = np.random.RandomState(3668028386)

    ntrials = 100
    M = 9

    for _ in ub.ProgIter(range(ntrials), desc='testing rand convT instances'):
        depth = rng.randint(1, M)
        params = []
        for i in range(depth):
            k = rng.randint(0, 1 + M // 2) * 2 + 1
            s = rng.randint(1, 1 + M)
            d = rng.randint(1, 1 + M)
            p = rng.randint(0, 1 + M)
            params.append((i, (k, s, d)))

        # Construct a series of forward convolutions and the tranpose
        # convolutions that should "invert" them. Assert that the strides and
        # crop of the RF are the same on every layer. Furthremote the RF size
        # should strictly increase.

        layers = ub.odict()
        for i, (k, s, d) in params:
            key = 'c{}'.format(i)
            conv = nn.Conv2d(1, 1, kernel_size=k, stride=s, padding=p, dilation=d)
            layers[key] = conv

        for i, (k, s, d) in reversed(params):
            key = 'c{}T'.format(i)
            convT = nn.ConvTranspose2d(1, 1, kernel_size=k, stride=s, padding=p, dilation=d)
            layers[key] = convT

        module = nn.Sequential(layers)
        field = nh.ReceptiveFieldFor(module)()

        input_rf = nh.ReceptiveFieldFor.input()
        symmetric = [('input', input_rf)] + list(field.hidden.items())

        for a, b, in ub.iter_window(symmetric, 2):
            k1, v1 = a
            k2, v2 = b
            assert np.all(v1['shape'] <= v2['shape']), 'v1={} v2={}'.format(v1, v2)

        for a, b in zip(symmetric, symmetric[::-1]):
            k1, v1 = a
            k2, v2 = b
            assert np.all(v1['stride'] == v2['stride']), 'v1={} v2={}'.format(v1, v2)
            assert np.all(v1['crop'] == v2['crop']), 'v1={} v2={}'.format(v1, v2)
