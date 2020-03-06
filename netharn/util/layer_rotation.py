"""
Implementation of
Layer rotation: a surprisingly powerful indicator of generalization in deep
networks?

References:
    https://arxiv.org/pdf/1806.01603.pdf
    https://github.com/vfdev-5/LayerRotation-pytorch/blob/master/code/handlers/layer_rotation.py
"""
import numpy as np
import torch


def get_kernel_named_params(model, copy=False):
    """
    Example:
        >>> import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> dict(get_kernel_named_params(model)).keys()
    """
    def fn(p):
        p = p.cpu().detach()
        if copy:
            p = p.clone()
        return p
    named_params = [
        (key, fn(p)) for key, p in model.named_parameters()
        if 'weight' in key
    ]
    return named_params


def layer_rotation(current_named_params, init_named_params):
    """
    Example:
        >>> import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> model2 = nh.models.ToyNet2d()
        >>> init_named_params = get_kernel_named_params(model)
        >>> current_named_params = get_kernel_named_params(model2)
        >>> ret = layer_rotation(current_named_params, init_named_params)

    """
    ret = []
    for (n1, p1), (n2, p2) in zip(current_named_params, init_named_params):
        assert n1 == n2, "{} vs {}".format(n1, n2)
        sim = torch.cosine_similarity(p1.reshape(-1), p2.reshape(-1), dim=0).item()
        dist = 1.0 - sim
        ret.append((n1, dist))
    return ret


def layer_rotation_stats(current_named_params, init_named_params):
    """
    Example:
        >>> import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> model2 = nh.models.ToyNet2d()
        >>> init_named_params = get_kernel_named_params(model)
        >>> current_named_params = get_kernel_named_params(model2)
        >>> stats = layer_rotation_stats(current_named_params, init_named_params)
        >>> print('stats = {}'.format(ub.repr2(stats, nl=1)))
    """
    import kwarray
    ret = layer_rotation(current_named_params, init_named_params)
    values = np.array([v for n, v in ret])
    stats = kwarray.stats_dict(values, median=True)
    return stats
