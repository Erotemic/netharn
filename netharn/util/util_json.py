# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import json
import six
import torch
import numpy as np
import ubelt as ub
from collections import OrderedDict


def walk_json(node):
    for key, val in node.items():
        if isinstance(val, dict):
            for item in walk_json(val):
                yield item
        elif isinstance(val, list):
            for subval in val:
                if isinstance(subval, dict):
                    for item in walk_json(subval):
                        yield item
                else:
                    assert False
        else:
            yield key, val


class LossyJSONEncoder(json.JSONEncoder):
    """
    Helps cooerce objects into a json-serializable format. Note that this is a
    lossy process. Information about object types / array types are lost. Only
    info directly translatable to json primitives are preserved as those
    primitive types.  (e.g: tuples and ndarrays are encoded as lists, and
    objects only remember their dict of attributes).

    Example:
        >>> import json
        >>> class MyClass(object):
        >>>     def __init__(self, foo='bar'):
        >>>         self.foo = foo
        >>>         self.spam = 32
        >>>         self.eggs = np.array([32])
        >>>     def __json__(self):
        >>>         return {self.__class__.__name__: self.__dict__}
        >>> self = MyClass()
        >>> text = json.dumps(self, cls=LossyJSONEncoder)
        >>> reloaded = json.loads(text)
        >>> expected = {'MyClass': {'foo': 'bar', 'spam': 32, 'eggs': [32]}}
        >>> assert reloaded == expected
    """
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__()
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(LossyJSONEncoder, self).default(obj)
        # return json.JSONEncoder.default(self, obj)


class NumpyEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
    """

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        import base64
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)

    @staticmethod
    def json_numpy_obj_hook(dct):
        """Decodes a previously encoded numpy ndarray with proper shape and dtype.

        :param dct: (dict) json encoded ndarray
        :return: (ndarray) if input was an encoded ndarray
        """
        import base64
        if isinstance(dct, dict) and '__ndarray__' in dct:
            data = base64.b64decode(dct['__ndarray__'])
            return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
        return dct


def write_json(fpath, data):
    """
    Write human readable json files
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd and isinstance(data, pd.DataFrame):
        # pretty pandas
        json_text = (json.dumps(json.loads(data.to_json()), indent=4))
    elif isinstance(data, dict):
        json_text = json.dumps(data, cls=LossyJSONEncoder, indent=4)
    else:
        raise TypeError(type(data))
    ub.writeto(fpath, json_text)


def read_json(fpath):
    """
    Write human readable json files
    """
    if isinstance(fpath, six.string_types):
        return json.load(open(fpath, 'r'))
    else:
        return json.load(fpath)


def ensure_json_serializable(dict_, normalize_containers=False, verbose=0):
    """
    Attempt to convert common types (e.g. numpy) into something json complient

    Convert numpy and tuples into lists

    Args:
        normalize_containers (bool, default=False):
            if True, normalizes dict containers to be standard python
            structures.

    Example:
        >>> data = ub.ddict(lambda: int)
        >>> data['foo'] = ub.ddict(lambda: int)
        >>> data['bar'] = np.array([1, 2, 3])
        >>> data['foo']['a'] = 1
        >>> data['foo']['b'] = torch.FloatTensor([1, 2, 3])
        >>> result = ensure_json_serializable(data, normalize_containers=True)
        >>> assert type(result) is dict
    """
    dict_ = copy.deepcopy(dict_)

    def _norm_container(c):
        if isinstance(c, dict):
            # Cast to a normal dictionary
            if isinstance(c, OrderedDict):
                if type(c) is not OrderedDict:
                    c = OrderedDict(c)
            else:
                if type(c) is not dict:
                    c = dict(c)
        return c

    # inplace convert any ndarrays to lists
    def _walk_json(data, prefix=[]):
        items = None
        if isinstance(data, list):
            items = enumerate(data)
        elif isinstance(data, tuple):
            items = enumerate(data)
        elif isinstance(data, dict):
            items = data.items()
        else:
            raise TypeError(type(data))

        root = prefix
        level = {}
        for key, value in items:
            level[key] = value

        # yield a dict so the user can choose to not walk down a path
        yield root, level

        for key, value in level.items():
            if isinstance(value, (dict, list, tuple)):
                path = prefix + [key]
                for _ in _walk_json(value, prefix=path):
                    yield _

    def _convert(dict_, root, key, new_value):
        d = dict_
        for k in root:
            d = d[k]
        d[key] = new_value

    def _flatmap(func, data):
        if isinstance(data, list):
            return [_flatmap(func, item) for item in data]
        else:
            return func(data)

    to_convert = []
    for root, level in ub.ProgIter(_walk_json(dict_), desc='walk json',
                                   verbose=verbose):
        for key, value in level.items():
            if isinstance(value, tuple):
                # Convert tuples on the fly so they become mutable
                new_value = list(value)
                _convert(dict_, root, key, new_value)
            elif isinstance(value, np.ndarray):
                new_value = value.tolist()
                if 0:
                    if len(value.shape) == 1:
                        if value.dtype.kind in {'i', 'u'}:
                            new_value = list(map(int, new_value))
                        elif value.dtype.kind in {'f'}:
                            new_value = list(map(float, new_value))
                        elif value.dtype.kind in {'c'}:
                            new_value = list(map(complex, new_value))
                        else:
                            pass
                    else:
                        if value.dtype.kind in {'i', 'u'}:
                            new_value = _flatmap(int, new_value)
                        elif value.dtype.kind in {'f'}:
                            new_value = _flatmap(float, new_value)
                        elif value.dtype.kind in {'c'}:
                            new_value = _flatmap(complex, new_value)
                        else:
                            pass
                            # raise TypeError(value.dtype)
                to_convert.append((root, key, new_value))
            elif isinstance(value, torch.Tensor):
                new_value = value.data.cpu().numpy().tolist()
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.int16, np.int32, np.int64,
                                    np.uint16, np.uint32, np.uint64)):
                new_value = int(value)
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.float32, np.float64)):
                new_value = float(value)
                to_convert.append((root, key, new_value))
            elif isinstance(value, (np.complex64, np.complex128)):
                new_value = complex(value)
                to_convert.append((root, key, new_value))
            elif hasattr(value, '__json__'):
                new_value = value.__json__()
                to_convert.append((root, key, new_value))
            elif normalize_containers:
                if isinstance(value, dict):
                    new_value = _norm_container(value)
                    to_convert.append((root, key, new_value))

    for root, key, new_value in to_convert:
        _convert(dict_, root, key, new_value)

    if normalize_containers:
        # normalize the outer layer
        dict_ = _norm_container(dict_)
    return dict_
