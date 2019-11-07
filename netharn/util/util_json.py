# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import numpy as np
import ubelt as ub
import pandas as pd


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


class JSONEncoder(json.JSONEncoder):
    def default(harn, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(harn, obj)


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


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


# def json_numpy_obj_hook(dct):
#     """Decodes a previously encoded numpy ndarray with proper shape and dtype.

#     :param dct: (dict) json encoded ndarray
#     :return: (ndarray) if input was an encoded ndarray
#     """
#     import base64
#     if isinstance(dct, dict) and '__ndarray__' in dct:
#         data = base64.b64decode(dct['__ndarray__'])
#         return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
#     return dct


def write_json(fpath, data):
    """
    Write human readable json files
    """
    if isinstance(data, pd.DataFrame):
        # pretty pandas
        json_text = (json.dumps(json.loads(data.to_json()), indent=4))
    elif isinstance(data, dict):
        json_text = json.dumps(data, cls=JSONEncoder, indent=4)
    else:
        raise TypeError(type(data))

    ub.writeto(fpath, json_text)


def read_json(fpath):
    """
    Write human readable json files
    """
    return json.load(open(fpath, 'r'))
