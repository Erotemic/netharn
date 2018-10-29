"""
References:
    https://github.com/alykhantejani/initializers
"""
import ubelt as ub
from os.path import dirname, exists, join, normpath
from netharn import util
import torch
import six
from netharn.initializers import nninit_base


class Pretrained(nninit_base._BaseInitializer, ub.NiceRepr):
    """
    Attributes:
        fpath (str): location of the pretrained weights file
        initializer (_BaseInitializer): backup initializer if the weights can
            only be partially applied
        info (dict, optional): specify explicit history info

    Example:
        >>> from netharn.initializers.nninit_core import *
        >>> from netharn.models import toynet
        >>> from os.path import join
        >>> model1 = toynet.ToyNet2d()
        >>> dpath = ub.ensure_app_cache_dir('netharn', 'tests')
        >>> fpath = join(dpath, 'toynet_weights.pt')
        >>> torch.save(model1.state_dict(), fpath)
        >>> model2 = toynet.ToyNet2d()
        >>> self = Pretrained(fpath)
        >>> self(model2)
        >>> #model2.state_dict() == model1.state_dict()
    """
    def __init__(self, fpath, initializer=None, info=None):
        self.fpath = fpath
        if isinstance(initializer, six.string_types):
            from netharn import initializers
            initializer = getattr(initializers, initializer)()
        self.initializer = initializer
        self.info = info

    def __nice__(self):
        return self.fpath

    def _rectify_deploy_zip_weights_path(self):
        # Find the path to the weights inside the zipfile
        import zipfile
        fpath = None
        candidates = []
        with zipfile.ZipFile(self.fpath, 'r') as myzip:
            for zinfo in myzip.filelist:
                if zinfo.filename.endswith('deploy_snapshot.pt'):
                    candidates = [zinfo.filename]
                    break
                elif zinfo.filename.endswith('.pt'):
                    candidates.append(zinfo)
        if len(candidates) == 0:
            raise OSError('Cannot find pretrained weights in {}'.format(
                self.fpath))
        elif len(candidates) > 1:
            raise OSError('Multiple weights files in {}'.format(
                self.fpath))
        else:
            fpath = join(self.fpath, candidates[0])
        return fpath

    def forward(self, model):
        from netharn import XPU
        xpu = XPU.from_data(model)
        # model_state_dict = xpu.load(self.fpath)
        if self.fpath is None:
            raise ValueError('Pretrained fpath is None!')

        # Handle torch deployment zipfiles
        if exists(self.fpath) and self.fpath.endswith('.zip'):
            fpath = self._rectify_deploy_zip_weights_path()
        else:
            fpath = self.fpath

        try:
            model_state_dict = xpu.load(util.zopen(fpath, 'rb', seekable=True))
        except Exception:
            print('Failed to open fpath = {!r}'.format(fpath))
            raise

        if 'model_state_dict' in model_state_dict:
            model_state_dict = model_state_dict['model_state_dict']
        elif 'weights' in model_state_dict:
            model_state_dict = model_state_dict['weights']
        else:
            # If the dictionary is flat (i.e. all values are tensors) then it
            # is safe to assume this file only contains weights.
            # Otherwise raise an exception.
            if not all(torch.is_tensor(v) for v in model_state_dict.values()):
                raise Exception(
                    'snapshot file is nested, but does not have expected keys: '
                    'model_state_dict or weights. Root keys are {}'.format(
                        sorted(model_state_dict.keys())
                    ))
        nninit_base.load_partial_state(model, model_state_dict,
                                       initializer=self.initializer)

    def history(self):
        """
        if available return the history of the model as well
        """
        import netharn as nh
        if self.info is None:
            if False:
                info_dpath = dirname(dirname(ub.truepath(self.fpath)))
                info_fpath = join(info_dpath, 'train_info.json')
                if exists(info_fpath):
                    info = nh.util.read_json(info_fpath)
                else:
                    info = '__UNKNOWN__'
            else:
                # TODO: check for train_info.json in a few different places
                snap_fpath = ub.truepath(self.fpath)
                candidate_paths = [
                    join(dirname(snap_fpath), 'train_info.json'),
                    join(dirname(dirname(snap_fpath)), 'train_info.json'),
                ]
                info = None
                for info_fpath in candidate_paths:
                    info_fpath = normpath(info_fpath)
                    try:
                        # Info might be inside of a zipfile
                        info = nh.util.read_json(nh.util.zopen(info_fpath))
                        break
                    except Exception as ex:
                        pass
                if info is None:
                    info = '__UNKNOWN__'
        else:
            info = self.info
        return info


class Orthogonal(nninit_base._BaseInitializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from netharn.initializers.nninit_core import *
        >>> from netharn.models import toynet
        >>> self = Orthogonal()
        >>> model = toynet.ToyNet2d()
        >>> try:
        >>>     self(model)
        >>> except RuntimeError:
        >>>     import pytest
        >>>     pytest.skip('geqrf: Lapack probably not availble')
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """
    def __init__(self, gain=1):
        self.gain = gain

    def forward(self, model):
        try:
            func = torch.nn.init.orthogonal_
        except AttributeError:
            func = torch.nn.init.orthogonal

        nninit_base.apply_initializer(model, func, self.__dict__)


class KaimingUniform(nninit_base._BaseInitializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from netharn.initializers.nninit_core import *
        >>> from netharn.models import toynet
        >>> self = KaimingUniform()
        >>> model = toynet.ToyNet2d()
        >>> self(model)
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """
    def __init__(self, param=0, mode='fan_in'):
        self.a = param
        self.mode = mode

    def forward(self, model):
        try:
            func = torch.nn.init.kaiming_uniform_
        except AttributeError:
            func = torch.nn.init.kaiming_uniform
        nninit_base.apply_initializer(model, func, self.__dict__)


class KaimingNormal(nninit_base._BaseInitializer):
    """
    Same as HeNormal, but uses pytorch implementation

    Example:
        >>> from netharn.initializers.nninit_core import *
        >>> from netharn.models import toynet
        >>> self = KaimingNormal()
        >>> model = toynet.ToyNet2d()
        >>> self(model)
        >>> layer = torch.nn.modules.Conv2d(3, 3, 3)
        >>> self(layer)
    """
    def __init__(self, param=0, mode='fan_in'):
        self.a = param
        self.mode = mode

    def forward(self, model):
        try:
            func = torch.nn.init.kaiming_normal_
        except AttributeError:
            func = torch.nn.init.kaiming_normal
        nninit_base.apply_initializer(model, func, self.__dict__)


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.initializers.nninit_core all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
