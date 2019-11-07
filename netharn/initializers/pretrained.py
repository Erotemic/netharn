import six
import torch
import ubelt as ub
from os.path import dirname
from os.path import exists
from os.path import join
from os.path import normpath
from netharn import api
from netharn.initializers.functional import load_partial_state


class Pretrained(api.Initializer, ub.NiceRepr):
    """
    Attributes:
        fpath (str): location of the pretrained weights file
        initializer (netharn.Initializer): backup initializer if the weights can
            only be partially applied
        info (dict, optional): specify explicit history info

    Example:
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
            initializer_ = api.Initializer.coerce(initializer=initializer)
            initializer = initializer_[0](**initializer_[1])
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

    def _load_model_state(self, xpu=None):
        """
        Load the model state from a path or from within a zipfile
        """
        import netharn as nh
        from netharn import XPU
        if self.fpath is None:
            raise ValueError('Pretrained fpath is None!')

        if xpu is None:
            xpu = XPU.coerce('cpu')

        # Handle torch deployment zipfiles
        if exists(self.fpath) and self.fpath.endswith('.zip'):
            fpath = self._rectify_deploy_zip_weights_path()
        else:
            fpath = self.fpath

        try:
            file = nh.util.zopen(fpath, 'rb', seekable=True)
            model_state_dict = xpu.load(file)
        except Exception:
            print('Failed to open fpath = {!r}'.format(fpath))
            raise
        return model_state_dict

    def forward(self, model, verbose=2):
        from netharn import XPU
        xpu = XPU.from_data(model)

        # model_state_dict = xpu.load(self.fpath)
        model_state_dict = self._load_model_state(xpu=xpu)

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
        # Remove any DataParallel / DataSerial
        raw_model = xpu.raw(model)
        info = load_partial_state(raw_model, model_state_dict,
                                  initializer=self.initializer,
                                  verbose=verbose)
        return info

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
