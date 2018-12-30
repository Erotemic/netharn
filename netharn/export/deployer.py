# -*- coding: utf-8 -*-
"""
Deployment component of the Pytorch exporter.

This file contains DeployedModel, which consists of logic to take the
model topology definition along with the "best" snapshot in a training
directory and package it up into a standalone zipfile. The DeployedModel can
also be used to reload model from this zipfile. Thus this zipfile can be passed
around as a pytorch model topology+pretrained weights transfer format.

The following docstring illustrates how this module may be used.

CommandLine:
    # Runs the following example
    xdoctest -m netharn.export.deployer __doc__:0

    # Runs all the doctests
    xdoctest -m netharn.export.deployer all

Example:
    >>> # xdoc: +IGNORE_WANT
    >>> # This example will train a small model and then deploy it.
    >>> import netharn as nh
    >>> #
    >>> #################################################
    >>> print('--- STEP 1: TRAIN A MODEL ---')
    >>> # This will train a toy model with toy data using netharn
    >>> hyper = nh.HyperParams(**{
    >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/tests/deploy'),
    >>>     'nice'        : 'deploy_demo',
    >>>     'xpu'         : nh.XPU.cast('cpu'),
    >>>     'datasets'    : {
    >>>         'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
    >>>         'test':  nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
    >>>     },
    >>>     'loaders'     : {'batch_size': 64},
    >>>     'model'       : (nh.models.ToyNet2d, {}),
    >>>     'optimizer'   : (nh.optimizers.SGD, {'lr': 0.0001}),
    >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
    >>>     'initializer' : (nh.initializers.KaimingNormal, {}),
    >>>     'scheduler'   : (nh.schedulers.ListedLR, {
    >>>         'points': {0: .01, 3: 0.1},
    >>>         'interpolate': True,
    >>>     }),
    >>>     'monitor'     : (nh.Monitor, {'max_epoch': 3,}),
    >>> })
    >>> harn = nh.FitHarn(hyper)
    >>> harn.initialize(reset='delete')
    >>> harn.run()
    --- STEP 1: TRAIN A MODEL ---
    Symlink: ...tests/deploy/fit/runs/deploy_demo/imlbrwnc -> ...tests/deploy/fit/nice/deploy_demo
    Model has 824 parameters
    Mounting ToyNet2d model on CPU
    Initializing model weights
     * harn.train_dpath = '...tests/deploy/fit/runs/deploy_demo/imlbrwnc'
     * harn.nice_dpath  = '...tests/deploy/fit/nice/deploy_demo'
    Snapshots will save to harn.snapshot_dpath = '...tests/deploy/fit/runs/deploy_demo/imlbrwnc/torch_snapshots'
    dont forget to start:
        tensorboard --logdir .../tests/deploy/fit/nice
    === begin training 0 / 3 ===
    epoch lr:0.1 │ vloss is unevaluated: 100%|██████████████████████████| 3/3 ...
    train x64 │ loss:0.692 │: 100%|███████████████████████████████████████████████████████| ...
    test x64 │ loss:0.735 │: 100%|████████████████████████████████████████████████████████| ...
    <BLANKLINE>
    Maximum harn.epoch reached, terminating ...
    <BLANKLINE>
    training completed
    exiting fit harness.
    >>> #
    >>> ##########################################
    >>> print('--- STEP 2: DEPLOY THE MODEL ---')
    >>> # First we export the model topology to a standalone file
    >>> # (This step is not done in the run itself)
    >>> from netharn.export import exporter
    >>> topo_fpath = exporter.export_model_code(harn.train_dpath, harn.hyper.model_cls, harn.hyper.model_params)
    >>> # Now create an instance of deployed model that points to the
    >>> # Training dpath. (Note the directory structure setup by netharn is
    >>> # itself a deployment, it just has multiple files)
    >>> deployer = DeployedModel(harn.train_dpath)
    >>> # Use the DeployedModel to package the imporant info in train_dpath
    >>> # into a standalone zipfile.
    >>> zip_fpath = deployer.package()
    >>> print('We exported the topology to: {!r}'.format(topo_fpath))
    >>> print('We exported the topology+weights to: {!r}'.format(zip_fpath))
    --- STEP 2: DEPLOY THE MODEL ---
    We exported the topology to: '...tests/deploy/fit/runs/deploy_demo/imlbrwnc/ToyNet2d_0962a2.py'
    We exported the topology+weights to: '...tests/deploy/fit/runs/deploy_demo/imlbrwnc/imlbrwnc.zip'
    >>> #
    >>> #################################################
    >>> print('--- STEP 3: LOAD THE DEPLOYED MODEL ---')
    >>> # Now we can move the zipfile anywhere we want, and we should
    >>> # still be able to load it (depending on how coupled the model is).
    >>> # Create an instance of DeployedModel that points to the zipfile
    >>> # (Note: DeployedModel is used to both package and load models)
    >>> loader = DeployedModel(zip_fpath)
    >>> model = loader.load_model()
    >>> # This model is now loaded with the corret weights.
    >>> # You can use it as normal.
    >>> model.eval()
    >>> images = harn._demo_batch(0)[0][0][0:1]
    >>> outputs = model(images)
    >>> print('outputs = {!r}'.format(outputs))
    >>> # Not that the loaded model is independent of harn.model
    >>> print('model.__module__ = {!r}'.format(model.__module__))
    >>> print('harn.model.module.__module__ = {!r}'.format(harn.model.module.__module__))
    --- STEP 3: LOAD THE DEPLOYED MODEL ---
    outputs = tensor([[ 0.5423,  0.4577]])
    model.__module__ = 'imlbrwnc/ToyNet2d_b558e1'
    harn.model.module.__module__ = 'netharn.models.toynet'
"""
import glob
import json
import six
import ubelt as ub
# import warnings
import zipfile
import os
from os.path import exists
from os.path import isdir
from os.path import join
from os.path import relpath

__all__ = ['DeployedModel']


def existing_snapshots(train_dpath):
    # NOTE: Specific to netharn directory structure
    import parse
    snapshot_dpath = join(train_dpath, 'torch_snapshots/')
    prev_states = sorted(glob.glob(join(snapshot_dpath, '_epoch_*.pt')))
    snapshots = {parse.parse('{}_epoch_{num:d}.pt', path).named['num']: path
                 for path in prev_states}
    return snapshots


def find_best_snapshot(train_dpath):
    """
    Returns snapshot written by monitor if available otherwise takes the last
    one.
    """
    # NOTE: Specific to netharn directory structure
    # Netharn should populate best_snapshot.pt if there is a validation set.
    # Other names are to support older codebases.
    expected_names = [
        'best_snapshot.pt',
        'best_snapshot2.pt',
        'final_snapshot.pt',
        'deploy_snapshot.pt',
    ]
    for snap_fname in expected_names:
        snap_fpath = join(train_dpath, snap_fname)
        if exists(snap_fpath):
            break

    if not exists(snap_fpath):
        snap_fpath = None

    if not snap_fpath:
        epoch_to_fpath = existing_snapshots(train_dpath)
        if epoch_to_fpath:
            snap_fpath = epoch_to_fpath[max(epoch_to_fpath)]
    return snap_fpath


def unpack_model_info(path):
    """
    return paths to the most relevant files in a zip or path deployment.

    If path is not a zipfile, this function expects a netharn fit directory
    structure.

    Args:
        path (PathLike): either a zip deployment or train_dpath
    """
    info = {
        'train_info_fpath': None,
        'snap_fpath': None,
        'model_fpath': None,

        # TODO: need to rename and allow a list of arbitrary files
        'glance': [],  # a list of files in the glance directory
    }
    def populate(root, fpaths):
        # TODO: make more robust
        for fpath in fpaths:
            # FIXME: make this more general and robust
            if fpath.endswith('.json'):
                info['train_info_fpath'] = join(root, fpath)
            if fpath.endswith('.pt'):
                info['snap_fpath'] = join(root, fpath)
            if fpath.endswith('.py'):
                new_fpath = join(root, fpath)
                if info['model_fpath'] is not None:
                    try:
                        # Try to take the most recent path if possible.
                        # This will fail if the file is in a zipfile
                        # (because we should not package multiple models)
                        cur_time = os.stat(info['model_fpath']).st_mtime
                        new_time = os.stat(new_fpath).st_mtime
                        if new_time < cur_time:
                            continue  # Keep the current path
                    except OSError:
                        raise Exception(
                            'Multiple model paths! {} and {}'.format(
                                info['model_fpath'], fpath))
                info['model_fpath'] = new_fpath
            # TODO: make including arbitrary files easier
            if fpath.startswith(('glance/', 'glance\\')):
                info['glance'].append(join(root, fpath))

    if path.endswith('.zip'):
        zipfpath = path
        myzip = zipfile.ZipFile(zipfpath, 'r')
        with zipfile.ZipFile(zipfpath, 'r') as myzip:
            populate(zipfpath, (f.filename for f in myzip.filelist))

    elif exists(path) and isdir(path):
        # Populate core files
        populate(path, os.listdir(path))
        # Populate extra glanceable files
        populate(path, [
            relpath(p, path) for p in glob.glob(join(path, 'glance/*'))])
        # If there are no snapshots in the root directory, then
        # use the latest snapshot from the torch_snapshots dir
        if info['snap_fpath'] is None:
            info['snap_fpath'] = find_best_snapshot(path)

    else:
        raise ValueError('cannot unpack model ' + path)
    return info


def _make_package_name2(info):
    """
    Construct a unique and descriptive name for the deployment
    """
    snap_fpath = info['snap_fpath']
    model_fpath = info['model_fpath']
    train_info_fpath = info['train_info_fpath']

    if train_info_fpath and exists(train_info_fpath):
        train_info = json.load(open(train_info_fpath, 'r'))
        model_name = train_info['hyper']['model'][0].split('.')[-1]
        train_hash = ub.hash_data(train_info['train_id'], hasher='sha512',
                                  base='abc', types=True)[0:8]
    else:
        model_name = os.path.splitext(os.path.basename(model_fpath))[0]
        train_hash = 'UNKNOWN-TRAINID'
        print('WARNING: Train info metadata does not exist')

    try:
        # netharn models contain epoch info in the weights file
        import torch
        state = torch.load(snap_fpath,
                           map_location=lambda storage, location: storage)
        epoch = '{:03d}'.format(state['epoch'])
    except Exception:
        epoch = 'UNKNOWN-EPOCH'

    weights_hash = ub.hash_file(snap_fpath, base='abc',
                                hasher='sha512')[0:6].upper()

    deploy_name = 'deploy_{model}_{trainid}_{epoch}_{weights}'.format(
        model=model_name, trainid=train_hash, epoch=epoch,
        weights=weights_hash)
    return deploy_name


def _package_deploy2(dpath, info):
    """
    Combine the model, weights, and info files into a single deployable file

    Args:
        dpath (PathLike): where to dump the deployment
        info (Dict): containing model_fpath and snap_fpath and optionally
            train_info_fpath and glance, which is a list of extra files.

    Ignore:
        dpath = '/home/joncrall/.cache/netharn/tests/_package_custom'
        path = '/home/joncrall/work/opir/fit/nice/_Sim3-kw6-99-finetune_ML3D_BEST_2018-9-20_LR1e-4_f2_vel0.0_hn0.25_bs64_nr5.0'
        info = unpack_model_info(path)
        zipfpath = _package_deploy2(dpath, info)


    """
    model_fpath = info['model_fpath']
    snap_fpath = info['snap_fpath']
    train_info_fpath = info.get('train_info_fpath', None)

    if not snap_fpath:
        raise FileNotFoundError('No weights are associated with the model')

    deploy_name = _make_package_name2(info)

    deploy_fname = deploy_name + '.zip'

    def zwrite(myzip, fpath, fname=None):
        if fname is None:
            fname = relpath(fpath, dpath)
        myzip.write(fpath, arcname=join(deploy_name, fname))

    zipfpath = join(dpath, deploy_fname)
    with zipfile.ZipFile(zipfpath, 'w') as myzip:
        if train_info_fpath and exists(train_info_fpath):
            zwrite(myzip, train_info_fpath, fname='train_info.json')
        zwrite(myzip, snap_fpath, fname='deploy_snapshot.pt')
        zwrite(myzip, model_fpath, fname=os.path.basename(model_fpath))
        # Add some quick glanceable info
        for p in info.get('glance', []):
            zwrite(myzip, p, fname=join('glance', os.path.basename(p)))
        # for bestacc_fpath in glob.glob(join(train_dpath, 'best_epoch_*')):
        #     zwrite(myzip, bestacc_fpath)
        # for p in glob.glob(join(train_dpath, 'glance/*')):
        #     zwrite(myzip, p)
    print('[DEPLOYER] Deployed zipfpath={}'.format(zipfpath))
    return zipfpath


class DeployedModel(ub.NiceRepr):
    """
    Can setup an initializer and model from a deployed zipfile or a train path

    CommandLine:
        xdoctest -m netharn.export.deployer DeployedModel

    Example:
        >>> # Test the train folder as the model deployment
        >>> train_dpath = _demodata_trained_dpath()
        >>> self = DeployedModel(train_dpath)
        >>> model_ = self.model_definition()
        >>> initializer_ = self.initializer_definition()
        >>> model = model_[0](**model_[1])
        >>> initializer = initializer_[0](**initializer_[1])
        >>> initializer(model)
        ...
        >>> print('model.__module__ = {!r}'.format(model.__module__))
        model.__module__ = 'ToyNet2d_2a3f49'

    Example:
        >>> # Test the zip file as the model deployment
        >>> zip_fpath = _demodata_zip_fpath()
        >>> self = DeployedModel(zip_fpath)
        >>> model_ = self.model_definition()
        >>> initializer_ = self.initializer_definition()
        >>> model = model_[0](**model_[1])
        >>> initializer = initializer_[0](**initializer_[1])
        >>> initializer(model)
        ...
        >>> print('model.__module__ = {!r}'.format(model.__module__))
        model.__module__ = 'deploy_ToyNet2d_rljhgepw_000_.../ToyNet2d_2a3f49'
    """
    def __init__(self, path):
        self.path = path
        self._model = None
        self._info = None

    @classmethod
    def custom(DeployedModel, snap_fpath, model, initkw=None, train_info_fpath=None):
        """
        Create a deployed model even if the model wasnt trained with FitHarn

        This just requires specifying a bit more information, which FitHarn
        would have tracked.

        Args:
            snap_fpath (PathLike):
                path to the exported weights file

            model (PathLike or nn.Module): can either be
                (1) a path to model topology (created via `export_model_code`)
                (2) the model class or an instance of the class

            initkw (Dict): if model is a class or instance, then
                you must pass the keyword arguments used to construct it.

            train_info_fpath (PathLike, optional):
                path to a json file containing additional training metadata

        Example:
            >>> # Setup raw components
            >>> train_dpath = _demodata_trained_dpath()
            >>> deployed = DeployedModel(train_dpath)
            >>> snap_fpath = deployed.info['snap_fpath']
            >>> model, initkw = deployed.model_definition()
            >>> train_info_fpath = deployed.info['train_info_fpath']
            >>> # Past raw components to custom
            >>> self = DeployedModel.custom(snap_fpath, model, initkw)
            >>> dpath = ub.ensure_app_cache_dir('netharn', 'tests/_package_custom')
            >>> self.package(dpath)

        Ignore:
            from netharn.export.deployer import *
            fcnn116 = ub.import_module_from_path(ub.truepath('~/remote/hermes/tmp/fcnn116.py'))
            model = fcnn116.FCNN116()
            initkw = {}
            snap_fpath = ub.truepath('~/remote/hermes/tmp/fcnn116.pt')
            train_info_fpath = None
            self = DeployedModel.custom(snap_fpath, model, initkw)
            zipfile = self.package(dpath)

            loaded = DeployedModel(zipfile).load_model()
        """
        if isinstance(model, six.string_types):
            model_fpath = model
            if initkw is not None:
                raise ValueError('initkw not used when model is a path')
        else:
            import tempfile
            from netharn.export import exporter
            dpath = tempfile.mkdtemp()
            model_fpath = exporter.export_model_code(dpath, model, initkw=initkw)

        _info = {
            'model_fpath': model_fpath,
            'snap_fpath': snap_fpath,
            'train_info_fpath': train_info_fpath,
        }
        self = DeployedModel(None)
        self._info = _info
        return self

    def __nice__(self):
        return self.__json__()

    def __json__(self):
        if self.path is None:
            if self._info:
                return ub.repr2(self._info, nl=0)
        else:
            return self.path

    def package(self, dpath=None):
        """
        If self.path is a directory, packages important info into a deployable
        zipfile.

        Args:
            dpath (PathLike, optional): directory to dump your packaged model.
                If not specified, it uses the netharn train_dpath if available.

        Returns:
            PathLike: path to single-file deployment
        """
        if dpath is None:
            if self.path is None:
                raise ValueError('Must specify dpath for custom deployments')
            else:
                if self.path.endswith('.zip'):
                    raise Exception('Deployed model is already a package')
                dpath = self.path

        zip_fpath = _package_deploy2(dpath, self.info)
        return zip_fpath

    @property
    def info(self):
        if self._info is None:
            self._info = self.unpack_info()
        return self._info

    def unpack_info(self):
        return unpack_model_info(self.path)

    def model_definition(self):
        model_fpath = self.info['model_fpath']
        module = ub.import_module_from_path(model_fpath)

        export_version = getattr(module, '__pt_export_version__', '0')
        export_version = list(map(int, export_version.split('.')))
        if export_version >= [0, 2, 0]:
            model_cls = module.get_model_cls()
            initkw = module.get_initkw()
        else:
            # Hack to get information from older versions of pytorch_export
            import inspect
            from xdoctest import static_analysis
            print('Hacking to grab model_cls and initkw')
            model = module.make()
            model_cls = model.__class__
            source = inspect.getsource(module.make)
            print(source)
            initkw = static_analysis.parse_static_value('initkw', source=source)
            # Try to reconstruct initkw
        model_ = (model_cls, initkw)
        return model_

    def initializer_definition(self):
        import netharn as nh
        initializer_ = (nh.initializers.Pretrained,
                        {'fpath': self.info['snap_fpath']})
        return initializer_

    def train_info(self):
        import netharn as nh
        train_info_fpath = self.info.get('train_info_fpath', None)
        if train_info_fpath is not None:
            train_info = json.load(nh.util.zopen(train_info_fpath, 'r'))
        else:
            train_info = None
        return train_info

    def load_model(self):
        if self._model is not None:
            return self._model

        model_cls, model_kw = self.model_definition()
        model = model_cls(**model_kw)

        if True:
            # Always load models onto the CPU first
            # import netharn as nh
            model = model.to('cpu')
            # devices = {k: item.device for k, item in model.state_dict().items()}
            # nh.XPU.from_data(model)

        # TODO: load directly from instead of using initializer self.info['snap_fpath']?
        # Actually we can't because we lose the zopen stuff. Its probably ok
        # To depend on netharn a little bit.
        # import torch
        # info = self.unpack_info()
        # state_dict = torch.load(self.info['snap_fpath'])
        # model.load_state_dict()

        initializer_ = self.initializer_definition()
        initializer = initializer_[0](**initializer_[1])

        assert model is not None

        initializer(model)
        return model


def _demodata_zip_fpath():
    zip_path = DeployedModel(_demodata_trained_dpath()).package()
    return zip_path


def _demodata_trained_dpath():
    # This will train a toy model with toy data using netharn
    import netharn as nh
    hyper = nh.HyperParams(**{
        'workdir'     : ub.ensure_app_cache_dir('netharn/tests/deploy'),
        'nice'        : 'deploy_demo_static',
        'xpu'         : nh.XPU.cast('cpu'),
        'datasets'    : {'train': nh.data.ToyData2d(size=3, rng=0)},
        'loaders'     : {'batch_size': 64},
        'model'       : (nh.models.ToyNet2d, {}),
        'optimizer'   : (nh.optimizers.SGD, {'lr': 0.0001}),
        'criterion'   : (nh.criterions.FocalLoss, {}),
        'initializer' : (nh.initializers.KaimingNormal, {}),
        'monitor'     : (nh.Monitor, {'max_epoch': 1}),
    })
    harn = nh.FitHarn(hyper)
    harn.run()  # TODO: make this run faster if we don't need to rerun
    if len(list(glob.glob(join(harn.train_dpath, '*.py')))) > 1:
        # If multiple models are deployed some hash changed. Need to reset
        harn.initialize(reset='delete')
        harn.run()  # don't relearn if we already finished this one
    return harn.train_dpath


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.export.deployer
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
