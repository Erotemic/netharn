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
import ubelt as ub
import warnings
import zipfile
import os
from os.path import exists
from os.path import isdir
from os.path import join
from os.path import relpath

__all__ = ['DeployedModel']


def existing_snapshots(train_dpath):
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


def _package_deploy(train_dpath):
    """
    Combine the model, weights, and info files into a single deployable file

    CommandLine:
        xdoctest -m netharn.export.deployer _package_deploy

    Args:
        train_dpath (PathLike): the netharn training directory

    Example:
        >>> dpath = ub.ensure_app_cache_dir('netharn', 'tests/_package_deploy')
        >>> train_dpath = ub.ensuredir((dpath, 'my_train_dpath'))
        >>> ub.touch(join(train_dpath, 'final_snapshot.pt'))
        >>> ub.touch(join(train_dpath, 'my_model.py'))
        >>> zipfpath = _package_deploy(train_dpath)
        ...
        >>> print(os.path.basename(zipfpath))
        deploy_UNKNOWN-ARCH_my_train_dpath_UNKNOWN-EPOCH_QOOEZT.zip
    """
    print('[DEPLOYER] Deploy to dpath={}'.format(train_dpath))
    snap_fpath = find_best_snapshot(train_dpath)

    model_fpaths = glob.glob(join(train_dpath, '*.py'))
    if len(model_fpaths) == 0:
        raise FileNotFoundError('The model topology cannot be found')
    elif len(model_fpaths) > 1:
        warnings.warn('There are multiple models here: {}'.format(model_fpaths))

    if not snap_fpath:
        raise FileNotFoundError('No weights are associated with the model')

    weights_hash = ub.hash_file(snap_fpath, base='abc', hasher='sha512')[0:6].upper()

    train_info_fpath = join(train_dpath, 'train_info.json')

    if exists(train_info_fpath):
        train_info = json.load(open(train_info_fpath, 'r'))
        model_name = train_info['hyper']['model'][0].split('.')[-1]
        train_hash = ub.hash_data(train_info['train_id'], hasher='sha512',
                                  base='abc', types=True)[0:8]
    else:
        model_name = 'UNKNOWN-ARCH'
        train_hash = os.path.basename(train_dpath)
        print('WARNING: Training metadata does not exist')

    try:
        import torch
        state = torch.load(snap_fpath)
        epoch = '{:03d}'.format(state['epoch'])
    except Exception:
        epoch = 'UNKNOWN-EPOCH'

    deploy_name = 'deploy_{model}_{trainid}_{epoch}_{weights}'.format(
        model=model_name,
        trainid=train_hash,
        epoch=epoch,
        weights=weights_hash)

    deploy_fname = deploy_name + '.zip'

    def zwrite(myzip, fpath, fname=None):
        if fname is None:
            fname = relpath(fpath, train_dpath)
        myzip.write(fpath, arcname=join(deploy_name, fname))

    zipfpath = join(train_dpath, deploy_fname)
    with zipfile.ZipFile(zipfpath, 'w') as myzip:
        if exists(train_info_fpath):
            zwrite(myzip, train_info_fpath)
        zwrite(myzip, snap_fpath, fname='deploy_snapshot.pt')
        for model_fpath in model_fpaths:
            zwrite(myzip, model_fpath)
        # Add some quick glanceable info
        # for bestacc_fpath in glob.glob(join(train_dpath, 'best_epoch_*')):
        #     zwrite(myzip, bestacc_fpath)
        for p in glob.glob(join(train_dpath, 'glance/*')):
            zwrite(myzip, p)
    print('[DEPLOYER] Deployed zipfpath={}'.format(zipfpath))
    return zipfpath


def unpack_model_info(path):
    """
    return paths to the most relevant files in a zip or path deployment

    Args:
        path (PathLike): either a zip deployment or train_dpath
    """
    info = {
        'train_info_fpath': None,
        'snap_fpath': None,
        'model_fpath': None,
    }
    def populate(root, fpaths):
        # TODO: make more robust
        for fpath in fpaths:
            if fpath.endswith('.json'):
                info['train_info_fpath'] = join(root, fpath)
            if fpath.endswith('.pt'):
                info['snap_fpath'] = join(root, fpath)
            if fpath.endswith('.py'):
                if info['model_fpath'] is not None:
                    raise Exception('Multiple model paths!')
                info['model_fpath'] = join(root, fpath)

    if path.endswith('.zip'):
        zipfpath = path
        myzip = zipfile.ZipFile(zipfpath, 'r')
        with zipfile.ZipFile(zipfpath, 'r') as myzip:
            populate(zipfpath, (f.filename for f in myzip.filelist))
    elif exists(path) and isdir(path):
        populate(path, os.listdir(path))
        # If there are no snapshots in the root directory, then
        # use the latest snapshot from the torch_snapshots dir
        if info['snap_fpath'] is None:
            info['snap_fpath'] = find_best_snapshot(path)
    else:
        raise ValueError('cannot unpack model ' + path)
    return info


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
        model.__module__ = 'deploy_ToyNet2d_rljhgepw_001_.../ToyNet2d_2a3f49'
    """
    def __init__(self, path):
        self.path = path

        self._model = None

    def __nice__(self):
        return self.path

    def package(self):
        """
        If self.path is a directory, packages important info into a deployable
        zipfile.
        """
        if self.path.endswith('.zip'):
            raise Exception('Deployed model is already a package')

        zip_fpath = _package_deploy(self.path)
        return zip_fpath

    def unpack_info(self):
        return unpack_model_info(self.path)

    def model_definition(self):
        info = self.unpack_info()

        model_fpath = info['model_fpath']
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
        info = self.unpack_info()
        initializer_ = (nh.initializers.Pretrained,
                        {'fpath': info['snap_fpath']})
        return initializer_

    def train_info(self):
        import netharn as nh
        info = self.unpack_info()
        train_info_fpath = info.get('train_info_fpath', None)
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

        # TODO: load directly from instead of using initializer info['snap_fpath']?
        # Actually we can't because we lose the zopen stuff. Its probably ok
        # To depend on netharn a little bit.
        # import torch
        # info = self.unpack_info()
        # state_dict = torch.load(info['snap_fpath'])
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
