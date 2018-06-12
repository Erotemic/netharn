# -*- coding: utf-8 -*-
"""
CommandLine:
    python ~/code/netharn/netharn/fit_harn.py __doc__

Notes:
    when profiling ensure CUDA_LAUNCH_BLOCKING=1

Notes:
    to use, your training session must have the concept of:
        * epochs
        * batch_size
        * xpu
        * train / validation datasets

    or better yet:
        * a model
        * a criterion
        * an optimizer

TODO:
    [ ] - output "glance" curves to disk
    [x] - move logs to a logs folder. Keep a single master log in the root
    [ ] - Why didnt the best_snapshot.pt get saved in the most recent yolo run?

Example:
    >>> import netharn as nh
    >>> size = 3
    >>> max_epoch = 10
    >>> datasets = {
    >>>     'train': nh.data.ToyData2d(size=size, border=1, n=256, rng=0),
    >>>     'vali': nh.data.ToyData2d(size=size, border=1, n=128, rng=1),
    >>> }
    >>> hyper = {
    >>>     # --- Data First
    >>>     'datasets'    : datasets,
    >>>     'nice'        : 'demo',
    >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/demo'),
    >>>     'loaders'     : {'batch_size': 64},
    >>>     'xpu'         : nh.XPU.cast('auto'),
    >>>     # --- Algorithm Second
    >>>     'model'       : (nh.models.ToyNet2d, {}),
    >>>     'optimizer'   : (nh.optimizers.SGD, {
    >>>         'lr': 0.0001
    >>>     }),
    >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
    >>>     #'criterion'   : (nh.criterions.FocalLoss, {}),
    >>>     'initializer' : (nh.initializers.KaimingNormal, {
    >>>         'param': 0,
    >>>     }),
    >>>     'scheduler'   : (nh.schedulers.ListedLR, {
    >>>         'points': {0: .0001, 2: .01, 5: .015, 6: .005, 9: .001},
    >>>         'interpolate': True,
    >>>     }),
    >>>     'dynamics'   : {'batch_step': 4},
    >>>     'monitor'     : (nh.Monitor, {
    >>>         'max_epoch': max_epoch,
    >>>     }),
    >>> }
    >>> harn = FitHarn(hyper)
    >>> harn.config['use_tqdm'] = 1
    >>> harn.initialize(reset='delete')
    >>> harn.run()
"""
import glob
import itertools as it
import logging
import numpy as np
import os
import parse
import shutil
import time
import torch
import ubelt as ub
import sys
from os.path import join
import warnings
import functools

from netharn import device  # NOQA
from netharn import folders
from netharn import hyperparams

from netharn import util
from netharn.util import profiler
from xdoctest.utils import strip_ansi

try:
    import tensorboard_logger
except ImportError:
    tensorboard_logger = None


__all__ = ['FitHarn']


MIXINS = []


class StopTraining(StopIteration):
    """
    Signals that training should terminate
    """
    pass


class CannotResume(Exception):
    pass


class TrainingDiverged(Exception):
    pass


def register_mixin(cls):
    MIXINS.append(cls)
    return cls


def _disjoint_dict_update(a, b):
    """
    Equivalent to a.update(b), but raises KeyError if a and b are not disjoint
    """
    if b:
        isect = set(a).intersection(set(b))
        if isect:
            raise KeyError('Conflicting keys: {}'.format(isect))
        a.update(b)


@register_mixin
class ExtraMixins:
    def _demo_batch(harn, index=0, tag='train', raw=False):
        """
        Returns a single batch for testing / demo purposes.

        Additionally, sets harn.current_tag to `tag`.

        Args:
            index (int): get the `index`-th batch
            tag (str): get batch from either train, vali, or test loader
            raw (bool): if True, does not prepare the batch

        Returns:
            object: output of the data loader
        """
        loader = harn.loaders[tag]
        harn.current_tag = tag
        for bx, batch in enumerate(iter(loader)):
            if bx >= index:
                break
        if raw:
            return batch
        else:
            return harn.prepare_batch(batch)

    def _check_thread_safety(harn):
        """
        References:
            https://github.com/pytorch/pytorch/issues/1355
        """
        import cv2
        n_workers = max(loader.num_workers for loader in harn.loaders.values())
        if n_workers > 1:
            n_threads = cv2.getNumThreads()
            if n_threads > 1:
                msg = ('OpenCV threadcount of {} is non-zero and a DataLoader '
                       'is using {} workers. This may cause deadlocks '
                       'To be safe use cv2.setNumThreads(0)').format(
                           n_threads, n_workers)
                warnings.warn(msg, RuntimeWarning)
                harn.warn(msg)


@register_mixin
class InitializeMixin:

    def initialize(harn, reset=False):
        """
        Uses the hyper parameters to initialize the necessary resources and
        restart from previously
        """
        if harn.train_dpath is None:
            harn.setup_paths()
        else:
            ub.ensuredir(harn.train_dpath)

        if reset == 'delete':
            ub.delete(harn.train_dpath)
            ub.ensuredir(harn.train_dpath)

        harn.setup_loggers()

        harn.setup_modules()

        assert harn.model is not None, 'required module'
        assert harn.optimizer is not None, 'required module'
        assert harn.monitor is not None, 'required module'

        try:
            harn.resume_from_previous_snapshots()
        except CannotResume:
            harn.reset_weights()

        if harn.train_dpath:
            harn.log(' * harn.train_dpath = {!r}'.format(harn.train_dpath))
            harn.log(' * harn.nice_dpath = {!r}'.format(harn.nice_dpath))
            harn.log('Snapshots will save to harn.snapshot_dpath = {!r}'.format(
                harn.snapshot_dpath))
        else:
            harn.warn('harn.train_dpath is None, all computation is in memory')
        harn._initialized = True

    def setup_paths(harn):
        if harn.hyper is None:
            harn.warn('harn.train_dpath is None, cannot setup_paths')
        else:
            paths = folders.Folders(hyper=harn.hyper)
            train_info = paths.setup_dpath()
            harn.nice_dpath = train_info['nice_dpath']
            harn.train_dpath = train_info['train_dpath']
            return harn.train_dpath

    def setup_loggers(harn):
        """
        Setup file logging and / or tensorboard logging
        """
        if harn.train_dpath is None:
            harn.warn('harn.train_dpath is None, cannot setup loggers')
            return

        use_file_logger = True
        if use_file_logger and harn.flog is None:

            flog = logging.getLogger(harn.__class__.__name__)
            formatter = logging.Formatter('%(asctime)s : %(message)s')

            # Add timestamped fpath write handler
            flog_fname = 'fitlog_{}.log'.format(ub.timestamp())
            flog_dpath = ub.ensuredir(join(harn.train_dpath, 'logs'))
            w_flog_fpath = join(flog_dpath, flog_fname)
            w_handler = logging.FileHandler(w_flog_fpath, mode='w')
            w_handler.setFormatter(formatter)

            # Add a simple root append handler
            a_flog_fpath = join(harn.train_dpath, 'fit.log')
            a_handler = logging.FileHandler(a_flog_fpath, mode='w')
            a_handler.setFormatter(formatter)

            flog.propagate = False
            flog.setLevel(logging.DEBUG)
            flog.addHandler(w_handler)
            flog.addHandler(a_handler)
            harn.flog = flog
            harn.debug('initialized file logger')
            # flog_link = join(harn.train_dpath, 'fit.log')
            # ub.symlink(flog_fpath, flog_link, overwrite=True)

        if tensorboard_logger:
            train_base = os.path.dirname(harn.nice_dpath or harn.train_dpath)
            harn.log('dont forget to start: tensorboard --logdir ' + train_base)
            harn.log('Initializing tensorboard')
            harn.tlog = tensorboard_logger.Logger(harn.train_dpath,
                                                     flush_secs=2)
        else:
            harn.log('Tensorboard is not available')

    def setup_modules(harn):
        """
        Construts the basic modules to be used by the harness, i.e:
            loaders, xpu, model, criterion, optimizer, initializer, scheduler,
            monitor, and dynamics.
        """
        if harn.hyper is None:
            raise ValueError(
                'Hyperparameters not specified, must setup modules yourself')

        harn.debug('make XPU')
        harn.xpu = harn.hyper.make_xpu()
        harn.debug('harn.xpu = {!r}'.format(harn.xpu))
        harn.xpu.set_as_default()

        if harn.hyper.criterion_cls:
            harn.log('Criterion: {}'.format(harn.hyper.criterion_cls.__name__))
        else:
            harn.log('Criterion: Custom')

        harn.log('Optimizer: {}'.format(harn.hyper.optimizer_cls.__name__))

        if harn.hyper.scheduler_cls:
            harn.log('Scheduler: {}'.format(harn.hyper.scheduler_cls.__name__))
        else:
            harn.log('No Scheduler')

        harn.debug('Making loaders')
        harn.datasets = harn.hyper.datasets
        harn.loaders = harn.hyper.make_loaders()

        harn.debug('Making model')
        harn.model = harn.hyper.make_model()
        harn.log('Mounting {} model on {}'.format(
            harn.model.__class__.__name__, harn.xpu))
        harn.model = harn.xpu.mount(harn.model)

        harn.debug('Making initializer')
        harn.initializer = harn.hyper.make_initializer()

        n_params = util.number_of_parameters(harn.model)
        harn.log('Model has {!r} parameters'.format(n_params))

        harn.criterion = harn.hyper.make_criterion()
        harn.log('Move {} model to {}'.format(harn.criterion, harn.xpu))
        harn.criterion = harn.xpu.move(harn.criterion)

        harn.log('Make optimizer')
        harn.optimizer = harn.hyper.make_optimizer(harn.model.parameters())

        harn.log('Make scheduler')
        harn.scheduler = harn.hyper.make_scheduler(harn.optimizer)

        harn.log('Make monitor')
        harn.monitor = harn.hyper.make_monitor()

        harn.log('Make dynamics')
        harn.dynamics = harn.hyper.dynamics.copy()

    def reset_weights(harn):
        """
        Use the initializer to set the weights for the model
        """
        harn.log('Initializing new model')
        if harn.initializer:
            if harn.initializer.__class__.__name__ == 'LSUV':
                #hack LSUV needs a batch of data to run
                with util.grad_context(False):
                    loader = harn.loaders['train']
                    input, labels = next(iter(loader))
                    data = harn.xpu.variable(input)
                    harn.initializer(harn.model, data)
            else:
                harn.initializer(harn.model)
        else:
            harn.warn('initializer was not specified')

        if not harn.dry:
            for group in harn.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

    def resume_from_previous_snapshots(harn):
        """
        Attempts to load one of the states in prev_states
        """
        if harn.train_dpath is None:
            raise CannotResume('harn.train_dpath is None')

        prev_states = harn.prev_snapshots()
        harn.log('There are {} existing snapshots'.format(len(prev_states)))
        if not prev_states:
            raise CannotResume('no previous snapshots')

        harn.log('Loading previous states')
        success = False
        # Ignore corrupted snapshots
        for load_path in reversed(prev_states):
            try:
                harn.load_snapshot(load_path)
            except RuntimeError:
                harn.log('Failed to load {}. Skiping.'.format(load_path))
            else:
                success = True
                break
        if not success:
            raise CannotResume('Previous snapshots are invalid or corrupted')

        for i, group in enumerate(harn.optimizer.param_groups):
            if 'initial_lr' not in group:
                raise KeyError(
                    'param "initial_lr" is not specified '
                    'in param_groups[{}] when resuming an optimizer'.format(i))

        harn.log('Resuming from epoch={}'.format(harn.epoch))


@register_mixin
class ProgMixin:
    def _make_prog(harn, **kw):
        if harn.config['use_tqdm'] is None:
            harn.config['use_tqdm'] = harn.config['prog_backend'] == 'tqdm'
            if harn.config['prog_backend'] not in {'tqdm', 'progiter'}:
                raise KeyError(harn.config['prog_backend'])
        if harn.config['use_tqdm']:
            import tqdm
            Prog = tqdm.tqdm
        else:
            Prog = functools.partial(ub.ProgIter, verbose=1)
        return Prog(**kw)

    def _batch_msg(harn, metric_dict, batch_size):
        bs = 'x{}'.format(batch_size)
        metric_parts = ['{}:{:.3f}'.format(k, v) for k, v in metric_dict.items()]
        msg = ' │ ' .join([bs] + metric_parts) + ' │'
        return msg

    def _close_prog(harn):
        if harn.main_prog is not None:
            harn.main_prog.close()
            harn.main_prog = None
            sys.stdout.write('\n\n\n\n')  # fixes progress bar formatting

    def _update_prog_postfix(harn, prog):
        if harn.config['use_tqdm']:
            prog.set_postfix({
                'wall': time.strftime('%h:%m') + ' ' + time.tzname[0]
            })

    def _update_main_prog_desc(harn):
        lrs = harn._current_lrs()
        lr_str = ','.join(['{:.2g}'.format(lr) for lr in lrs])
        desc = 'epoch lr:{} │ {}'.format(lr_str, harn.monitor.message())
        harn.debug(desc)
        harn.main_prog.set_description(desc, refresh=False)
        if isinstance(harn.main_prog, ub.ProgIter):
            if not harn.main_prog.started:
                # harn.main_prog.ensure_newline()
                harn.main_prog.clearline = False
                harn.main_prog.freq = 1
                harn.main_prog.adjust = False
                harn.main_prog.begin()
        else:
            harn._update_prog_postfix(harn.main_prog)


@register_mixin
class LogMixin:
    def log(harn, msg):
        harn.debug(msg)
        print(msg)

    def debug(harn, msg):
        if harn.flog:
            msg = strip_ansi(msg)
            harn.flog.debug(msg)

    def error(harn, msg):
        print(msg)
        if harn.flog:
            msg = strip_ansi(msg)
            harn.flog.error(msg)

    def warn(harn, msg):
        if harn.flog:
            msg = strip_ansi(msg)
            harn.flog.warn(msg)
        print(msg)

    def log_value(harn, key, value, n_iter):
        if harn.tlog:
            harn.tlog.log_value(key, value, n_iter)
        harn.debug('log_value({}, {}, {}'.format(key, value, n_iter))

    def log_histogram(harn, key, value, n_iter):
        if harn.tlog:
            harn.tlog.log_histogram(key, value, n_iter)

    def log_images(harn, key, value, n_iter):
        if harn.tlog:
            harn.tlog.log_images(key, value, n_iter)


@register_mixin
class SnapshotMixin:

    @property
    def snapshot_dpath(harn):
        if harn.train_dpath is None:
            raise ValueError('harn.train_dpath is None')
        return join(harn.train_dpath, 'torch_snapshots')

    def _epochs_to_remove(harn, existing_epochs, num_keep_recent,
                          num_keep_best, keep_freq):
        """
        Unit testable helper for `cleanup_snapshots`. Determines which epochs
        to remove given which epoches exist.

        Keeps `keep_freq` most recent, `num_keep_best` best, and one every
        `keep_freq` epochs.

        Doctest:
            >>> import netharn as nh
            >>> harn = FitHarn({})
            >>> rng = np.random.RandomState(0)
            >>> harn.monitor = nh.Monitor(minimize=['loss'], maximize=['miou'])
            >>> for epoch in range(200):
            >>>     harn.monitor.update(epoch, {'loss': rng.rand(),
            >>>                                 'miou': rng.rand()})
            >>> existing_epochs = list(range(0, 200, 4))
            >>> num_keep_best = 10
            >>> num_keep_recent = 10
            >>> keep_freq = 10
            >>> to_remove = harn._epochs_to_remove(existing_epochs,
            >>>                                    num_keep_recent, num_keep_best,
            >>>                                    keep_freq)
            >>> assert len(existing_epochs) - len(to_remove) < 40
        """
        keep = set()

        recent = existing_epochs[-num_keep_recent:]
        keep.update(recent)

        # TODO: add a config for always keeping specific iterations in
        # multiples of X.

        if harn.monitor:
            for best_epochs in harn.monitor.best_epochs().values():
                best = ub.oset(best_epochs).intersection(existing_epochs)
            keep.update(best[:num_keep_best])

        # Keep a strided sampling of epochs
        epoch_arr = np.array(existing_epochs)
        flags = ((epoch_arr % keep_freq) == 0)
        sampled = epoch_arr[flags]
        keep.update(sampled)

        to_remove = set(existing_epochs) - keep
        return to_remove

    def cleanup_snapshots(harn):
        """
        remove old snapshots

        TODO:
            [ ] - keep the top epochs for every metric
        """
        snapshots = harn.prev_snapshots()
        existing_epochs = sorted([
            int(parse.parse('{}_epoch_{num:d}.pt', path).named['num'])
            for path in snapshots
        ])

        num_keep_recent = harn.config['num_keep']
        num_keep_best = harn.config['num_keep']
        keep_freq = harn.config['keep_freq']

        epoch_to_fpath = dict(zip(existing_epochs, snapshots))
        to_remove = harn._epochs_to_remove(existing_epochs, num_keep_recent,
                                           num_keep_best, keep_freq)
        for fpath in ub.take(epoch_to_fpath, to_remove):
            ub.delete(fpath)

    def prev_snapshots(harn):
        ub.ensuredir(harn.snapshot_dpath)
        prev_states = sorted(glob.glob(join(harn.snapshot_dpath, '_epoch_*.pt')))
        return prev_states

    def load_snapshot(harn, load_path):
        """
        Sets the harness to its state just after an epoch finished

        Args:
            str: path to previously saved snapshot
        """
        harn.log('Loading previous state: {}'.format(load_path))
        snapshot_state = harn.xpu.load(load_path)
        harn.set_snapshot_state(snapshot_state)
        harn.log('Previous snapshot loaded...')

    def save_snapshot(harn):
        # save snapshot
        ub.ensuredir(harn.snapshot_dpath)
        save_fname = '_epoch_{:08d}.pt'.format(harn.epoch)
        safe_fpath = join(harn.snapshot_dpath, save_fname)
        harn.debug('Saving snapshot to {}'.format(safe_fpath))
        snapshot_state = harn.get_snapshot_state()
        if harn.dry:
            harn.debug('Dry run, not saving to {}'.format(safe_fpath))
        else:
            torch.save(snapshot_state, safe_fpath)
            harn.debug('Snapshot saved to {}'.format(safe_fpath))
            return safe_fpath

    def backtrack_weights(harn, epoch):
        """
        Reset the weights to a previous good state
        """
        load_path = join(harn.snapshot_dpath, '_epoch_{:08d}.pt'.format(epoch))
        snapshot = harn.xpu.load(load_path)

        print('\n\n\n\n')
        harn.log('Backtracking to weights from previous state: {}'.format(load_path))
        # only load the model state to simulate a big step back
        harn.model.load_state_dict(snapshot['model_state_dict'])
        harn.optimizer.zero_grad()


@register_mixin
class ScheduleMixin:

    def _current_lrs(harn):
        """
        Get the of distinct learning rates (usually only 1) currently in use
        """
        optim_lrs = {group['lr'] for group in harn.optimizer.param_groups}

        if harn.scheduler is None:
            if harn.optimizer is None:
                assert harn.dry
                lrs = [.01]
            else:
                lrs = set(map(lambda group: group['lr'], harn.optimizer.param_groups))
        elif hasattr(harn.scheduler, '_current_lrs'):
            lrs = set(harn.scheduler._current_lrs())
        elif hasattr(harn.scheduler, 'get_lr'):
            lrs = set(harn.scheduler.get_lr())
        else:
            # workaround for ReduceLROnPlateau
            lrs = {group['lr'] for group in harn.scheduler.optimizer.param_groups}

        if optim_lrs != lrs:
            raise AssertionError('optimizer and scheduler are out of sync')
        return lrs

    def _check_termination(harn):
        if harn.epoch >= harn.monitor.max_epoch:
            harn._close_prog()
            harn.log('Maximum harn.epoch reached, terminating ...')
            return True
        if harn.monitor.is_done():
            harn._close_prog()
            harn.log('Validation set is not improving, terminating ...')
            return True
        return False

    def _step_scheduler(harn, improved=None):
        """
        helper function to change the learning rate that handles the way that
        different schedulers might be used.
        """
        if harn.scheduler is None:
            pass
        elif harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            assert improved is not None, 'must validate for ReduceLROnPlateau schedule'

            def hack_lr_step(self, improved, epoch=None):
                if epoch is None:
                    epoch = self.last_epoch = self.last_epoch + 1
                self.last_epoch = epoch

                if improved:
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1

                if self.in_cooldown:
                    self.cooldown_counter -= 1
                    self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

                if self.num_bad_epochs > self.patience:
                    self._reduce_lr(epoch)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0

                    # todo: make a pytorch pr where there is a callback on
                    # lr_reduction.
                    # the scheduler has stepped, we should now backtrack the
                    # weights to the previous best state
                    backtrack = False
                    if backtrack:
                        harn.backtrack_weights(harn.monitor.best_epoch)

            # # hack to determine if the rlrop scheduler stepped
            hack_lr_step(harn.scheduler, improved)
        else:
            harn.scheduler.step(epoch=harn.epoch)


@register_mixin
class CoreMixin:
    """
    The core main execution loop
    """
    def run(harn):
        """
        main training loop
        """
        if not harn._initialized:
            harn.initialize()

        harn.log('begin training')

        if harn._check_termination():
            return

        harn.main_prog = harn._make_prog(desc='epoch',
                                         total=harn.monitor.max_epoch,
                                         disable=not harn.config['show_prog'],
                                         leave=True, dynamic_ncols=True,
                                         position=0, initial=harn.epoch)
        harn._update_main_prog_desc()

        # Loader dict should be ordered
        harn.loaders = ub.odict([
            (key, harn.loaders[key]) for key in ['train', 'vali', 'test']
            if key in harn.loaders
        ])

        train_loader = harn.loaders['train']
        vali_loader  = harn.loaders.get('vali', None)
        test_loader  = harn.loaders.get('test', None)

        harn._check_thread_safety()

        if not vali_loader:
            # if harn.monitor:
            #     harn.warn('Need a validation set to use nh.Monitor')
            if harn.scheduler:
                if harn.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    raise ValueError('need a validataion dataset to use ReduceLROnPlateau')

        # keep track of moving metric averages across epochs
        harn._run_metrics = {
            tag: util.WindowedMovingAve(window=len(loader))
            for tag, loader in harn.loaders.items()
        }

        # if harn.scheduler:
        #     # prestep scheduler?
        #     if getattr(harn.scheduler, 'last_epoch', 0) == -1:
        #         harn.scheduler.step()

        # Clear any existing gradients before starting
        harn.optimizer.zero_grad()

        try:
            for harn.epoch in it.count(harn.epoch):
                harn._run_tagged_epochs(train_loader, vali_loader, test_loader)
        except StopTraining:
            pass
        except Exception as ex:
            print('\n\n\n')
            harn.error('an {} error occurred in the train loop: {}'.format(
                type(ex), repr(ex)))
            import traceback
            tb = traceback.format_exc()
            harn.log(tb)
            harn._close_prog()
            raise

        harn.log('\n\n\n')
        harn.log('training completed')
        harn.log('current lrs: {}'.format(harn._current_lrs()))

        harn.on_complete()
        harn.log('exiting fit harness.')

    @profiler.profile
    def _run_tagged_epochs(harn, train_loader, vali_loader, test_loader):
        """
        Runs one epoch of train, validation, and testing
        """
        harn.debug('=== start epoch {} ==='.format(harn.epoch))

        current_lr = max(harn._current_lrs())
        harn.log_value('epoch lr', current_lr, harn.epoch)

        # run training epoch
        harn._run_epoch(train_loader, tag='train', learn=True)

        # run validation epoch
        improved = False
        if vali_loader and harn.check_interval('vali', harn.epoch):
            vali_metrics = harn._run_epoch(vali_loader, tag='vali',
                                           learn=False)
            improved = harn.monitor.update(harn.epoch, vali_metrics)
            harn._update_main_prog_desc()

        # run test epoch
        if test_loader and harn.check_interval('test', harn.epoch):
            harn._run_epoch(test_loader, tag='test', learn=False)

        if harn.train_dpath is not None:
            if improved:
                save_fpath = harn.save_snapshot()
                if save_fpath:
                    harn.debug('new best_snapshot {}'.format(save_fpath))
                    # copy the best snapshot the the main directory
                    best_path = join(harn.train_dpath, 'best_snapshot.pt')
                    shutil.copy2(save_fpath, best_path)
            else:
                # todo: allow monitor to clean up old snapshots
                if harn.check_interval('snapshot', harn.epoch):
                    save_fpath = harn.save_snapshot()

            if harn.check_interval('cleanup', harn.epoch):
                harn.cleanup_snapshots()

        harn.main_prog.update(1)

        # check for termination
        if harn._check_termination():
            raise StopTraining()

        # change learning rate (modified optimizer inplace)
        harn._step_scheduler(improved)

        harn._update_main_prog_desc()

    @profiler.profile
    def _run_epoch(harn, loader, tag, learn=False):
        """
        evaluate the model on test / train / or validation data
        """
        harn.debug('_run_epoch {}, tag={}, learn={}'.format(harn.epoch, tag, learn))
        harn.debug(' * len(loader) = {}'.format(len(loader)))
        harn.debug(' * loader.batch_size = {}'.format(loader.batch_size))

        harn.current_tag = tag

        # use exponentially weighted or windowed moving averages across epochs
        iter_moving_metrics = harn._run_metrics[tag]
        # use simple moving average within an epoch
        epoch_moving_metrics = util.CumMovingAve(nan_method='ignore')

        # Flag if model is training (influences batch-norm / dropout)
        if harn.model.training != learn or learn:
            harn.model.train(learn)

        bsize = loader.batch_sampler.batch_size
        msg = harn._batch_msg({'loss': -1}, bsize)
        desc = tag + ' ' + msg
        position = (list(harn.loaders.keys()).index(tag) +
                    harn.main_prog.pos + 1)
        prog = harn._make_prog(desc=desc, total=len(loader), disable=not
                               harn.config['show_prog'], position=position,
                               leave=True, dynamic_ncols=True)
        harn._update_prog_postfix(prog)

        if isinstance(prog, ub.ProgIter):
            prog.begin()
        with util.grad_context(learn):
            harn.debug('Making batch iterator')
            batch_iter = iter(loader)
            harn.debug('Starting batch iteration for tag={}, epoch={}'.format(
                tag, harn.epoch))

            for bx in range(len(loader)):
                raw_batch = next(batch_iter)

                harn.bxs[tag] = bx
                # harn.debug('{} batch iteration {}'.format(tag, bx))

                batch = harn.prepare_batch(raw_batch)

                # core learning / backprop
                outputs, loss = harn._run_batch(bx, batch, learn=learn)

                # measure train accuracy and other informative metrics
                cur_metrics = harn._on_batch(bx, batch, outputs, loss)

                # accumulate measures
                epoch_moving_metrics.update(cur_metrics)
                iter_moving_metrics.update(cur_metrics)

                # display_train training info
                if harn.check_interval('display_' + tag, bx):
                    ave_metrics = iter_moving_metrics.average()

                    msg = harn._batch_msg({'loss': ave_metrics['loss']}, bsize)
                    prog.set_description(tag + ' ' + msg)

                    # log_iter_train, log_iter_test, log_iter_vali
                    if harn.check_interval('log_iter_' + tag, bx):
                        iter_idx = (harn.epoch * len(loader) + bx)
                        for key, value in ave_metrics.items():
                            harn.log_value(tag + ' iter ' + key, value, iter_idx)

                    prog.update(harn.intervals['display_' + tag])
                    harn._update_prog_postfix(prog)

        prog.close()

        # record a True average for the entire batch
        epoch_metrics = epoch_moving_metrics.average()

        # call hooks after every epoch
        custom_metrics = harn.on_epoch()
        _disjoint_dict_update(epoch_metrics, custom_metrics)

        for key, value in epoch_metrics.items():
            harn.log_value(tag + ' epoch ' + key, value, harn.epoch)
        harn.debug('Finished batch iteration for tag={}, epoch={}'.format(
            tag, harn.epoch))

        return epoch_metrics

    @profiler.profile
    def _run_batch(harn, bx, batch, learn=False):
        """
        batch with weight updates
        """
        try:
            outputs, loss = harn.run_batch(batch)
        except Exception:
            harn.error('May need to override `run_batch` with a custom func')
            raise

        # backprop and learn
        if learn:
            loss.backward()
            # approximates a batch size of (bsize * bstep) if step > 1,
            bstep = harn.dynamics['batch_step']
            if (bx + 1) % bstep == 0:
                # NOTE: the last few batches might be skipped if bstep > 1
                # harn.log("STEP")
                harn.optimizer.step()
                harn.optimizer.zero_grad()

        return outputs, loss

    def _on_batch(harn, bx, batch, outputs, loss):
        if torch.__version__.startswith('0.3'):
            loss_value = float(loss.data.cpu().sum())
        else:
            loss_value = float(loss.data.cpu().item())
        if not np.isfinite(loss_value):
            harn.warn('WARNING: got inf loss, setting loss to a large value')
            loss_value = harn.config['large_loss'] * 10

        if harn.current_tag == 'train':
            if loss_value > harn.config['large_loss']:
                # if the loss is getting large, check if the weights are ok
                harn._check_divergence()

        metrics_dict = {
            'loss': loss_value,
        }
        custom_metrics = harn.on_batch(batch, outputs, loss)
        _disjoint_dict_update(metrics_dict, custom_metrics)

        return metrics_dict

    def _check_divergence(harn):
        # Eventually we may need to remove
        # num_batches_tracked once 0.5.0 lands
        state = harn.model.module.state_dict()
        weights = sum(v.sum() for v in state.values())
        if not np.isfinite(weights):
            raise TrainingDiverged(
                'NON-FINITE WEIGHTS weights = {!r}'.format(weights))


@register_mixin
class CoreCallback:
    """
    We encourage you to overwrite these methods
    """

    def _tovar(harn, data):
        # handle cases when labels are unstructured
        if isinstance(data, list):
            # handle one level of nesting
            return [harn.xpu.variable(d) for d in data]
        else:
            return harn.xpu.variable(data)

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure

        Overload Encouraged, but not always necessary
        """
        batch_inputs, batch_labels = raw_batch

        # the dataset should return a inputs/target 2-tuple of lists.
        # in most cases each list will be length 1, unless there are
        # multiple input branches or multiple output branches.
        if not isinstance(batch_inputs, (list, tuple)):
            batch_inputs = [batch_inputs]
        if not isinstance(batch_labels, (list, tuple)):
            batch_labels = [batch_labels]

        inputs = [harn.xpu.variable(d) for d in batch_inputs]
        labels = [harn._tovar(d) for d in batch_labels]

        batch = (inputs, labels)
        return batch

    def run_batch(harn, batch):
        """
        Basic connection inputs -> model -> outputs -> criterion -> loss

        Overload Encouraged, but not always necessary

        Returns:
            tuple: (outputs, loss)
        """
        # Simple forward prop and loss computation
        inputs, labels = batch
        outputs = harn.model(*inputs)
        loss = harn.criterion(outputs, *labels)
        return outputs, loss

    def on_batch(harn, batch, outputs, loss):
        """
        custom callback typically used to compute batch evaluation measures
        or accumulate data.

        If a dict is returned its items are added to batch measures, and
        accumulated via moving averages into epoch measures.

        Overload Encouraged

        Returns:
            dict or None: dictionary of scalar batch measures
        """
        pass

    def on_epoch(harn):
        """
        custom callback typically used to compute epoch evaluation measures.

        If a dict is returned its items are added to epoch measures

        Overload Encouraged

        Returns:
            dict or None: dictionary of scalar epoch measures
        """
        pass

    def on_complete(harn):
        """
        custom callback typically used to evaluate or deploy the final model.

        Overload Encouraged
        """
        pass

    def get_snapshot_state(harn):
        """
        Returns a dictionary containing the base snapshot state.
        This can be overrided for specific applications.

        Returns:
            dict: snapshot_state
        """
        snapshot_state = {
            'epoch': harn.epoch,
            'model_state_dict': harn.model.state_dict(),
            'optimizer_state_dict': harn.optimizer.state_dict(),
            'monitor_state_dict': harn.monitor.state_dict(),
        }
        return snapshot_state

    def set_snapshot_state(harn, snapshot_state):
        """
        Sets harness state based on a previous snapshot.

        This can be overrided for specific applications.  In this case,
        it is the users responsibility to ensure that this handles all relevant
        items returned by `harn.get_snapshot_state`.

        Args:
            snapshot_state (dict): information corresponding to
        """
        if 'epoch' in snapshot_state:
            # the snapshot holds the previous epoch; add one to move to current
            harn.epoch = snapshot_state['epoch'] + 1

        if 'model_state_dict' in snapshot_state:
            harn.model.load_state_dict(snapshot_state['model_state_dict'])
            harn.debug('loaded model_state_dict')

        if 'monitor_state_dict' in snapshot_state:
            # hack: dont override patience, use whatever the current val is
            patience = harn.monitor.patience
            harn.monitor.load_state_dict(snapshot_state['monitor_state_dict'])
            harn.monitor.patience = patience
            harn.debug('loaded monitor_state_dict')

        if 'optimizer_state_dict' in snapshot_state:
            harn.optimizer.load_state_dict(snapshot_state['optimizer_state_dict'])
            harn.debug('loaded optimizer_state_dict')

        # Ensure scheduler is given current information
        if harn.scheduler:
            harn.scheduler.step(epoch=harn.epoch - 1)


# Define the exposed class as a union of mixin classes
class FitHarn(*MIXINS):
    """
    Basic harness for training a pytorch model.

    Note:
        The following methods can be overriden to customize the harness

            * prepare_batch(harn, raw_batch)

            * run_batch(harn, batch)

            * on_batch(harn, batch, outputs, loss)

            * on_epoch(harn)

    Args:
        hyper (netharn.HyperParams): Parameters that determine the system.
            This serializable class encodes enough information to
            deterministically reproduce an experiment.

            Because it is serializable it also has a dict representation.

        train_dpath (str or None): if specified, all progress information is
            stored in this path and the path computed via hyper is ignored.
            Note: it is recomended that this is left None, and you allow
            `hyper` to create a directory based on the hyperparamters.

    Note:
        hyper is optional. If you choose not to specify it then you must
        overwrite harn.setup_modules and create the requires class instances
        (i.e. model, optimizer, monitor, etc...). You also need to specify
        train_dpath yourself if you want to save progress snapshots.
    """
    def __init__(harn, hyper=None, train_dpath=None):
        if isinstance(hyper, dict):
            hyper = hyperparams.HyperParams(**hyper)
        harn.hyper = hyper

        harn.main_prog = None

        harn.datasets = None
        harn.loaders = None

        harn.model = None
        harn.optimizer = None
        harn.scheduler = None
        harn.monitor = None
        harn.criterion = None
        harn.dynamics = {'batch_step': 1}

        harn.paths = None
        harn.train_dpath = train_dpath
        harn.nice_dpath = None

        harn._initialized = False
        harn.flog = None
        harn.tlog = None

        harn.dry = None

        # Track current epoch number
        harn.epoch = 0

        # Track current iteration within an epoch
        harn.bxs = {
            'train': 0,
            'vali': 0,
            'test': 0,
        }

        harn.intervals = {
            'display_train': 1,
            'display_vali': 1,
            'display_test': 1,

            'log_iter_train': None,
            'log_iter_test': None,
            'log_iter_vali': None,

            'vali': 1,
            'test': 1,

            # how often to take a snapshot
            'snapshot': 1,

            # how often to remove old snapshots
            'cleanup': 10,
        }
        harn.config = {
            'show_prog': True,
            'use_tqdm': None,
            'prog_backend': 'tqdm',

            # A loss that would be considered large
            'large_loss': 100,

            # number of recent / best snapshots to keep
            'num_keep': 10,
            'keep_freq': 10,
        }
        harn.current_tag = None

    def check_interval(harn, tag, idx):
        """
        check if its time to do something that happens every few iterations
        """
        n = harn.intervals[tag]
        if n is None:
            return False
        return (idx + 1) % n == 0


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.fit_harn all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
