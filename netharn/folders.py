# from os.path import join, normpath, dirname
from os.path import dirname
from os.path import join
import ubelt as ub
from netharn import util


class Folders(object):
    def __init__(self, hyper):
        self.hyper = hyper

    def train_info(self, short=True, hashed=True):
        """
        CommandLine:
            python ~/code/netharn/netharn/folders.py Folders.train_info

        Example:
            >>> import netharn as nh
            >>> datasets = {
            >>>     'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
            >>>     'vali': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
            >>> }
            >>> hyper = nh.hyperparams.HyperParams(**{
            >>>     # --- Data First
            >>>     'datasets'    : datasets,
            >>>     'nice'        : 'demo',
            >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/demo'),
            >>>     'loaders'     : {'batch_size': 64},
            >>>     'xpu'         : nh.XPU.cast('auto'),
            >>>     # --- Algorithm Second
            >>>     'model'       : (nh.models.ToyNet2d, {}),
            >>>     'optimizer'   : (nh.optimizers.SGD, {
            >>>         'lr': 0.001
            >>>     }),
            >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
            >>>     #'criterion'   : (nh.criterions.FocalLoss, {}),
            >>>     'initializer' : (nh.initializers.KaimingNormal, {
            >>>         'param': 0,
            >>>     }),
            >>>     'scheduler'   : (nh.schedulers.ListedLR, {
            >>>         'step_points': {0: .001, 2: .01, 5: .015, 6: .005, 9: .001},
            >>>         'interpolate': True,
            >>>     }),
            >>>     'monitor'     : (nh.Monitor, {
            >>>         'max_epoch': 10
            >>>     }),
            >>> })
            >>> folders = Folders(hyper)
            >>> info = folders.train_info()
            >>> print(ub.repr2(info))
        """
        # TODO: needs MASSIVE cleanup and organization

        # TODO: if pretrained is another netharn model, then we should read that
        # train_info if it exists and append it to a running list of train_info
        hyper = self.hyper

        if hyper.model_cls is None:
            # import utool
            # utool.embed()
            raise ValueError('model_cls is None')
        # arch = hyper.model_cls.__name__

        train_dset = hyper.datasets['train']
        if hasattr(train_dset, 'input_id'):
            input_id = train_dset.input_id
            if callable(input_id):
                input_id = input_id()
        else:
            input_id = 'none'

        train_hyper_id_long = hyper.hyper_id()
        train_hyper_id_brief = hyper.hyper_id(short=short, hashed=hashed)
        train_hyper_hashid = ub.hash_data(train_hyper_id_long)[:8]

        # TODO: hash this to some degree
        other_id = hyper.other_id()

        augment_json = hyper.augment_json()

        aug_brief = 'AU' + ub.hash_data(augment_json)[0:6]
        # extra_hash = ub.hash_data([hyper.centering])[0:6]

        train_id = '{}_{}_{}_{}'.format(
            ub.hash_data(input_id)[:6], train_hyper_id_brief,
            aug_brief, other_id)

        # Gather all information about this run into a single hash
        train_hashid = ub.hash_data(train_id)[0:8]

        # input_dname = 'input_' + input_id
        # verbose_dpath = join(self.hyper.workdir, 'fit', 'link', 'arch', arch, input_dname, train_id)
        hashed_dpath = join(self.hyper.workdir, 'fit', 'runs', train_hashid)

        # setup a cannonical and a linked symlink dir
        train_dpath = hashed_dpath
        # link_dpath = verbose_dpath

        # also setup a "nice" custom name, which may conflict, but oh well
        if hyper.nice:
            nice_dpath = join(self.hyper.workdir, 'fit', 'nice', hyper.nice)
        else:
            nice_dpath = None

        # make temporary initializer so we can infer the history
        temp_initializer = hyper.make_initializer()
        init_history = temp_initializer.history()

        train_info =  ub.odict([
            ('train_hashid', train_hashid),

            ('train_id', train_id),

            ('workdir', self.hyper.workdir),

            ('aug_brief', aug_brief),

            ('input_id', input_id),

            ('other_id', other_id),

            ('hyper', hyper.get_initkw()),

            ('train_hyper_id_long', train_hyper_id_long),
            ('train_hyper_id_brief', train_hyper_id_brief),
            ('train_hyper_hashid', train_hyper_hashid),
            ('init_history', init_history),
            ('init_history_hashid', ub.hash_data(util.make_idstr(init_history))),

            ('nice', hyper.nice),

            ('train_dpath', train_dpath),
            # ('link_dpath', link_dpath),
            ('nice_dpath', nice_dpath),

            # TODO, add in n_classes if applicable
            # TODO, add in centering if applicable
            # ('centering', hyper.centering),

            # HACKED IN
            ('augment', hyper.augment_json()),
        ])
        return train_info

    def setup_dpath(self, short=True, hashed=True):
        train_info = self.train_info(short, hashed)

        train_dpath = ub.ensuredir(train_info['train_dpath'])
        train_info_fpath = join(train_dpath, 'train_info.json')

        util.write_json(train_info_fpath, train_info)

        # setup symlinks
        # ub.ensuredir(dirname(train_info['link_dpath']))
        # ub.symlink(train_info['train_dpath'], train_info['link_dpath'],
        #            overwrite=True, verbose=3)

        if train_info['nice_dpath']:
            ub.ensuredir(dirname(train_info['nice_dpath']))
            ub.symlink(train_info['train_dpath'], train_info['nice_dpath'],
                       overwrite=True, verbose=3)

        print('+=========')
        # print('hyper_strid = {!r}'.format(params.hyper_id()))
        # print('train_init_id = {!r}'.format(train_info['input_id']))
        # print('arch = {!r}'.format(train_info['arch_id']))
        # print('train_hyper_hashid = {!r}'.format(train_info['train_hyper_hashid']))
        print('hyper = {}'.format(ub.repr2(train_info['hyper'], nl=3)))
        print('train_hyper_id_brief = {!r}'.format(train_info['train_hyper_id_brief']))
        print('train_id = {!r}'.format(train_info['train_id']))
        print('+=========')
        return train_info

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.folders all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
