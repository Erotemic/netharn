# from os.path import join, normpath, dirname
from os.path import dirname
from os.path import join
import ubelt as ub
from clab import util


class Folders(object):
    def __init__(self, workdir='.', hyper=None, datasets=None, nice=None):
        self.datasets = datasets
        self.workdir = workdir
        self.hyper = hyper
        self.nice = nice

    def train_info(self, short=True, hashed=True):
        # TODO: needs MASSIVE cleanup and organization

        # TODO: if pretrained is another clab model, then we should read that
        # train_info if it exists and append it to a running list of train_info
        hyper = self.hyper

        arch = hyper.model_cls.__name__

        arch_base = join(self.workdir, 'arch', arch)

        if 'train' in hyper.input_ids:
            # NEW WAY
            input_id = hyper.input_ids['train']
        else:
            # OLD WAY
            input_id = self.datasets['train'].input_id
            if callable(input_id):
                input_id = input_id()

        train_hyper_id_long = hyper.hyper_id()
        train_hyper_id_brief = hyper.hyper_id(short=short, hashed=hashed)
        train_hyper_hashid = ub.hash_data(train_hyper_id_long)[:8]

        # TODO: hash this to some degree
        other_id = hyper.other_id()

        augment_json = hyper.augment_json()

        aug_brief = 'AU' + ub.hash_data(augment_json)[0:6]
        extra_hash = ub.hash_data([hyper.centering])[0:6]

        train_id = '{}_{}_{}_{}_{}'.format(
            ub.hash_data(input_id)[:6], train_hyper_id_brief,
            aug_brief, extra_hash, other_id)

        # Gather all information about this run into a single hash
        train_hashid = ub.hash_data(train_id)[0:8]

        full_dname = 'fit_{}'.format(train_id)

        link_dname = train_hashid

        input_dname = 'input_' + input_id

        train_dpath = join(arch_base, input_dname, full_dname)

        # setup a short symlink directory as well
        link_base = join(self.workdir, 'link')
        link_dpath = join(link_base, link_dname)

        # also setup a "nice" custom name, which may conflict, but oh well
        if self.nice:
            nice_base = join(self.workdir, 'nice')
            nice_dpath = join(nice_base, self.nice)
        else:
            nice_dpath = None

        # make temporary initializer so we can infer the history
        temp_initializer = hyper.make_initializer()
        init_history = temp_initializer.history()

        train_info =  ub.odict([
            ('train_hashid', train_hashid),

            ('train_id', train_id),

            ('workdir', self.workdir),

            ('aug_brief', aug_brief),

            ('input_id', input_id),

            ('other_id', other_id),

            ('train_dpath', train_dpath),
            ('hyper', hyper.get_initkw()),

            ('train_hyper_id_long', train_hyper_id_long),
            ('train_hyper_id_brief', train_hyper_id_brief),
            ('train_hyper_hashid', train_hyper_hashid),
            ('init_history', init_history),
            ('init_history_hashid', ub.hash_data(util.make_idstr(init_history))),

            ('nice', self.nice),

            ('link_dname', link_dname),
            ('link_dpath', link_dpath),
            ('nice_dpath', nice_dpath),

            # TODO, add in n_classes if applicable
            # TODO, add in centering if applicable
            ('centering', hyper.centering),

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
        ub.ensuredir(dirname(train_info['link_dpath']))
        ub.symlink(train_info['train_dpath'], train_info['link_dpath'],
                   overwrite=True, verbose=3)

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

