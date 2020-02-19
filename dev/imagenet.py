"""
This example attempts to reproduce a large scale training example on ImageNet

We use "ImageNet LSVRC 2013 Validation Set (Object Detection)" and "ImageNet
LSVRC 2014 Training Set (Object Detection)" from
http://academictorrents.com/collection/imagenet-lsvrc-2015

To set these up I downloaded the torrents linked in the above url.  It looks
like it may take about 2 days to complete the training set download.

Here are estimated stats:

Training:
    Name: ImageNet LSVRC 2014 Training Set (Object Detection)
    Size: 50.12GB
    DL-ETA: 3 days

Validation:
    Name: ImageNet LSVRC 2013 Validation Set (Object Detection)
    Size: 2.71GB
    DL-ETA: 3 hours


Notes the LSVRC 2017:
    url: http://academictorrents.com/details/943977d8c96892d24237638335e481f3ccd54cfb
    fname: ILSVRC2017_CLS-LOC.tar.gz
    size: 166.02 GB
    DL-ETA: 14 days


Tiny ImageNet:
    pass


Work in progress


"""
from os.path import exists
from os.path import join
import netharn as nh
import torch  # NOQA

import scriptconfig as scfg
import ubelt as ub

# import numpy as np
# import six
# import kwarray
# import kwimage

# import imgaug.augmenters as iaa
# import imgaug

# from os.path import join
# from torch.nn import functional as F
from torch.utils import data as torch_data


def grab_tiny_imagenet_as_coco():
    import ubelt as ub

    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    dpath = ub.ensure_app_cache_dir('netharn', 'tiny-imagenet-200')
    dset_root = join(dpath, 'tiny-imagenet-200')

    zip_fpath = ub.grabdata(url, dpath=dpath)

    if not exists(dset_root):
        import zipfile
        zip_ref = zipfile.ZipFile(zip_fpath, 'r')
        zip_ref.extractall(dpath)
        zip_ref.close()

    tiny_imgnet_info = {
        'train': join(dset_root, 'train'),
        'test': join(dset_root, 'test'),
        'vali': join(dset_root, 'val'),

        'wnids': join(dset_root, 'wnids.txt'),
        'words': join(dset_root, 'words.txt'),
    }

    import glob
    train_annots = list(glob.glob(join(tiny_imgnet_info['train'], '*/*boxes.txt')))
    vali_annots = list(glob.glob(join(tiny_imgnet_info['vali'], 'val_annotations.txt')))

    import ndsampler

    img_root = {
        'train': join(tiny_imgnet_info['train']),
        'vali': join(tiny_imgnet_info['vali'], 'images'),
        'test': join(tiny_imgnet_info['test'], 'images'),
    }
    gpaths = {
        'train': list(glob.glob(join(tiny_imgnet_info['train'], '*/images/*.JPEG'))),
        'vali': list(glob.glob(join(tiny_imgnet_info['vali'], 'images/*.JPEG'))),
        'test': list(glob.glob(join(tiny_imgnet_info['test'], 'images/*.JPEG')))
    }
    annots_text = {
        'train': ''.join(ub.readfrom(fpath) for fpath in train_annots),
        'vali': ''.join(ub.readfrom(fpath) for fpath in vali_annots),
    }
    coco_datasets = {
        'train': ndsampler.CocoDataset(tag='tiny-imagenet-train'),
        'vali': ndsampler.CocoDataset(tag='tiny-imagenet-vali'),
    }

    for catname in (_ for _ in ub.readfrom(tiny_imgnet_info['wnids']).split('\n') if _):
        for dset in coco_datasets.values():
            dset.add_category(name=catname)

    for tag in ['train', 'vali']:
        gpaths_ = gpaths[tag]
        annots_ = annots_text[tag]
        dset = coco_datasets[tag]

        dset.img_root = img_root[tag]

        for gpath in gpaths_:
            dset.add_image(file_name=gpath)

        for line in (_ for _ in annots_.split('\n') if _):
            parts = line.split('\t')
            if tag == 'train':
                gname = parts[0]
                catname = gname.split('_')[0]
                x, y, w, h = list(map(float, parts[1:]))
                gpath = join(img_root[tag], catname, 'images', gname)
            else:
                gname, catname = parts[0:2]
                x, y, w, h = list(map(float, parts[2:]))
                gpath = join(img_root[tag], gname)

            bbox = (x, y, w + 1, h + 1)
            cat = dset.name_to_cat[catname]
            img = dset.index.file_name_to_img[gpath]

            dset.add_annotation(image_id=img['id'], bbox=bbox,
                                category_id=cat['id'])

        dset._ensure_imgsize()
        dset._build_hashid()
        print('dset.hashid = {!r}'.format(dset.hashid))

    return coco_datasets


class ImageClfHarn(nh.FitHarn):
    pass


class ImageClfConfig(scfg.Config):
    """
    Default configuration for setting up a training session
    """
    default = {
        'nice': scfg.Path('untitled', help='A human readable tag that is "nice" for humans'),
        'workdir': scfg.Path('~/work/tiny-imagenet', help='Dump all results in your workdir'),

        'workers': scfg.Value(0, help='number of parallel dataloading jobs'),
        'xpu': scfg.Value('argv', help='See netharn.XPU for details. can be cpu/gpu/cuda0/0,1,2,3)'),

        'augmenter': scfg.Value(True, help='type of training dataset augmentation'),

        # 'datasets': scfg.Value('special:tiny_imgnet', help='Eventually you may be able to sepcify a coco file'),
        'train_dataset': scfg.Value(None),
        'vali_dataset': scfg.Value(None),
        'test_dataset': scfg.Value(None),

        'arch': scfg.Value('resnet50', help='Network architecture code'),
        'optim': scfg.Value('adam', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'input_dims': scfg.Value((224, 224), help='Window size to input to the network'),

        'batch_size': scfg.Value(6, help='number of items per batch'),

        'max_epoch': scfg.Value(100, help='Maximum number of epochs'),
        'patience': scfg.Value(100, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-3, help='Base learning rate'),
        'decay':  scfg.Value(1e-5, help='Base weight decay'),

        'schedule': scfg.Value('onecycle71', help=('Special coercable netharn code. Eg: onecycle50, step50, gamma')),

        'init': scfg.Value('kaiming_normal', help='How to initialized weights. (can be a path to a pretrained model)'),
        'pretrained': scfg.Path(help=('alternative way to specify a path to a pretrained model')),
    }

    def normalize(self):
        if self['pretrained'] in ['null', 'None']:
            self['pretrained'] = None

        if self['pretrained'] is not None:
            self['init'] = 'pretrained'


class ImagClfDataset(torch_data.Dataset):
    """

    self = torch_datasets['train']

    """
    def __init__(self, sampler, input_dims=(224, 224), augmenter=False):
        self.input_dims = None
        self.input_id = None
        self.cid_to_cidx = None
        self.classes = None
        self.sampler = None

        self.sampler = sampler

        self.input_id = self.sampler.dset.hashid

        self.cid_to_cidx = sampler.catgraph.id_to_idx
        self.classes = sampler.catgraph

        # self.augmenter = self._rectify_augmenter(augmenter)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.load_positive(index)
        im = sample['im']
        cid = sample['tr']['category_id']

        cidx = self.classes.id_to_idx()[cid]
        im_chw = torch.FloatTensor(im.transpose(2, 0, 1))

        item = {
            'im': im_chw,
            'cidx': cidx,
        }
        return item


def setup_harn(cmdline=True, **kwargs):
    """
    cmdline, kwargs = False, {}
    """
    import sys
    import ndsampler

    config = ImageClfConfig(default=kwargs)
    config.load(cmdline=cmdline)
    nh.configure_hacks(config)  # fix opencv bugs

    cacher = ub.Cacher('tiny-imagenet', cfgstr='v4', verbose=3)
    data = cacher.tryload()
    if data is None:
        data = grab_tiny_imagenet_as_coco()
        cacher.save(data)
    coco_datasets = data  # setup_coco_datasets()
    dset = coco_datasets['train']
    print('train dset = {!r}'.format(dset))

    workdir = ub.ensuredir(ub.expandpath(config['workdir']))
    samplers = {
        # tag: ndsampler.CocoSampler(dset, workdir=workdir, backend='cog')
        tag: ndsampler.CocoSampler(dset, workdir=workdir, backend='npy')
        for tag, dset in coco_datasets.items()
    }
    torch_datasets = {
        tag: ImagClfDataset(
            sampler, config['input_dims'],
            augmenter=((tag == 'train') and config['augmenter']),
        )
        for tag, sampler in samplers.items()
    }
    torch_loaders = {
        tag: torch_data.DataLoader(dset,
                                   batch_size=config['batch_size'],
                                   num_workers=config['workers'],
                                   shuffle=(tag == 'train'),
                                   pin_memory=True)
        for tag, dset in torch_datasets.items()
    }

    import torchvision
    # TODO: netharn should allow for this
    model_ = torchvision.models.resnet50(pretrained=False)

    # model_ = (, {
    #     'classes': torch_datasets['train'].classes,
    #     'in_channels': 3,
    # })
    initializer_ = nh.Initializer.coerce(config)

    hyper = nh.HyperParams(
        nice=config['nice'],
        workdir=config['workdir'],
        xpu=nh.XPU.coerce(config['xpu']),

        datasets=torch_datasets,
        loaders=torch_loaders,

        model=model_,
        initializer=initializer_,

        scheduler=nh.Scheduler.coerce(config),
        optimizer=nh.Optimizer.coerce(config),
        dynamics=nh.Dynamics.coerce(config),

        criterion=(nh.criterions.FocalLoss, {
            'focus': 0.0,
        }),

        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': config['patience'],
            'max_epoch': config['max_epoch'],
            'smoothing': .6,
        }),

        other={
            'batch_size': config['batch_size'],
        },
        extra={
            'argv': sys.argv,
            'config': ub.repr2(config.asdict()),
        }
    )

    # Create harness
    harn = ImageClfHarn(hyper=hyper)
    harn.classes = torch_datasets['train'].classes
    harn.preferences.update({
        'num_keep': 5,
        'keyboard_debug': True,
        # 'export_modules': ['netharn'],
    })
    harn.intervals.update({
        'vali': 1,
        'test': 10,
    })
    harn.script_config = config
    return harn
