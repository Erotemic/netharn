"""
The :mod:`netharn.examples.cifar` is a clear dataset-specific example of what
netharn is and it's trying to do / not do. For a dataset-agnostic example see
:mod:`netharn.examples.segmentation`.

The basic idea is make an object that inherits from nh.FitHarn. This is our
harness object. It will contain the hyperparameters as well as the learning
state. All the training loop boilerplate has already been written in the parent
class, so all our child class needs to do is: define `prepare_batch` (not
usually needed) and `run_batch`. Code to measure and record performance should
be placed in `on_batch` and `on_epoch`.

The `train` function is our main entry point. It reads parameters from the
command line to override defaults. It then consructs the `HyperParams` object
and constructs an instance of `CIFAR_FitHarn` and calls `harn.run()`.

This begins the training process. At a high level the harness will load the
data using torch DataLoaders, and call `run_batch` when it needs to compute the
model outputs and loss based on the input data. The returned loss is used to
update the model weights if `harn.tag === 'train'`, for validation, test, and
calibration (todo) datasets the loss is simply recorded.

After `run_batch` finishes the `on_batch` function is called, where you can
optionally return a dict of scalars to log as measurements for this batch (note
loss is always recorded, so we need not return it here, but loss components may
be useful). A similar thing happens in `on_epoch`, where you should return
metrics about the entire dataset.

The training harness manages the fit directory structure based on a hash of the
hyperparameters, the creation of algorithm component instance (e.g. model,
optimizer), initializing model weights, restarting from the most recent epoch,
updating the learning rates, various training loop boilerplate details,
checking divergence, reporting progress, handling differences between train,
validation, and test sets. In short, netharn handles the necessary parts and
let the developer focus on the important parts.


CommandLine:
    python -m netharn.examples.cifar.py --gpu=0 --arch=resnet50
    python -m netharn.examples.cifar.py --gpu=0 --arch=wrn_22 --lr=0.003 --schedule=onecycle --optim=adamw
    python -m netharn.examples.cifar.py --gpu=1,2,3 --arch=wrn_22 --lr=0.003 --schedule=onecycle --optim=adamw --batch_size=1800
    python -m netharn.examples.cifar.py --gpu=1,2 --arch=resnet50 --lr=0.003 --schedule=onecycle --optim=adamw

"""
import sys
from os.path import join
import numpy as np
import ubelt as ub
import torch
import os
import pickle
import netharn as nh
import scriptconfig as scfg


class CIFARConfig(scfg.Config):
    """
    This is the default configuration for running the CIFAR training script.

    Defaults can be modified via passing command line arguments, specifing a
    json configuration file, or using python kwargs.

    Note that scriptconfig is not necessary to use netharn, but it does make it
    easy to specify a default configuration and modify.
    """
    default = {
        'nice': scfg.Value('untitled', help='A human readable tag that is "nice" for humans'),
        'workdir': scfg.Path('~/work/cifar', help='Dump all results in your workdir'),

        'workers': scfg.Value(2, help='number of parallel dataloading jobs'),
        'xpu': scfg.Value('argv', help='See netharn.XPU for details. can be cpu/gpu/cuda0/0,1,2,3)'),

        'dataset': scfg.Value('cifar10', choices=['cifar10', 'cifar100'],
                              help='which cifar network to use'),
        'num_vali': scfg.Value(0, help='number of validation examples'),

        'arch': scfg.Value('resnet50', help='Network architecture code'),
        'optim': scfg.Value('sgd', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'input_dims': scfg.Value((32, 32), help='Image size passed to the network'),

        'batch_size': scfg.Value(64, help='number of items per batch'),

        'max_epoch': scfg.Value(350, help='Maximum number of epochs'),
        'patience': scfg.Value(350, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-1, help='Base learning rate'),
        'decay':  scfg.Value(5e-4, help='Base weight decay'),

        'schedule': scfg.Value('simplestep', help=('Special coercable netharn code. Eg: onecycle50, step50, gamma')),

        'init': scfg.Value('noop', help='How to initialized weights. (can be a path to a pretrained model)'),
        'pretrained': scfg.Path(help=('alternative way to specify a path to a pretrained model')),

        'deterministic': scfg.Value(False, help='run deterministically'),
        'seed': scfg.Value(137852547, help='seed for determinism'),
    }

    def normalize(self):
        if self['pretrained'] in ['null', 'None']:
            self['pretrained'] = None

        if self['pretrained'] is not None:
            self['init'] = 'pretrained'


class CIFAR_FitHarn(nh.FitHarn):
    """
    The `FitHarn` class contains a lot of reusable boilerplate. We inherit
    from it and override relevant methods to customize the training procedure
    to our particular problem and dataset.

    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> from cifar import *
        >>> harn = setup_harn().initialize()
        >>> batch = harn._demo_batch(0, tag='vali')
        >>> test_metrics = harn._demo_epoch('vali')
    """

    def after_initialize(harn):
        """
        Custom function that performs final initialization steps
        """
        # Our cifar harness will record confusion vectors after every batch.
        # Then, after every epoch we will transform these into quality measures
        # like Accuracy, MCC, AUC, AP, etc...
        harn._accum_confusion_vectors = {
            'y_true': [],
            'y_pred': [],
            'probs': [],
        }

    def run_batch(harn, batch):
        """
        Custom function to compute the output of a batch and its loss.

        The primary purpose of `FitHarn` is to abstract away the boilerplate code
        necessary to run these lines of code.

        After the `FitHarn` prepares the batch it tries to run it.
        In most cases it is best to explicitly override this function to
        describe (in code) exactly how data should be sent to your model and
        exactly how the loss should be computed.

        Returns:
            Tuple[object, Tensor]: This function MUST return:
                (1) the arbitrary output of the model, and
                (2) the loss as a Tensor scalar. The harness will take care
                    of calling `.backwards` on the loss and updating the
                    gradient via the optimizer.
        """
        inputs, labels = ub.take(batch, ['input', 'label'])
        inputs.shape
        outputs = harn.model(inputs)
        loss = harn.criterion(outputs, labels)
        return outputs, loss

    # def backpropogate(harn, bx, batch, loss):
    #     """
    #     Note: this function usually does not need to be overloaded,
    #     but you can if you want to. The actual base implementation is
    #     slightly more nuanced. For details see:
    #     :func:netharn.fit_harn.CoreCallbacks.backpropogate
    #     """
    #     loss.backward()
    #     harn.optimizer.step()
    #     harn.optimizer.zero_grad()

    def on_batch(harn, batch, outputs, loss):
        """
        Custom code executed at the end of each batch.

        This function can optionally return a dictionary containing any scalar
        quality metrics that you wish to log and monitor. (Note these will be
        plotted to tensorboard if that is installed).

        Notes:
            It is best to keep this function small as it is run very often
        """
        y_pred = outputs.data.max(dim=1)[1].cpu().numpy()
        y_true = batch['label'].data.cpu().numpy()

        bx = harn.bxs[harn.current_tag]
        if bx < 3:
            stacked = harn._draw_batch(batch, outputs)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'batch_{}_epoch_{}.jpg'.format(bx, harn.epoch))
            import kwimage
            kwimage.imwrite(fpath, stacked)

        probs = outputs.softmax(dim=1).data.cpu().numpy()
        harn._accum_confusion_vectors['y_true'].append(y_true)
        harn._accum_confusion_vectors['y_pred'].append(y_pred)
        harn._accum_confusion_vectors['probs'].append(probs)

    def on_epoch(harn):
        """
        Custom code executed at the end of each epoch.

        This function can optionally return a dictionary containing any scalar
        quality metrics that you wish to log and monitor. (Note these will be
        plotted to tensorboard if that is installed).

        Notes:
            It is ok to do some medium lifting in this function because it is
            run relatively few times.

        Example:
            >>> # xdoctest: +REQUIRES(--download)
            >>> harn = setup_harn().initialize()
            >>> harn._demo_epoch('vali')
        """
        from netharn.metrics import clf_report
        dset = harn.datasets[harn.current_tag]

        probs = np.vstack(harn._accum_confusion_vectors['probs'])
        y_true = np.hstack(harn._accum_confusion_vectors['y_true'])
        y_pred = np.hstack(harn._accum_confusion_vectors['y_pred'])

        # _pred = probs.argmax(axis=1)
        # assert np.all(_pred == y_pred)

        # from netharn.metrics import confusion_vectors
        # cfsn_vecs = confusion_vectors.ConfusionVectors.from_arrays(
        #     true=y_true, pred=y_pred, probs=probs, classes=dset.classes)
        # report = cfsn_vecs.classification_report()
        # combined_report = report['metrics'].loc['combined'].to_dict()

        # ovr_cfsn = cfsn_vecs.binarize_ovr()
        # Compute multiclass metrics (new way!)
        target_names = dset.classes
        ovr_report = clf_report.ovr_classification_report(
            y_true, probs, target_names=target_names, metrics=[
                'auc', 'ap', 'mcc', 'brier'
            ])

        # percent error really isn't a great metric, but its standard.
        errors = (y_true != y_pred)
        acc = 1.0 - errors.mean()
        percent_error = (1.0 - acc) * 100

        metrics_dict = ub.odict()
        metrics_dict['ave_brier'] = ovr_report['ave']['brier']
        metrics_dict['ave_mcc'] = ovr_report['ave']['mcc']
        metrics_dict['ave_auc'] = ovr_report['ave']['auc']
        metrics_dict['ave_ap'] = ovr_report['ave']['ap']
        metrics_dict['percent_error'] = percent_error
        metrics_dict['acc'] = acc

        harn.info('ACC FOR {!r}: {!r}'.format(harn.current_tag, acc))

        # Clear confusion vectors accumulator for the next epoch
        harn._accum_confusion_vectors = {
            'y_true': [],
            'y_pred': [],
            'probs': [],
        }
        return metrics_dict

    def _draw_batch(harn, batch, outputs, limit=32):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--download)
            >>> from netharn.examples.cifar import *  # NOQA
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='test')
            >>> outputs, loss = harn.run_batch(batch)
            >>> stacked = harn._draw_batch(batch, outputs, limit=12)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked, colorspace='rgb', doclf=True)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        inputs = batch['input'][0:limit].data.cpu().numpy()
        true_cxs = batch['label'].data.cpu().numpy()
        class_probs = outputs.softmax(dim=1).data.cpu().numpy()
        pred_cxs = class_probs.argmax(axis=1)

        dset = harn.datasets[harn.current_tag]
        classes = dset.classes

        todraw = []
        for im, pcx, tcx, probs in zip(inputs, pred_cxs, true_cxs, class_probs):
            im_ = im.transpose(1, 2, 0)

            # Renormalize and resize image for drawing
            min_, max_ = im_.min(), im_.max()
            im_ = ((im_ - min_) / (max_ - min_) * 255).astype(np.uint8)
            im_ = np.ascontiguousarray(im_)
            im_ = kwimage.imresize(im_, dsize=(200, 200))

            # Draw classification information on the image
            im_ = kwimage.draw_clf_on_image(im_, classes=classes, tcx=tcx,
                                            pcx=pcx, probs=probs)
            todraw.append(im_)

        stacked = kwimage.stack_images_grid(todraw, overlap=-10,
                                            bg_value=(10, 40, 30),
                                            chunksize=8)
        return stacked


def setup_harn():
    """
    Replicates parameters from https://github.com/kuangliu/pytorch-cifar

    The following is a table of kuangliu's reported accuracy and our measured
    accuracy for each architecture.

    The first column is kuangliu's reported accuracy, the second column is me
    running kuangliu's code, and the final column is using my own training
    harness (handles logging and whatnot) called netharn.

           arch |  kuangliu  | rerun-kuangliu  |  netharn |
    -------------------------------------------------------
    ResNet50    |    93.62%  |         95.370% |  95.72%  |
    DenseNet121 |    95.04%  |         95.420% |  94.47%  |
    DPN92       |    95.16%  |         95.410% |  94.92%  |

    CommandLine:
        python -m netharn.examples.cifar --gpu=0 --nice=resnet --arch=resnet50 --optim=sgd --schedule=simplestep --lr=0.1
        python -m netharn.examples.cifar --gpu=0 --nice=wrn --arch=wrn_22 --optim=sgd --schedule=simplestep --lr=0.1
        python -m netharn.examples.cifar --gpu=0 --nice=densenet --arch=densenet121 --optim=sgd --schedule=simplestep --lr=0.1
        python -m netharn.examples.cifar --gpu=0 --nice=efficientnet_scratch --arch=efficientnet-b0 --optim=sgd --schedule=simplestep --lr=0.01 --init=noop --decay=1e-5

        python -m netharn.examples.cifar --gpu=0 --nice=efficientnet \
            --arch=efficientnet-b0 --optim=rmsprop --lr=0.064 \
            --batch_size=512 --max_epoch=120 --schedule=Exponential-g0.97-s2

        python -m netharn.examples.cifar --gpu=0 --nice=efficientnet-scratch3 \
            --arch=efficientnet-b0 --optim=adamw --lr=0.016 --init=noop \
            --batch_size=1024 --max_epoch=450 --schedule=Exponential-g0.96-s3 --decay=1e-5

        python -m netharn.examples.cifar --gpu=0 --nice=efficientnet-pretrained2 \
            --arch=efficientnet-b0 --optim=adamw --lr=0.0064 --init=cls \
            --batch_size=512 --max_epoch=350 --schedule=Exponential-g0.97-s2 --decay=1e-5
    """
    import random
    import torchvision
    from torchvision import transforms

    # Create an instance of the CIFAR config and allow command line args
    config = CIFARConfig(cmdline=True)

    xpu = nh.XPU.coerce(config['xpu'])

    # The work directory is where all intermediate results are dumped.
    ub.ensuredir(config['workdir'])

    if config['deterministic']:
        # Take care of random seeding and ensuring appropriate determinisim
        torch.manual_seed((config['seed'] + 0) % int(2 ** 32 - 1))
        random.seed((config['seed'] + 2360097502) % int(2 ** 32 - 1))
        np.random.seed((config['seed'] + 893874269) % int(2 ** 32 - 1))

    if torch.backends.cudnn.enabled:
        # TODO: ensure the CPU mode is also deterministic
        torch.backends.cudnn.deterministic = config['deterministic']

    # Define preprocessing + augmentation strategy
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(config['input_dims']),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.5, 0.5),  # Cutout
                                 value=0),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(config['input_dims']),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if config['dataset'] == 'cifar10':
        DATASET = torchvision.datasets.CIFAR10
        dset = DATASET(root=config['workdir'], download=True)
        meta_fpath = os.path.join(dset.root, dset.base_folder, 'batches.meta')
        meta_dict = pickle.load(open(meta_fpath, 'rb'))
        classes = meta_dict['label_names']
        # For some reason the torchvision objects dont have the label names
        # in the dataset. But the download directory will have them.
        # classes = [
        #     'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        #     'horse', 'ship', 'truck',
        # ]
    elif config['dataset'] == 'cifar100':
        DATASET = torchvision.datasets.CIFAR100
        dset = DATASET(root=config['workdir'], download=True)
        meta_fpath = os.path.join(dset.root, dset.base_folder, 'meta')
        meta_dict = pickle.load(open(meta_fpath, 'rb'))
        classes = meta_dict['fine_label_names']
        # classes = [
        #     'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        #     'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        #     'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        #     'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
        #     'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
        #     'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
        #     'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        #     'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        #     'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
        #     'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
        #     'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
        #     'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
        #     'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
        #     'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
        #     'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        #     'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        #     'worm']
    else:
        raise KeyError(config['dataset'])

    datasets = {
        'train': DATASET(root=config['workdir'], train=True,
                         transform=transform_train),
        'test': DATASET(root=config['workdir'], train=False,
                        transform=transform_test),
    }

    if config['num_vali']:
        # Create a test train split
        learn = datasets['train']
        learn_copy = DATASET(root=config['workdir'], train=True,
                             transform=transform_test)

        import kwarray
        indices = np.arange(len(learn))
        indices = kwarray.shuffle(indices, rng=0)
        num_vali = config['num_vali']

        datasets['vali'] = torch.utils.data.Subset(learn_copy, indices[:num_vali])
        datasets['train'] = torch.utils.data.Subset(learn, indices[num_vali:])

    # For some reason the torchvision objects do not make the category names
    # easily available. We set them here for ease of use.
    reduction = int(ub.argval('--reduction', default=1))
    for key, dset in datasets.items():
        dset.classes = classes
        if reduction > 1:
            indices = np.arange(len(dset))[::reduction]
            dset = torch.utils.data.Subset(dset, indices)
        dset.classes = classes
        datasets[key] = dset

    loaders = {
        key: torch.utils.data.DataLoader(dset, shuffle=key == 'train',
                                         num_workers=config['workers'],
                                         batch_size=config['batch_size'],
                                         pin_memory=True)
        for key, dset in datasets.items()
    }

    if config['workers'] > 0:
        # Solves pytorch deadlock issue #1355.
        import cv2
        cv2.setNumThreads(0)

    # Choose which network architecture to train
    available_architectures = {
        'densenet121': (nh.models.densenet.DenseNet, {
            'nblocks': [6, 12, 24, 16],
            'growth_rate': 12,
            'reduction': 0.5,
            'num_classes': len(classes),
        }),

        'resnet50': (nh.models.resnet.ResNet, {
            'num_blocks': [3, 4, 6, 3],
            'num_classes': len(classes),
            'block': 'Bottleneck',
        }),

        'dpn26': (nh.models.dual_path_net.DPN, dict(cfg={
            'in_planes': (96, 192, 384, 768),
            'out_planes': (256, 512, 1024, 2048),
            'num_blocks': (2, 2, 2, 2),
            'dense_depth': (16, 32, 24, 128),
            'num_classes': len(classes),
        })),

        'dpn92': (nh.models.dual_path_net.DPN, dict(cfg={
            'in_planes': (96, 192, 384, 768),
            'out_planes': (256, 512, 1024, 2048),
            'num_blocks': (3, 4, 20, 3),
            'dense_depth': (16, 32, 24, 128),
            'num_classes': len(classes),
        })),
    }

    if config['arch'].startswith('wrn'):
        import fastai
        import fastai.vision
        available_architectures['wrn_22'] = (
            fastai.vision.models.WideResNet, dict(
                num_groups=3, N=3, num_classes=len(classes), k=6, drop_p=0.
            )
        )

    if config['arch'].startswith('efficientnet'):
        # Directly create the model instance...
        # (as long as it has an `_initkw` attribute)
        from netharn.models import efficientnet

        if config['init'] == 'cls':
            model_ = efficientnet.EfficientNet.from_pretrained(
                config['arch'], override_params={
                    'classes': classes,
                }
            )
            print('pretrained cls init')
        else:
            model_ = efficientnet.EfficientNet.from_name(
                config['arch'], override_params={
                    'classes': classes,
                }
            )
    else:
        model_ = available_architectures[config['arch']]

    if config['init'] in ['cls', 'noop']:
        initializer_ = (nh.initializers.NoOp, {})
    if config['init'] == 'kaiming_normal':
        initializer_ = (nh.initializers.KaimingNormal, {'param': 0, 'mode': 'fan_in'})
    else:
        # Note there are lots of different initializers including a special
        # pretrained initializer.
        initializer_ = nh.api.Initializer.coerce(config)

    if config['schedule'] == 'simplestep':
        scheduler_ = (nh.schedulers.ListedLR, {
            'points': {
                0: config['lr'],
                150: config['lr'] * 0.1,
                250: config['lr'] * 0.01,
            },
            'interpolate': False
        })
    elif config['schedule'] == 'onecycle':
        # TODO: Fast AI params
        # TODO: https://github.com/fastai/fastai/blob/c7df6a5948bdaa474f095bf8a36d75dbc1ee8e6a/fastai/callbacks/one_cycle.py
        # config['lr'] = 3e-3
        # cyc_len=35
        # max_lr = 3e-3
        # moms = (0.95,0.85)
        # div_factor = 25
        # pct_start=0.3,
        # wd=0.4
        # pct = np.linspace(0, 1.0, 35)
        # cos_up = (np.cos(np.pi * (1 - pct)) + 1) / 2
        # cos_down = cos_up[::-1]

        # pt1 = config['lr'] / 25.0
        # pt2 = config['lr']
        # pt3 = config['lr'] / (1000 * 25.0)

        # phase1 = (pt2 - pt1) * cos_up + pt1
        # phase2 = (pt2 - pt3) * cos_down + pt3
        # points = dict(enumerate(ub.flatten([phase1, phase2])))

        # scheduler_ = (nh.schedulers.ListedLR, {
        #     'points': points,
        #     'interpolate': False
        # })
        scheduler_ = (nh.schedulers.ListedScheduler, {
            'points': {
                'lr': {
                    0   : config['lr'] * 1.00,
                    35  : config['lr'] * 25,
                    70  : config['lr'] / 1000,
                },
                'momentum': {
                    0   : 0.95,
                    35  : 0.85,
                    70  : 0.95,
                },
                # 'weight_decay': {
                #     0: 3e-6,
                # }
            }
        })
    else:
        # The netharn API can construct a scheduler from standard keys in a
        # configuration dictionary. There is a bit of magic involved. Read docs
        # for coerce for more details.
        scheduler_ = nh.api.Scheduler.coerce(config)

    if config['optim'] == 'sgd':
        optimizer_ = (torch.optim.SGD, {
            'lr': config['lr'],
            'weight_decay': config['decay'],
            'momentum': 0.9,
            'nesterov': True,
        })
    elif config['optim'] == 'adamw':
        optimizer_ = (nh.optimizers.AdamW, {
            'lr': config['lr'],
            'betas': (0.9, 0.999),
            'weight_decay': config['decay'],
            'amsgrad': False,
        })
    else:
        # The netharn API can construct an optimizer from standard keys in a
        # configuration dictionary. There is a bit of magic involved. Read docs
        # for coerce for more details.
        optimizer_ = nh.api.Optimizer.coerce(config)

    # Notice that arguments to hyperparameters are typically specified as a
    # tuple of (type, Dict), where the dictionary are the keyword arguments
    # that can be used to instantiate an instance of that class. While
    # this may be slightly awkward, it enables netharn to track hyperparameters
    # more effectively. Note that it is possible to simply pass an already
    # constructed instance of a class, but this causes information loss.
    hyper = nh.HyperParams(
        # Datasets must be preconstructed
        datasets=datasets,
        nice=config['nice'],
        # Loader may be preconstructed
        loaders=loaders,
        workdir=config['workdir'],
        xpu=xpu,
        # The 6 major hyper components are best specified as a Tuple[type,
        # dict] However, in recent releases of netharn, these may be
        # initialized manually in certain conditions. See docs for details.
        # TODO: write docs about this.
        model=model_,
        optimizer=optimizer_,
        scheduler=scheduler_,
        dynamics=nh.api.Dynamics.coerce(config),
        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': config['patience'],
            'max_epoch': config['max_epoch'],
        }),
        initializer=initializer_,
        criterion=(torch.nn.CrossEntropyLoss, {}),
        # The rests of the keyword arguments are simply dictionaries used to
        # track other information.
        # Specify what augmentations you are performing for experiment tracking
        # augment=transform_train,
        other={
            # Specify anything else that is special about your hyperparams here
            # Especially if you make a custom_batch_runner
        },
        # These extra arguments are recorded in the train_info.json but do
        # not contribute to the hyperparameter hash.
        extra={
            'config': ub.repr2(dict(config)),
            'argv': sys.argv,
        }
    )

    # Creating an instance of a Fitharn object is typically fast.
    harn = CIFAR_FitHarn(hyper=hyper)
    harn.preferences['prog_backend'] = 'progiter'
    harn.preferences['keyboard_debug'] = True
    harn.preferences['eager_dump_tensorboard'] = True
    harn.preferences['tensorboard_groups'] = ['loss']

    harn.intervals.update({
        'vali': 1,
        'test': 1,
    })

    harn.script_config = config
    return harn


def main():
    harn = setup_harn()

    # Initializing a FitHarn object can take a little time, but not too much.
    # This is where instances of the model, optimizer, scheduler, monitor, and
    # initializer are created. This is also where we check if there is a
    # pre-existing checkpoint that we can restart from.
    harn.initialize()

    # This starts the main loop which will run until the monitor's terminator
    # criterion is satisfied. If the initialize step loaded a checkpointed that
    # already met the termination criterion, then this will simply return.
    deploy_fpath = harn.run()

    # The returned deploy_fpath is the path to an exported netharn model.
    # This model is the on with the best weights according to the monitor.
    print('deploy_fpath = {!r}'.format(deploy_fpath))
    return harn


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.examples.cifar --gpu=0 --arch=resnet50 --num_vali=0
        python -m netharn.examples.cifar --gpu=0 --arch=efficientnet-b0 --num_vali=0

        python -m netharn.examples.cifar --gpu=0 --arch=efficientnet-b0

        # This next command requires a bit more compute
        python -m netharn.examples.cifar --gpu=0 --arch=efficientnet-b0 --nice=test_cifar2 --schedule=step-3-6-50 --lr=0.1 --init=cls --batch_size=2718

        python -m netharn.examples.cifar --gpu=0 --arch=efficientnet-b0 --nice=test_cifar3 --schedule=step-3-6-12-16 --lr=0.256 --init=cls --batch_size=3000 --workers=2 --num_vali=0 --optim=rmsprop

        python -m netharn.examples.cifar --gpu=0 --arch=efficientnet-b0 --nice=test_cifar3 --schedule=onecycle70 --lr=0.01  --init=cls --batch_size=3000 --workers=2 --num_vali=0 --optim=sgd --datasets=cifar100

        python -m netharn.examples.cifar --gpu=0 --arch=efficientnet-b0 --nice=test_cifar2 --schedule=ReduceLROnPlateau-p1-c1-f0.9 --lr=0.1 --init=cls --batch_size=2719 --workers=4 --optim=sgd --datasets=cifar100

        python -m netharn.examples.cifar.py --gpu=0 --arch=densenet121
        # Train on two GPUs with a larger batch size
        python -m netharn.examples.cifar.py --arch=dpn92 --batch_size=256 --gpu=0,1
    """
    import seaborn
    seaborn.set()
    main()
