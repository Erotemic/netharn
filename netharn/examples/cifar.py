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
    python -m netharn.examples.cifar.py --xpu=0 --arch=resnet50
    python -m netharn.examples.cifar.py --xpu=0 --arch=wrn_22 --lr=0.003 --schedule=onecycle --optim=adamw
    python -m netharn.examples.cifar.py --xpu=1,2,3 --arch=wrn_22 --lr=0.003 --schedule=onecycle --optim=adamw --batch_size=1800
    python -m netharn.examples.cifar.py --xpu=1,2 --arch=resnet50 --lr=0.003 --schedule=onecycle --optim=adamw

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
        'xpu': scfg.Value('auto', help='See netharn.XPU for details. can be auto/cpu/xpu/cuda0/0,1,2,3)'),

        'dataset': scfg.Value('cifar10', choices=['cifar10', 'cifar100'],
                              help='which cifar network to use'),
        'num_vali': scfg.Value(0, help='number of validation examples'),
        'augment': scfg.Value('baseline', help='an augmentation comma separated list or a code'),

        'arch': scfg.Value('resnet50', help='Network architecture code'),
        'optim': scfg.Value('sgd', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'input_dims': scfg.Value((32, 32), help='Image size passed to the network'),

        'batch_size': scfg.Value(64, help='number of items per batch'),

        'max_epoch': scfg.Value(350, help='Maximum number of epochs'),
        'patience': scfg.Value(350, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-1, help='Base learning rate'),
        'decay':  scfg.Value(5e-4, help='Base weight decay'),

        'schedule': scfg.Value('step-150-250', help=('Special coercable netharn code. Eg: onecycle50, step50, gamma')),

        'grad_norm_max': scfg.Value(None, help='clip gradients exceeding this value'),
        'warmup_iters': scfg.Value(0, help='number of iterations to warmup learning rate'),

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

        # percent error really isn't a great metric, but its easy and standard.
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

    The first three experiments are with simple augmentation. The rest have
    more complex augmentation.

           arch        |  kuangliu  | rerun-kuangliu |  netharn |  train rate | num params
    ---------------------------------------------------------------------------------------
    ResNet50           |    93.62%  |        95.370% |  95.72%  |             |
    DenseNet121        |    95.04%  |        95.420% |  94.47%  |             |
    DPN92              |    95.16%  |        95.410% |  94.92%  |             |
    --------------------
    ResNet50_newaug*   |        --  |             -- |  96.13%  |   498.90 Hz | 23,520,842
    EfficientNet-7*    |        --  |             -- |  85.36%  |   214.18 Hz | 63,812,570
    EfficientNet-3*    |        --  |             -- |  86.87%  |   568.30 Hz | 10,711,602
    EfficientNet-0*    |        --  |             -- |  87.13%  |   964.21 Hz |  4,020,358


   600025177002,


    CommandLine:
        python -m netharn.examples.cifar --xpu=0 --nice=resnet50_baseline --arch=resnet50 --optim=sgd --schedule=step-150-250 --lr=0.1
        python -m netharn.examples.cifar --xpu=0 --nice=wrn --arch=wrn_22 --optim=sgd --schedule=step-150-250 --lr=0.1
        python -m netharn.examples.cifar --xpu=0 --nice=densenet --arch=densenet121 --optim=sgd --schedule=step-150-250 --lr=0.1

        python -m netharn.examples.cifar --xpu=0 --nice=se_resnet18 --arch=se_resnet18 --optim=sgd --schedule=step-150-250 --lr=0.01 --init=noop --decay=1e-5 --augment=simple

        python -m netharn.examples.cifar --xpu=0 --nice=resnet50_newaug_b128 --batch_size=128 --arch=resnet50 --optim=sgd --schedule=step-150-250 --lr=0.1 --init=kaiming_normal --augment=simple

        python -m netharn.examples.cifar --xpu=0 --nice=efficientnet7_newaug_b128 --batch_size=128 --arch=efficientnet-b7 --optim=sgd --schedule=step-150-250 --lr=0.1 --init=kaiming_normal --augment=simple

        python -m netharn.examples.cifar --xpu=0 --nice=efficientnet3_newaug_b128 --batch_size=128 --arch=efficientnet-b3 --optim=sgd --schedule=step-150-250 --lr=0.1 --init=kaiming_normal --augment=simple

        python -m netharn.examples.cifar --xpu=0 --nice=efficientnet0_newaug_b128 --batch_size=128 --arch=efficientnet-b0 --optim=sgd --schedule=step-150-250 --lr=0.1 --init=kaiming_normal --augment=simple


        python -m netharn.examples.cifar --xpu=0 --nice=efficientnet0_newaug_yogi \
                --batch_size=1028 --arch=efficientnet-b0 --optim=Yogi \
                --schedule=step-60-120-160-250-350-f5 --decay=5e-4 --lr=0.01549 \
                --init=kaiming_normal --augment=simple --grad_norm_max=35 \
                --warmup_iters=100


        # Params from Cutout paper: https://arxiv.org/pdf/1708.04552.pdf
        python -m netharn.examples.cifar --xpu=0 --nice=repro_cutout \
                --batch_size=128 \
                --arch=efficientnet-b0 \
                --optim=sgd --lr=0.01 --decay=5e-4 \
                --schedule=step-60-120-160-f5 --max_epoch=200 \
                --init=kaiming_normal --augment=simple \
                --grad_norm_max=35 --warmup_iters=100

        0.015219216761025578


        python -m netharn.examples.cifar --xpu=0 --nice=efficientnet7_scratch \
            --arch=efficientnet-b7 --optim=sgd --schedule=step-150-250-350 \
            --batch_size=512 --lr=0.01 --init=noop --decay=1e-5
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

    inplace = True

    # Define preprocessing + augmentation strategy
    if ',' in config['augment']:
        augmentors = config['augment'].split(',')
    elif config['augment'] == 'baseline':
        augmentors = ['crop', 'flip']
    elif config['augment'] == 'simple':
        augmentors = ['crop', 'flip', 'gray', 'cutout']
    else:
        raise KeyError(config['augment'])

    train_augmentors = []

    if 'crop' in augmentors:
        train_augmentors += [
            transforms.RandomCrop(32, padding=4),
        ]
    if 'flip' in augmentors:
        train_augmentors += [
            transforms.RandomHorizontalFlip(),
        ]
    if 'gray' in augmentors:
        train_augmentors += [
            transforms.RandomGrayscale(p=0.1),
        ]
    if 'jitter' in augmentors:
        raise NotImplementedError
        # transforms.RandomChoice([
        #     transforms.ColorJitter(brightness=(0, .01), contrast=(0, .01),
        #                            saturation=(0, .01), hue=(-0.01, 0.01),),
        #     ub.identity,
        # ]),

    # A more general system could infer (and cache) this from the data
    input_mean = (0.4914, 0.4822, 0.4465)
    input_std = (0.4914, 0.4822, 0.4465)

    train_augmentors += [
        transforms.Resize(config['input_dims']),
        transforms.ToTensor(),
        transforms.Normalize(input_mean, input_std, inplace=inplace),
    ]
    if 'cutout' in augmentors:
        train_augmentors += [
            transforms.RandomChoice([  # Cutout
                # transforms.RandomErasing(p=0.5,
                #                          scale=(0.1, 0.4),
                #                          ratio=(1.0, 1.0),
                #                          value='random', inplace=inplace),
                transforms.RandomErasing(p=0.5,
                                         scale=(0.1, 0.4),
                                         ratio=(1.0, 1.0),
                                         value=input_mean, inplace=inplace),
            ])
        ]

    transform_train = transforms.Compose(train_augmentors)

    transform_test = transforms.Compose([
        transforms.Resize(config['input_dims']),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010), inplace=inplace),
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
        key: torch.utils.data.DataLoader(dset, shuffle=(key == 'train'),
                                         num_workers=config['workers'],
                                         batch_size=config['batch_size'],
                                         pin_memory=True)
        for key, dset in datasets.items()
    }

    if config['workers'] > 0:
        # Solves pytorch deadlock issue #1355.
        import cv2
        cv2.setNumThreads(0)

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

    if config['arch'].startswith('se_resnet18'):
        from nethar.models import se_resnet
        model = se_resnet.se_resnet18(
            num_classes=len(classes),
        )

    if config['arch'].startswith('se_resnet50'):
        from nethar.models import se_resnet
        model = se_resnet.se_resnet50(
            num_classes=len(classes),
            pretrained=config['init'] == 'cls',
        )

    if config['arch'].startswith('efficientnet'):
        # Directly create the model instance...
        # (as long as it has an `_initkw` attribute)
        from netharn.models import efficientnet

        zero_gamma = False
        if config['init'] == 'cls':
            model_ = efficientnet.EfficientNet.from_pretrained(
                config['arch'], override_params={
                    'classes': classes,
                    'noli': 'mish'
                })
            print('pretrained cls init')
        else:
            model_ = efficientnet.EfficientNet.from_name(
                config['arch'], override_params={
                    'classes': classes,
                    'noli': 'mish'
                }
            )

        # For efficient nets we need to dramatically reduce the weight decay on
        # the depthwise part of the depthwise separable convolution.  To do
        # this we need to manually construct the param groups for the
        # optimizer.
        model = model_

        params = dict(model.named_parameters())
        key_groups = ub.ddict(list)

        seen_ = set()
        def append_once(group, key):
            if key not in seen_:
                key_groups[group].append(key)
                seen_.add(key)

        if zero_gamma:
            for key, layer in model.trainable_layers(names=True):
                if getattr(layer, '_residual_bn', False):
                    # zero bn after residual layers.
                    layer.weight.data.fill_(0)
                    # dont decay batch norm
                    # append_once('nodecay', key + '.weight')

        for key in params.keys():
            if key.endswith('.bias'):
                append_once('nodecay', key)
            elif 'depthwise_conv' in key:
                append_once('nodecay', key)
            else:
                append_once('default', key)

        named_param_groups = {}
        for group_name, keys in key_groups.items():
            if keys:
                # very important that groups are alway in the same order
                keys = sorted(keys)
                param_group = {
                    'params': list(ub.take(params, keys)),
                }
                named_param_groups[group_name] = param_group

        # Override the default weight decay of chosen groups
        named_param_groups['nodecay']['weight_decay'] = 0

        param_groups = [v for k, v in sorted(named_param_groups.items())]

        optim_cls, optim_kw = optimizer_
        optim = optim_cls(param_groups, **optim_kw)
        optimizer_ = optim
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

    if config['schedule'] == 'onecycle':
        # TODO: Fast AI params
        # TODO: https://github.com/fastai/fastai/blob/c7df6a5948bdaa474f095bf8a36d75dbc1ee8e6a/fastai/callbacks/one_cycle.py
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
            'smoothing': 0.0,
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
            'augment': config['augment'],
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
    # harn.preferences['tensorboard_groups']

    harn.intervals.update({
        'vali': 1,
        'test': 1,
    })

    harn.script_config = config
    return harn


def main():
    # Run your code that sets up your custom FitHarn object.
    harn = setup_harn()

    # Initializing a FitHarn object can take a little time, but not too much.
    # This is where instances of the model, optimizer, scheduler, monitor, and
    # initializer are created. This is also where we check if there is a
    # pre-existing checkpoint that we can restart from.
    harn.initialize()

    if ub.argflag('--lrtest'):
        """
        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b0 \
                --nice=test_cifar9 --optim=adamw --schedule=Exponential-g0.98 \
                --lr=0.1 --init=kaiming_normal \
                --batch_size=2048 --lrtest --show

        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b7 \
                --nice=test_cifar9 --optim=adamw --schedule=Exponential-g0.98 \
                --lr=0.1 --init=kaiming_normal \
                --batch_size=256  --lrtest --show

        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b7 \
                --nice=test_cifar9 --optim=adamw --schedule=Exponential-g0.98 \
                --lr=4e-2 --init=kaiming_normal \
                --batch_size=256
        """
        # Undocumented hidden feature,
        # Perform an LR-test, then resetup the harness. Optionally draw the
        # results using matplotlib.
        from netharn.prefit.lr_tests import lr_range_test

        result = lr_range_test(
            harn, init_value=1e-4, final_value=0.5, beta=0.3,
            explode_factor=10, num_iters=200)

        if ub.argflag('--show'):
            import kwplot
            plt = kwplot.autoplt()
            result.draw()
            plt.show()

        # Recreate a new version of the harness with the recommended LR.
        config = harn.script_config.asdict()
        config['lr'] = (result.recommended_lr * 10)
        harn = setup_harn(**config)
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
        python -m netharn.examples.cifar --xpu=0 --arch=resnet50 --num_vali=0
        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b0 --num_vali=0

        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b0

        # This next command requires a bit more compute
        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b0 --nice=test_cifar2 --schedule=step-3-6-50 --lr=0.1 --init=cls --batch_size=2718

        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b0 --nice=test_cifar3 --schedule=step-3-6-12-16 --lr=0.256 --init=cls --batch_size=3000 --workers=2 --num_vali=0 --optim=rmsprop

        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b0 --nice=test_cifar3 --schedule=onecycle70 --lr=0.01  --init=cls --batch_size=3000 --workers=2 --num_vali=0 --optim=sgd --datasets=cifar100

        python -m netharn.examples.cifar --xpu=0 --arch=efficientnet-b0 --nice=test_cifar2 --schedule=ReduceLROnPlateau-p1-c1-f0.9 --lr=0.1 --init=cls --batch_size=2719 --workers=4 --optim=sgd --datasets=cifar100

        python -m netharn.examples.cifar.py --xpu=0 --arch=densenet121
        # Train on two GPUs with a larger batch size
        python -m netharn.examples.cifar.py --arch=dpn92 --batch_size=256 --xpu=0,1
    """
    import seaborn
    seaborn.set()
    main()
