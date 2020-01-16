"""
The examples/cifar.py is probably the most clear example of what netharn is and
what it's trying to do / not do.

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
    python examples/cifar.py --gpu=0 --arch=resnet50
    python examples/cifar.py --gpu=0 --arch=wrn_22 --lr=0.003 --schedule=onecycle --optim=adamw
    python examples/cifar.py --gpu=1,2,3 --arch=wrn_22 --lr=0.003 --schedule=onecycle --optim=adamw --batch_size=1800
    python examples/cifar.py --gpu=1,2 --arch=resnet50 --lr=0.003 --schedule=onecycle --optim=adamw

"""
import sys
from os.path import join
import numpy as np
import ubelt as ub
import torch
import os
import pickle
import netharn as nh


class CIFAR_FitHarn(nh.FitHarn):
    """
    The `FitHarn` class contains a lot of reusable boilerplate. We inherit
    from it and override relevant methods to customize the training procedure
    to our particular problem and dataset.

    Example:
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
        probs = outputs.data.cpu().numpy()

        bx = harn.bxs[harn.current_tag]
        if bx < 3:
            decoded = harn._decode(outputs, batch['label'])
            stacked = harn._draw_batch(batch, decoded)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'batch_{}_epoch_{}.jpg'.format(bx, harn.epoch))
            nh.util.imwrite(fpath, stacked)

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
            >>> harn = setup_harn().initialize()
            >>> harn._demo_epoch('vali')
        """
        from netharn.metrics import clf_report

        dset = harn.datasets[harn.current_tag]
        target_names = dset.categories

        probs = np.vstack(harn._accum_confusion_vectors['probs'])
        y_true = np.hstack(harn._accum_confusion_vectors['y_true'])
        y_pred = np.hstack(harn._accum_confusion_vectors['y_pred'])

        # Compute multiclass metrics (new way!)
        report = clf_report.ovr_classification_report(
            y_true, probs, target_names=target_names, metrics=[
                'auc', 'ap', 'mcc', 'brier'
            ])

        # percent error really isn't a great metric, but its standard.
        errors = (y_true != y_pred)
        acc = 1 - errors.mean()
        percent_error = (1 - acc) * 100

        metrics_dict = ub.odict()
        metrics_dict['ave_brier'] = report['ave']['brier']
        metrics_dict['ave_mcc'] = report['ave']['mcc']
        metrics_dict['ave_auc'] = report['ave']['auc']
        metrics_dict['ave_ap'] = report['ave']['ap']
        metrics_dict['percent_error'] = percent_error
        metrics_dict['acc'] = acc

        # Clear confusion vectors accumulator for the next epoch
        harn._accum_confusion_vectors = {
            'y_true': [],
            'y_pred': [],
            'probs': [],
        }
        return metrics_dict

    def _decode(harn, outputs, true_cxs=None):
        class_probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_scores, pred_cxs = class_probs.data.max(dim=1)

        decoded = {
            'class_probs': class_probs,
            'pred_cxs': pred_cxs,
            'pred_scores': pred_scores,
        }
        if true_cxs is not None:
            import kwarray
            hot = kwarray.one_hot_embedding(true_cxs, class_probs.shape[1])
            true_probs = (hot * class_probs).sum(dim=1)
            decoded['true_scores'] = true_probs
        return decoded

    def _draw_batch(harn, batch, decoded, limit=32):
        """
        CommandLine:
            xdoctest -m ~/code/netharn/examples/cifar.py CIFAR_FitHarn._draw_batch --show --arch=resnet50

        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/netharn/examples')
            >>> from cifar import *
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='test')
            >>> outputs, loss = harn.run_batch(batch)
            >>> decoded = harn._decode(outputs, batch['label'])
            >>> stacked = harn._draw_batch(batch, decoded, limit=42)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked, colorspace='rgb', doclf=True)
            >>> kwplot.show_if_requested()
        """
        import kwimage
        inputs = batch['input']
        inputs = inputs[0:limit]

        input_shape = inputs.shape
        dims = [160] * (len(input_shape) - 2)
        min_, max_ = inputs.min(), inputs.max()
        inputs = (inputs - min_) / (max_ - min_)
        inputs = torch.nn.functional.interpolate(inputs, size=dims)
        inputs = (inputs * 255).byte()
        inputs = inputs.data.cpu().numpy()

        dset = harn.datasets[harn.current_tag]
        catgraph = dset.categories

        pred_cxs = decoded['pred_cxs'].data.cpu().numpy()
        pred_scores = decoded['pred_scores'].data.cpu().numpy()

        true_cxs = batch['label'].data.cpu().numpy()
        true_scores = decoded['true_scores'].data.cpu().numpy()

        todraw = []
        for im, pcx, tcx, pred_score, true_score in zip(inputs, pred_cxs, true_cxs, pred_scores, true_scores):
            im_ = im.transpose(1, 2, 0)
            im_ = np.ascontiguousarray(im_)
            h, w = im_.shape[0:2][::-1]

            true_name = catgraph[tcx]
            pred_name = catgraph[pcx]
            org1 = np.array((2, h - 32))
            org2 = np.array((2, 25))
            pred_label = 'p:{pcx}@{pred_score:.2f}:\n{pred_name}'.format(**locals())
            true_label = 't:{tcx}@{true_score:.2f}:\n{true_name}'.format(**locals())
            if pcx == tcx:
                true_label = 't:{tcx}:{true_name}'.format(**locals())

            fontkw = {
                'fontScale': 1.0,
                'thickness': 2
            }
            color = 'dodgerblue' if pcx == tcx else 'orangered'

            im_ = kwimage.draw_text_on_image(im_, pred_label, org=org1 - 2,
                                             color='white', **fontkw)
            im_ = kwimage.draw_text_on_image(im_, true_label, org=org2 - 2,
                                             color='white', **fontkw)

            for i in [-2, -1, 1, 2]:
                for j in [-2, -1, 1, 2]:
                    im_ = kwimage.draw_text_on_image(im_, pred_label, org=org1 + i,
                                                     color='black', **fontkw)
                    im_ = kwimage.draw_text_on_image(im_, true_label, org=org2 + j,
                                                     color='black', **fontkw)

            im_ = kwimage.draw_text_on_image(im_, pred_label, org=org1,
                                             color=color, **fontkw)
            im_ = kwimage.draw_text_on_image(im_, true_label, org=org2,
                                             color='lawngreen', **fontkw)
            todraw.append(im_)

        stacked = kwimage.stack_images_grid(todraw, overlap=-10, bg_value=(10, 40, 30), chunksize=8)
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
        python examples/cifar.py --gpu=0 --arch=resnet50 --optim=sgd --schedule=step250 --lr=0.1 --vd
        python examples/cifar.py --gpu=0 --arch=wrn_22 --optim=sgd --schedule=step250 --lr=0.1 --vd
    """
    import random
    import torchvision
    from torchvision import transforms

    # Note that most netharn training scripts will use scriptconfig instead of
    # this more explicit approach.
    config = {
        # A conservative traditional baseline
        'arch': ub.argval('--arch', default='resnet50'),
        'lr': float(ub.argval('--lr', default=0.1)),
        'schedule': ub.argval('--schedule', default='step250'),
        'optim': ub.argval('--optim', default='sgd'),

        'batch_size': int(ub.argval('--batch_size', default=64)),

        'workers': int(ub.argval('--workers', default=2)),
        'xpu': ub.argval('--xpu', default='argv'),

        'dataset': ub.argval('--dataset', default='cifar10'),
        'workdir': ub.expandpath(ub.argval('--workdir', default='~/work/cifar')),

        'seed': int(ub.argval('--seed', default=137852547)),
        'deterministic': False,

    }
    xpu = nh.XPU.coerce(config['xpu'])

    # The work directory is where all intermediate results are dumped.
    ub.ensuredir(config['workdir'])

    # Take care of random seeding and ensuring appropriate determinisim
    torch.manual_seed((config['seed'] + 0) % int(2 ** 32 - 1))
    random.seed((config['seed'] + 2360097502) % int(2 ** 32 - 1))
    np.random.seed((config['seed'] + 893874269) % int(2 ** 32 - 1))
    if torch.backends.cudnn.enabled:
        # TODO: ensure the CPU mode is also deterministic
        torch.backends.cudnn.deterministic = config['deterministic']

    # Define augmentation strategy
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if config['dataset'] == 'cifar10':
        DATASET = torchvision.datasets.CIFAR10
        dset = DATASET(root=config['workdir'], download=True)
        meta_fpath = os.path.join(dset.root, dset.base_folder, 'batches.meta')
        meta_dict = pickle.load(open(meta_fpath, 'rb'))
        categories = meta_dict['label_names']
        # For some reason the torchvision objects dont have the label names
        # in the dataset. But the download directory will have them.
        # categories = [
        #     'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        #     'horse', 'ship', 'truck',
        # ]
    elif config['dataset'] == 'cifar100':
        DATASET = torchvision.datasets.CIFAR100
        dset = DATASET(root=config['workdir'], download=True)
        meta_fpath = os.path.join(dset.root, dset.base_folder, 'meta')
        meta_dict = pickle.load(open(meta_fpath, 'rb'))
        categories = meta_dict['fine_label_names']
        # categories = [
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
    if True:
        # Create a test train split
        learn = datasets['train']
        indices = np.arange(len(learn))
        indices = nh.util.shuffle(indices, rng=0)
        num_vali = 300
        datasets['vali'] = torch.utils.data.Subset(learn, indices[:num_vali])
        datasets['train'] = torch.utils.data.Subset(learn, indices[num_vali:])

    # For some reason the torchvision objects do not make the category names
    # easily available. We set them here for ease of use.
    reduction = int(ub.argval('--reduction', default=1))
    for key, dset in datasets.items():
        dset.categories = categories
        if reduction > 1:
            indices = np.arange(len(dset))[::reduction]
            dset = torch.utils.data.Subset(dset, indices)
        dset.categories = categories
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
            'num_classes': len(categories),
        }),

        'resnet50': (nh.models.resnet.ResNet, {
            'num_blocks': [3, 4, 6, 3],
            'num_classes': len(categories),
            'block': 'Bottleneck',
        }),

        'dpn26': (nh.models.dual_path_net.DPN, dict(cfg={
            'in_planes': (96, 192, 384, 768),
            'out_planes': (256, 512, 1024, 2048),
            'num_blocks': (2, 2, 2, 2),
            'dense_depth': (16, 32, 24, 128),
            'num_classes': len(categories),
        })),

        'dpn92': (nh.models.dual_path_net.DPN, dict(cfg={
            'in_planes': (96, 192, 384, 768),
            'out_planes': (256, 512, 1024, 2048),
            'num_blocks': (3, 4, 20, 3),
            'dense_depth': (16, 32, 24, 128),
            'num_classes': len(categories),
        })),
    }

    if config['arch'].startswith('wrn'):
        import fastai
        import fastai.vision
        available_architectures['wrn_22'] = (
            fastai.vision.models.WideResNet, dict(
                num_groups=3, N=3, num_classes=10, k=6, drop_p=0.
            )
        )

    model_ = available_architectures[config['arch']]

    # Note there are lots of different initializers including a special
    # pretrained initializer.
    initializer_ = (nh.initializers.KaimingNormal, {'param': 0, 'mode': 'fan_in'})

    if config['schedule'] == 'step250':
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
        raise KeyError(config['schedule'])

    if config['optim'] == 'sgd':
        optimizer_ = (torch.optim.SGD, {
            'lr': config['lr'],
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'nesterov': True,
        })
    elif config['optim'] == 'adamw':
        optimizer_ = (nh.optimizers.AdamW, {
            'lr': config['lr'],
            'betas': (0.9, 0.999),
            'weight_decay': 0,
            'amsgrad': False,
        })
    else:
        raise KeyError(config['optim'])

    # Notice that arguments to hyperparameters are typically specified as a
    # tuple of (type, Dict), where the dictionary are the keyword arguments
    # that can be used to instantiate an instance of that class. While
    # this may be slightly awkward, it enables netharn to track hyperparameters
    # more effectively. Note that it is possible to simply pass an already
    # constructed instance of a class, but this causes information loss.
    hyper = nh.HyperParams(
        # Datasets must be preconstructed
        datasets=datasets,
        nice='cifar10_' + config['arch'],
        # Loader may be preconstructed
        loaders=loaders,
        workdir=config['workdir'],
        xpu=xpu,
        # The 6 major hyper components are best specified as a Tuple[type, dict]
        # However, in recent releases of netharn, these may be preconstructed
        # as well.
        model=model_,
        optimizer=optimizer_,
        scheduler=scheduler_,
        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': 350,
            'max_epoch': 350,
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
            'config': ub.repr2(config.asdict()),
            'argv': sys.argv,
        }
    )

    # Creating an instance of a Fitharn object is typically fast.
    harn = CIFAR_FitHarn(hyper=hyper)
    harn.config['prog_backend'] = 'progiter'
    return harn


def main():
    harn = setup_harn()

    # Initializing a FitHarn object can take a little time, but not too much.
    # This is where instances of the model, optimizer, scheduler, monitor, and
    # initializer are created. This is also where we check if there is a
    # pre-existing checkpoint that we can restart from.
    harn.initialize()

    if ub.argval(('--vd', '--view-directory')):
        ub.startfile(harn.train_dpath)

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
        python examples/cifar.py --gpu=0 --arch=resnet50

        python examples/cifar.py --gpu=0 --arch=densenet121
        # Train on two GPUs with a larger batch size
        python examples/cifar.py --arch=dpn92 --batch_size=256 --gpu=0,1
    """
    main()
