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
"""
import numpy as np
import ubelt as ub
import torch
import netharn as nh


class CIFAR_FitHarn(nh.FitHarn):

    def __init__(harn, *args, **kw):
        super(CIFAR_FitHarn, self).__init__(*args, **kw)
        harn.batch_confusions = []

    def run_batch(harn, batch):
        """
        Custom function to compute the output of a batch and its loss.
        """
        inputs, labels = batch
        output = harn.model(*inputs)
        label = labels[0]
        loss = harn.criterion(output, label)
        outputs = [output]
        return outputs, loss

    def on_batch(harn, batch, outputs, loss):
        inputs, labels = batch
        label = labels[0]
        output = outputs[0]

        y_pred = output.data.max(dim=1)[1].cpu().numpy()
        y_true = label.data.cpu().numpy()
        probs = output.data.cpu().numpy()

        harn.batch_confusions.append((y_true, y_pred, probs))

    def on_epoch(harn):
        """
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 2, 2])
        y_pred = np.array([1, 1, 1, 2, 1, 0, 0, 0, 2, 2])
        all_labels = np.array([0, 1, 2])
        """
        from netharn.metrics import clf_report

        dset = harn.datasets[harn.current_tag]
        target_names = dset.class_names

        all_trues, all_preds, all_probs = zip(*harn.batch_confusions)

        probs = np.vstack(all_probs)
        y_true = np.hstack(all_trues)
        y_pred = np.hstack(all_preds)

        # Compute multiclass metrics (new way!)
        report = clf_report.ovr_classification_report(
            y_true, probs, target_names=target_names, metrics=[
                'auc', 'ap', 'mcc', 'brier'
            ])
        #print(ub.repr2(report))

        # from netharn.metrics import (confusion_matrix,
        #                              global_accuracy_from_confusion,
        #                              class_accuracy_from_confusion)
        # all_labels = np.arange(10)
        # percent error really isn't a great metric, but its standard.
        errors = (y_true != y_pred)
        percent_error = errors.mean() * 100
        # cfsn = confusion_matrix(y_true, y_pred, labels=all_labels)

        # global_acc = global_accuracy_from_confusion(cfsn)
        # class_acc = class_accuracy_from_confusion(cfsn)

        metrics_dict = ub.odict()
        # metrics_dict['global_acc'] = global_acc
        # metrics_dict['class_acc'] = class_acc
        metrics_dict['ave_brier'] = report['ave']['brier']
        metrics_dict['ave_mcc'] = report['ave']['mcc']
        metrics_dict['ave_auc'] = report['ave']['auc']
        metrics_dict['ave_ap'] = report['ave']['ap']
        metrics_dict['percent_error'] = percent_error
        metrics_dict['acc'] = 1 - percent_error

        harn.batch_confusions.clear()
        return metrics_dict


def train():
    """
    Replicates parameters from https://github.com/kuangliu/pytorch-cifar

    The following is a table of kuangliu's reported accuracy and our measured
    accuracy for each model.

    The first column is kuangliu's reported accuracy, the second column is me
    running kuangliu's code, and the final column is using my own training
    harness (handles logging and whatnot) called netharn.

          model |  kuangliu  | rerun-kuangliu  |  netharn |
    -------------------------------------------------------
    ResNet50    |    93.62%  |         95.370% |  95.72%  |  <- how did that happen?
    DenseNet121 |    95.04%  |         95.420% |  94.47%  |
    DPN92       |    95.16%  |         95.410% |  94.92%  |

    """
    import random
    import torchvision
    from torchvision import transforms

    np.random.seed(1031726816 % 4294967295)
    torch.manual_seed(137852547 % 4294967295)
    random.seed(2497950049 % 4294967295)

    # batch_size = int(ub.argval('--batch_size', default=128))
    batch_size = int(ub.argval('--batch_size', default=64))
    workers = int(ub.argval('--workers', default=2))
    model_key = ub.argval('--model', default='densenet121')
    xpu = nh.XPU.cast('argv')

    lr = 0.1

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

    workdir = ub.ensure_app_cache_dir('netharn')

    datasets = {
        'train': torchvision.datasets.CIFAR10(root=workdir, train=True,
                                              download=True,
                                              transform=transform_train),
        'test': torchvision.datasets.CIFAR10(root=workdir, train=False,
                                             download=True,
                                             transform=transform_test),
    }

    # For some reason the torchvision objects dont have the label names
    CIFAR10_CLASSNAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck',
    ]
    datasets['train'].class_names = CIFAR10_CLASSNAMES
    datasets['test'].class_names = CIFAR10_CLASSNAMES

    n_classes = 10  # hacked in
    loaders = {
        key: torch.utils.data.DataLoader(dset, shuffle=key == 'train',
                                         num_workers=workers,
                                         batch_size=batch_size,
                                         pin_memory=True)
        for key, dset in datasets.items()
    }

    if workers > 0:
        import cv2
        cv2.setNumThreads(0)

    initializer_ = (nh.initializers.KaimingNormal, {'param': 0, 'mode': 'fan_in'})
    # initializer_ = (initializers.LSUV, {})

    available_models = {
        'densenet121': (nh.models.densenet.DenseNet, {
            'nblocks': [6, 12, 24, 16],
            'growth_rate': 12,
            'reduction': 0.5,
            'num_classes': n_classes,
        }),

        'resnet50': (nh.models.resnet.ResNet, {
            'num_blocks': [3, 4, 6, 3],
            'num_classes': n_classes,
            'block': 'Bottleneck',
        }),

        'dpn26': (nh.models.dual_path_net.DPN, dict(cfg={
            'in_planes': (96, 192, 384, 768),
            'out_planes': (256, 512, 1024, 2048),
            'num_blocks': (2, 2, 2, 2),
            'dense_depth': (16, 32, 24, 128),
            'num_classes': n_classes,
        })),

        'dpn92': (nh.models.dual_path_net.DPN, dict(cfg={
            'in_planes': (96, 192, 384, 768),
            'out_planes': (256, 512, 1024, 2048),
            'num_blocks': (3, 4, 20, 3),
            'dense_depth': (16, 32, 24, 128),
            'num_classes': n_classes,
        })),
    }

    model_ = available_models[model_key]

    hyper = nh.HyperParams(
        datasets=datasets,
        nice='cifar10_' + model_key,
        loaders=loaders,
        workdir=workdir,
        xpu=xpu,
        model=model_,
        optimizer=(torch.optim.SGD, {
            'lr': lr,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'nesterov': True,
        }),
        scheduler=(nh.schedulers.ListedLR, {
            'points': {
                0: lr,
                150: lr * 0.1,
                250: lr * 0.01,
            },
            'interpolate': False
        }),
        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': 350,
            'max_epoch': 350,
        }),
        initializer=initializer_,
        criterion=(torch.nn.CrossEntropyLoss, {}),
        # Specify anything else that is special about your hyperparams here
        # Especially if you make a custom_batch_runner
        # TODO: type of augmentation as a parameter dependency
        # augment=str(datasets['train'].augmenter),
        # other=ub.dict_union({
        #     # 'colorspace': datasets['train'].output_colorspace,
        # }, datasets['train'].center_inputs.__dict__),
    )
    harn = CIFAR_FitHarn(hyper=hyper)
    harn.initialize()
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python examples/cifar.py --gpu=0 --model=densenet121
        python examples/cifar.py --gpu=0 --model=resnet50
        # Train on two GPUs with a larger batch size
        python examples/cifar.py --model=dpn92 --batch_size=256 --gpu=0,1
    """
    train()
