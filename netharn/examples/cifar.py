import numpy as np
import ubelt as ub
import torch
import netharn as nh


class CIFAR_FitHarn(nh.FitHarn):

    def __init__(harn, *args, **kw):
        super().__init__(*args, **kw)
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

        harn.batch_confusions.append((y_true, y_pred))

    def on_epoch(harn):
        """
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 2, 2])
        y_pred = np.array([1, 1, 1, 2, 1, 0, 0, 0, 2, 2])
        all_labels = np.array([0, 1, 2])
        """
        from netharn.metrics import (confusion_matrix,
                                     global_accuracy_from_confusion,
                                     class_accuracy_from_confusion)
        all_labels = np.arange(10)

        all_trues, all_preds = zip(*harn.batch_confusions)
        y_true = np.hstack(all_trues)
        y_pred = np.hstack(all_preds)

        # percent error really isn't a great metric, but its standard.
        errors = (y_true != y_pred)
        percent_error = errors.mean() * 100

        cfsn = confusion_matrix(y_true, y_pred, labels=all_labels)

        global_acc = global_accuracy_from_confusion(cfsn)
        class_acc = class_accuracy_from_confusion(cfsn)

        metrics_dict = ub.odict()
        metrics_dict['global_acc'] = global_acc
        metrics_dict['class_acc'] = class_acc
        metrics_dict['percent_error'] = percent_error

        harn.batch_confusions.clear()
        return metrics_dict


def train():
    """
    Replicates parameters from https://github.com/kuangliu/pytorch-cifar

    The following is a table of kuangliu's reported accuracy and our measured
    accuracy for each model.

          model |  kuangliu  |    ours  |
    -------------------------------------
    ResNet50    |    93.62%  |
    DenseNet121 |    95.04%  |  94.47%  |
    DPN92       |    95.16%  |

     reports the following test accuracies for these models:


    """
    import random
    import torchvision
    from torchvision import transforms

    np.random.seed(1031726816 % 4294967295)
    torch.manual_seed(137852547 % 4294967295)
    random.seed(2497950049 % 4294967295)

    batch_size = 128
    lr = 0.1
    workers = 2

    xpu = nh.XPU.cast('argv')

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

    model_key = ub.argval('--model', default='densenet121')

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
    """
    train()
