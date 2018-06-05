import numpy as np
import ubelt as ub
import torch
import netharn as nh


class CIFAR_FitHarn(nh.FitHarn):

    def run_batch(harn, inputs, labels):
        """
        Custom function to compute the output of a batch and its loss.
        """
        output = harn.model(*inputs)
        label = labels[0]
        loss = harn.criterion(output, label)
        outputs = [output]
        return outputs, loss

    def on_batch(harn, outputs, labels):
        from netharn.metrics import (confusion_matrix,
                                     pixel_accuracy_from_confusion,
                                     perclass_accuracy_from_confusion)

        task = harn.datasets['train'].task
        all_labels = task.labels

        label = labels[0]
        output = outputs[0]

        y_pred = output.data.max(dim=1)[1].cpu().numpy()
        y_true = label.data.cpu().numpy()

        cfsn = confusion_matrix(y_pred, y_true, labels=all_labels)

        global_acc = pixel_accuracy_from_confusion(cfsn)  # same as acc
        perclass_acc = perclass_accuracy_from_confusion(cfsn)
        # class_accuracy = perclass_acc.fillna(0).mean()
        class_accuracy = np.nan_to_num(perclass_acc).mean()

        metrics_dict = ub.odict()
        metrics_dict['global_acc'] = global_acc
        metrics_dict['class_acc'] = class_accuracy
        return metrics_dict


def train():
    """
    Replicates parameters from https://github.com/kuangliu/pytorch-cifar

    Example:
        >>> train()
    """
    import random
    import torchvision
    from torchvision import transforms

    np.random.seed(1031726816 % 4294967295)
    torch.manual_seed(137852547 % 4294967295)
    random.seed(2497950049 % 4294967295)

    batch_size = 128
    lr = 0.1

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
                                         batch_size=batch_size,
                                         pin_memory=True)
        for key, dset in datasets.items()
    }

    initializer_ = (nh.initializers.KaimingNormal, {'param': 0, 'mode': 'fan_in'})
    # initializer_ = (initializers.LSUV, {})

    hyper = nh.HyperParams(
        datasets=datasets,
        nice='cifar10',
        loaders=loaders,
        workdir=workdir,
        xpu=xpu,
        model=(nh.models.densenet.DenseNet, {
            'nblocks': [6, 12, 24, 16],
            'growth_rate': 12,
            'reduction': 0.5,
            'num_classes': n_classes,
        }),
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
    harn = nh.FitHarn(hyper=hyper)
    harn.initialize()
    harn.run()


if __name__ == '__main__':
    r"""
    CommandLine:
        python examples/cifar.py train
        python examples/cifar.py train --lab
        python examples/cifar.py train --rgb-indie
    """
    train()
    # import xdoctest
    # xdoctest.doctest_module(__file__)
