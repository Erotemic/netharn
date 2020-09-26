"""
Adapated from : https://github.com/rrmina/fast-neural-style-pytorch/blob/master/video.py
"""
from torchvision import models, transforms
import sys
import torch
import torch.nn as nn
import numpy as np
import ubelt as ub
import ndsampler
import netharn as nh
import scriptconfig as scfg
import kwimage


class StyleTransferConfig(scfg.Config):
    default = {
        'name': scfg.Value('style_example', help='A human readable tag that is "name" for humans'),
        'workdir': scfg.Path('~/work/netharn', help='Dump all results in your workdir'),

        'workers': scfg.Value(2, help='number of parallel dataloading jobs'),
        'xpu': scfg.Value('auto', help='See netharn.XPU for details. can be auto/cpu/xpu/cuda0/0,1,2,3)'),

        'datasets': scfg.Value('special:shapes256', help='Either a special key or a coco file'),
        'train_dataset': scfg.Value(None),
        'vali_dataset': scfg.Value(None),
        'test_dataset': scfg.Value(None),

        'sampler_backend': scfg.Value(None, help='ndsampler backend'),

        'channels': scfg.Value('rgb', help='special channel code. See ChannelSpec'),

        # 'arch': scfg.Value('resnet50', help='Network architecture code'),
        'optim': scfg.Value('adam', help='Weight optimizer. Can be SGD, ADAM, ADAMW, etc..'),

        'input_dims': scfg.Value((256, 256), help='Window size to input to the network'),

        # TODO
        'normalize_inputs': scfg.Value(True, help=(
            'if True, precompute training mean and std for data whitening')),

        'balance': scfg.Value(None, help='balance strategy. Can be category or None'),
        # 'augmenter': scfg.Value('simple', help='type of training dataset augmentation'),

        'batch_size': scfg.Value(3, help='number of items per batch'),
        'num_batches': scfg.Value('auto', help='Number of batches per epoch (mainly for balanced batch sampling)'),

        'max_epoch': scfg.Value(140, help='Maximum number of epochs'),
        'patience': scfg.Value(140, help='Maximum "bad" validation epochs before early stopping'),

        'lr': scfg.Value(1e-4, help='Base learning rate'),
        'decay':  scfg.Value(1e-5, help='Base weight decay'),
        'schedule': scfg.Value(
            'step90-120', help=(
                'Special coercible netharn code. Eg: onecycle50, step50, gamma, ReduceLROnPlateau-p10-c10')),
        'init': scfg.Value('noop', help='How to initialized weights: e.g. noop, kaiming_normal, path-to-a-pretrained-model)'),
        # 'pretrained': scfg.Path(help=('alternative way to specify a path to a pretrained model')),
    }


class StyleTransferHarn(nh.FitHarn):

    def after_initialize(harn):
        STYLE_IMAGE_PATH = ub.grabdata('https://raw.githubusercontent.com/iamRusty/fast-neural-style-pytorch/master/images/mosaic.jpg')

        device = harn.xpu.device
        harn.MSELoss = nn.MSELoss().to(device)

        vgg_path = ub.grabdata('https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth')

        # TODO: should be tracked
        harn.vgg = VGG16(**{'vgg_path': vgg_path})
        harn.vgg = harn.xpu.move(harn.vgg)

        def itot(img, max_size=None):
            # Rescale the image
            if (max_size is None):
                itot_t = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.mul(255))
                ])
            else:
                H, W, C = img.shape
                image_size = tuple(
                    [int((float(max_size) / max([H, W])) * x) for x in [H, W]])
                itot_t = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.mul(255))
                ])

            # Convert image to tensor
            tensor = itot_t(img)

            # Add the batch_size dimension
            tensor = tensor.unsqueeze(dim=0)
            return tensor

        # Get Style Features
        imagenet_neg_mean = torch.tensor(
            [-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
        style_image = kwimage.imread(STYLE_IMAGE_PATH)
        style_tensor = itot(style_image).to(device)
        style_tensor = style_tensor.add(imagenet_neg_mean)
        B, C, H, W = style_tensor.shape

        harn.imagenet_neg_mean = imagenet_neg_mean
        harn.style_tensor = style_tensor

        batch_size = harn.script_config['batch_size']
        im = style_tensor.expand([batch_size, C, H, W])

        style_features = harn.vgg(im)
        style_gram = {}
        for key, value in style_features.items():
            style_gram[key] = gram(value)
        harn.style_gram = style_gram

    def run_batch(harn, batch):
        """
        Ignore:
            import sys, ubelt
            sys.path.append(ubelt.expandpath('~/code/netharn'))
            from netharn.examples.style_transfer import *  # NOQA
            kw = {}
            cmdline = False
            harn = setup_harn()
            harn.initialize()
            batch = harn._demo_batch()
            harn.run_batch(batch)
        """
        # Current Batch size in case of odd batches
        content_batch, _ = batch
        curr_batch_size = content_batch.shape[0]

        model = harn.model
        # Zero-out Gradients

        # Generate images and get features
        content_batch = harn.xpu.move(content_batch[:, [2, 1, 0]])

        generated_batch = model(content_batch)

        generated_batch = harn.model(content_batch)
        content_features = harn.vgg(content_batch.add(harn.imagenet_neg_mean))
        generated_features = harn.vgg(generated_batch.add(harn.imagenet_neg_mean))

        # Content Loss
        CONTENT_WEIGHT = 17
        STYLE_WEIGHT = 50

        content_loss = CONTENT_WEIGHT * \
            harn.MSELoss(
                content_features['relu2_2'],
                generated_features['relu2_2'])

        # Style Loss
        style_loss = 0
        for key, value in generated_features.items():
            s_loss = harn.MSELoss(
                gram(value),
                harn.style_gram[key][:curr_batch_size]
            )
            style_loss += s_loss
        style_loss *= STYLE_WEIGHT

        # Total Loss
        loss_parts = {
            'content_loss': content_loss,
            'style_loss': style_loss,
        }
        return generated_batch, loss_parts

    def on_batch(harn, batch, generated_batch, loss):
        _do_draw = harn.batch_index % 500 == 0
        _do_draw |= harn.batch_index < 4
        if _do_draw:
            # Save sample generated image
            from os.path import join
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
            sample_image = sample_tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
            sample_image_path = join(dpath, "sample0_" + str(harn.batch_index) + '_' + str(harn.batch_index) + ".png")
            kwimage.imwrite(sample_image_path, sample_image.clip(0, 255))
            print("Saved sample tranformed image at {}".format(sample_image_path))


class SamplerDataset(torch.utils.data.Dataset):
    def __init__(self, sampler, transform=None, return_style='torchvision'):
        self.sampler = sampler
        self.transform = transform
        self.return_style = return_style
        self.input_id = self.sampler.hashid

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        item = self.sampler.load_item(index)
        numpy_im = item['im']

        if self.transform:
            from PIL import Image
            pil_im = Image.fromarray(numpy_im)
            torch_chw = self.transform(pil_im)
        else:
            torch_chw = torch.from_numpy(numpy_im).permute(2, 0, 1).float()
            # raise NotImplementedError

        if self.return_style == 'torchvision':
            cid = item['tr']['category_id']
            cidx = self.sampler.classes.id_to_idx[cid]
            return torch_chw, cidx
        else:
            raise NotImplementedError

    def make_loader(self, batch_size=16, num_batches='auto', num_workers=0,
                    shuffle=False, pin_memory=False, drop_last=False,
                    balance=None):

        import kwarray
        if len(self) == 0:
            raise Exception('must have some data')

        def worker_init_fn(worker_id):
            for i in range(worker_id + 1):
                seed = np.random.randint(0, int(2 ** 32) - 1)
            seed = seed + worker_id
            kwarray.seed_global(seed)
            # if self.augmenter:
            #     rng = kwarray.ensure_rng(None)
            #     self.augmenter.seed_(rng)

        loaderkw = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'worker_init_fn': worker_init_fn,
        }
        if balance is None:
            loaderkw['shuffle'] = shuffle
            loaderkw['batch_size'] = batch_size
            loaderkw['drop_last'] = drop_last
        elif balance == 'classes':
            from netharn.data.batch_samplers import BalancedBatchSampler
            index_to_cid = [
                cid for cid in self.sampler.regions.targets['category_id']
            ]
            batch_sampler = BalancedBatchSampler(
                index_to_cid, batch_size=batch_size,
                shuffle=shuffle, num_batches=num_batches)
            loaderkw['batch_sampler'] = batch_sampler
        else:
            raise KeyError(balance)

        loader = torch.utils.data.DataLoader(self, **loaderkw)
        return loader


def setup_harn(cmdline=False, **kw):
    """
    Ignore:
        kw = {}
        cmdline = False
        harn = setup_harn()
        harn.initialize()
        batch = harn._demo_batch()
    """
    config = StyleTransferConfig(default=kw)
    config.load(cmdline=cmdline)
    print('config = {}'.format(ub.repr2(config.asdict())))

    nh.configure_hacks(config)

    dataset_info = coerce_datasets(config)

    # input_stats = dataset_info['input_stats']

    model = (TransformerNetwork, {})

    hyper = nh.HyperParams(
        name=config['name'],

        workdir=config['workdir'],
        xpu=nh.XPU.coerce(config['xpu']),

        datasets=dataset_info['torch_datasets'],
        loaders=dataset_info['torch_loaders'],

        model=model,
        criterion=None,

        optimizer=nh.Optimizer.coerce(config),
        dynamics=nh.Dynamics.coerce(config),
        scheduler=nh.Scheduler.coerce(config),

        initializer=None,

        monitor=(nh.Monitor, {
            'minimize': ['loss'],
            'patience': config['patience'],
            'max_epoch': config['max_epoch'],
            'smoothing': 0.0,
        }),
        other={
            'name': config['name'],
            'batch_size': config['batch_size'],
            'balance': config['balance'],
        },
        extra={
            'argv': sys.argv,
            'config': ub.repr2(config.asdict()),
        }
    )
    harn = StyleTransferHarn(hyper=hyper)
    harn.preferences.update({
        'num_keep': 3,
        'keep_freq': 10,
        'tensorboard_groups': ['loss'],
        'eager_dump_tensorboard': True,
    })
    harn.intervals.update({})
    harn.script_config = config
    return harn


def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H * W)
    x_t = x.transpose(1, 2)
    return torch.bmm(x, x_t) / (C * H * W)


class VGG16(nn.Module):
    def __init__(self, vgg_path="models/vgg16-00b39a1b.pth"):
        super(VGG16, self).__init__()
        self.vgg_path = vgg_path
        # Load VGG Skeleton, Pretrained Weights
        vgg16_features = models.vgg16(pretrained=False)
        vgg16_features.load_state_dict(torch.load(vgg_path), strict=False)
        self.features = vgg16_features.features

        # Turn-off Gradient History
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if name == '22':
                    break

        return features


class TransformerNetwork(nn.Module):
    """Feedforward Transformation Network without Tanh
    reference: https://arxiv.org/abs/1603.08155
    exact architecture: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """

    def __init__(self):
        super().__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out


class TransformerNetworkTanh(TransformerNetwork):
    """A modification of the transformation network that uses Tanh function as output
    This follows more closely the architecture outlined in the original paper's supplementary material
    his net produces darker images and provides retro styling effect
    Reference: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    """
    # override __init__ method

    def __init__(self, tanh_multiplier=150):
        super(TransformerNetworkTanh, self).__init__()
        # Add a Tanh layer before output
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None"),
            nn.Tanh()
        )
        self.tanh_multiplier = tanh_multiplier

    # Override forward method
    def forward(self, x):
        return super(TransformerNetworkTanh, self).forward(
            x) * self.tanh_multiplier


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if (norm == "instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm == "batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type == "None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out


class ResidualLayer(nn.Module):
    """
    Deep Residual Learning for Image Recognition

    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, channels=128, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x                     # preserve residual
        out = self.relu(self.conv1(x))   # 1st conv layer + activation
        out = self.conv2(out)            # 2nd conv layer
        out = out + identity             # add residual
        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding_size,
            output_padding)

        # Normalization Layers
        self.norm_type = norm
        if (norm == "instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm == "batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if (self.norm_type == "None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out


def coerce_datasets(config):
    coco_datasets = nh.api.Datasets.coerce(config)
    print('coco_datasets = {}'.format(ub.repr2(coco_datasets, nl=1)))
    for tag, dset in coco_datasets.items():
        dset._build_hashid(hash_pixels=False)

    workdir = ub.ensuredir(ub.expandpath(config['workdir']))
    samplers = {
        tag: ndsampler.CocoSampler(dset, workdir=workdir, backend=config['sampler_backend'])
        for tag, dset in coco_datasets.items()
    }

    for tag, sampler in ub.ProgIter(list(samplers.items()), desc='prepare frames'):
        sampler.frames.prepare(workers=config['workers'])

    # TODO: basic ndsampler torch dataset, likely has to support the transforms
    # API, bleh.

    TRAIN_IMAGE_SIZE = 256
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    torch_datasets = {
        key: SamplerDataset(
            sapmler, transform=transform,
            # input_dims=config['input_dims'],
            # augmenter=config['augmenter'] if key == 'train' else None,
        )
        for key, sapmler in samplers.items()
    }
    # self = torch_dset = torch_datasets['train']

    if config['normalize_inputs']:
        # Get stats on the dataset (todo: turn off augmentation for this)
        import kwarray
        _dset = torch_datasets['train']
        stats_idxs = kwarray.shuffle(np.arange(len(_dset)), rng=0)[0:min(1000, len(_dset))]
        stats_subset = torch.utils.data.Subset(_dset, stats_idxs)

        cacher = ub.Cacher('dset_mean', cfgstr=_dset.input_id + 'v3')
        input_stats = cacher.tryload()

        from netharn.data.channel_spec import ChannelSpec
        channels = ChannelSpec.coerce(config['channels'])

        if input_stats is None:
            # Use parallel workers to load data faster
            from netharn.data.data_containers import container_collate
            from functools import partial
            collate_fn = partial(container_collate, num_devices=1)

            loader = torch.utils.data.DataLoader(
                stats_subset,
                collate_fn=collate_fn,
                num_workers=config['workers'],
                shuffle=True,
                batch_size=config['batch_size'])

            # Track moving average of each fused channel stream
            channel_stats = {key: nh.util.RunningStats()
                             for key in channels.keys()}
            assert len(channel_stats) == 1, (
                'only support one fused stream for now')
            for batch in ub.ProgIter(loader, desc='estimate mean/std'):
                if isinstance(batch, (tuple, list)):
                    inputs = {'rgb': batch[0]}  # make assumption
                else:
                    inputs = batch['inputs']

                for key, val in inputs.items():
                    try:
                        for part in val.numpy():
                            channel_stats[key].update(part)
                    except ValueError:  # final batch broadcast error
                        pass

            perchan_input_stats = {}
            for key, running in channel_stats.items():
                running = ub.peek(channel_stats.values())
                perchan_stats = running.simple(axis=(1, 2))
                perchan_input_stats[key] = {
                    'std': perchan_stats['mean'].round(3),
                    'mean': perchan_stats['std'].round(3),
                }

            input_stats = ub.peek(perchan_input_stats.values())
            cacher.save(input_stats)
    else:
        input_stats = {}

    torch_loaders = {
        tag: dset.make_loader(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_workers=config['workers'],
            shuffle=(tag == 'train'),
            balance=(config['balance'] if tag == 'train' else None),
            pin_memory=True)
        for tag, dset in torch_datasets.items()
    }

    dataset_info = {
        'torch_datasets': torch_datasets,
        'torch_loaders': torch_loaders,
        'input_stats': input_stats
    }
    return dataset_info


def load_cifar(key='cifar10', workdir=None, transform=None):
    """
    key = 'cifar10'
    load_cifar(key, workdir=None)
    """
    import torchvision
    import pickle
    import os
    if workdir is None:
        workdir = ub.ensure_app_cache_dir('netharn')

    if key == 'cifar10':
        DATASET = torchvision.datasets.CIFAR10
        dset = DATASET(root=workdir, download=True, transform=transform)
        meta_fpath = os.path.join(dset.root, dset.base_folder, 'batches.meta')
        meta_dict = pickle.load(open(meta_fpath, 'rb'))
        dset.classes = meta_dict['label_names']
        # For some reason the torchvision objects dont have the label names
        # in the dataset. But the download directory will have them.
        # classes = [
        #     'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        #     'horse', 'ship', 'truck',
        # ]
    elif key == 'cifar100':
        DATASET = torchvision.datasets.CIFAR100
        dset = DATASET(root=workdir, download=True, transform=transform)
        meta_fpath = os.path.join(dset.root, dset.base_folder, 'meta')
        meta_dict = pickle.load(open(meta_fpath, 'rb'))
        dset.classes = meta_dict['fine_label_names']
    return dset


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/style_transfer.py \
            --xpu=0 \
            --train_dataset=shapes1024 \
    """
    harn = setup_harn()
    harn.run()
