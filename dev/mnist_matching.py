import cv2
from os.path import join
import netharn as nh
import numpy as np
import torch
import torchvision
import ubelt as ub
from torch import nn
from sklearn import metrics
import kwimage
import kwarray


class MNISTEmbeddingNet(nh.layers.Module):
    """
    References:
        https://github.com/adambielski/siamese-triplet/blob/master/networks.py

    Example:
        >>> input_shape = (None, 1, 28, 28)
        >>> self = MNISTEmbeddingNet(input_shape)
        >>> print('self = {!r}'.format(self))
        >>> print('flat_shape = {!r}'.format(self._flat_shape))
        >>> output_shape = self.output_shape_for(input_shape)
        >>> print(ub.repr2(output_shape.hidden.shallow(1), nl=-1))
        {
            'convnet': (None, 64, 4, 4),
            'reshape': (None, 1024),
            'fc': (None, 256),
        }
        >>> nh.OutputShapeFor(self)(self.input_shape)
        ...dvecs...(None, 256)...
        >>> input_shape = [4] + list(self.input_shape[1:])
        >>> nh.OutputShapeFor(self)._check_consistency(input_shape)
        >>> inputs = torch.rand(input_shape)
        >>> dvecs = self(inputs)['dvecs']
        >>> pdists = torch.nn.functional.pdist(dvecs, p=2)
        >>> pos_dist = pdists[0::2]
        >>> neg_dist = pdists[1::2]
        >>> margin = 1
        >>> x = pos_dist - neg_dist + margin
        >>> loss = torch.nn.functional.softplus(x).mean()
        >>> loss.backward()
    """
    def __init__(self, input_shape=None, desc_size=256):
        super(MNISTEmbeddingNet, self).__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[1]
        self.out_channels = desc_size

        self.convnet = nh.layers.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                            nn.MaxPool2d(2, stride=2),
                                            nn.Conv2d(32, 64, 5), nn.PReLU(),
                                            nn.MaxPool2d(2, stride=2))
        self._conv_output_shape = self.convnet.output_shape_for(self.input_shape)
        self._num_flat = np.prod(self._conv_output_shape[1:])
        self.reshape = nh.layers.Reshape(-1, self._num_flat)
        self._flat_shape = self.reshape.output_shape_for(self._conv_output_shape)
        self.fc = nh.layers.Sequential(nn.Linear(self._num_flat, 256),
                                       nn.PReLU(),
                                       nn.Linear(256, 256),
                                       nn.PReLU(),
                                       nn.Linear(256, desc_size))

    def forward(self, inputs):
        conv_out = self.convnet(inputs)
        flat_conv = self.reshape(conv_out)
        dvecs = self.fc(flat_conv)
        outputs = {'dvecs': dvecs}
        return outputs

    # def forward(self, inputs):
    #     class _ForwardOutput(object):
    #         def coerce(self, x, hidden=None):
    #             return x
    #     _Hidden = ub.odict
    #     _OutputFor = ub.identity
    #     _Output = _ForwardOutput
    #     return self._output_for(inputs, _Hidden, _OutputFor, _Output)

    def _analytic_forward(self, inputs, _Hidden, _OutputFor, _Output):
        hidden = _Hidden()
        hidden['convnet'] = output = _OutputFor(self.convnet)(inputs)
        hidden['reshape'] = output = _OutputFor(self.reshape)(output)
        hidden['fc'] = output = _OutputFor(self.fc)(output)
        return _Output.coerce({'dvecs': output}, hidden)

    def output_shape_for(self, input_shape):
        inputs = input_shape
        _OutputFor = nh.OutputShapeFor
        _Hidden = nh.HiddenShapes
        _Output = nh.OutputShape
        return self._output_for(inputs, _Hidden, _OutputFor, _Output)


class MNIST_MatchingHarness(nh.FitHarn):
    """
    Define how to process a batch, compute loss, and evaluate validation
    metrics.

    Example:
        >>> harn = setup_harn().initialize()
        >>> batch = harn._demo_batch(0, 'train')
        >>> batch = harn._demo_batch(0, 'vali')
    """

    def after_initialize(harn, **kw):
        """ custom netharn callback """
        harn._has_preselected = False
        harn.POS_LABEL = 1
        harn.NEG_LABEL = 0
        harn.confusion_vectors = kwarray.DataFrameLight(
            columns=['y_true', 'y_dist']
        )

    def before_epochs(harn):
        """ custom netharn callback """
        verbose = 0
        for tag, dset in harn.datasets.items():
            if isinstance(dset, torch.utils.data.Subset):
                dset = dset.dataset

            if not hasattr(dset, 'preselect'):
                continue

            # Presample negatives before each epoch.
            if tag == 'train':
                if not harn._has_preselected:
                    harn.log('Presample {} dataset'.format(tag))
                # Randomly resample training negatives every epoch
                dset.preselect(verbose=verbose)
            else:
                if not harn._has_preselected:
                    harn.log('Presample {} dataset'.format(tag))
                    dset.preselect(verbose=verbose)
        harn._has_preselected = True

    def prepare_batch(harn, raw_batch):
        """
        One - prepare the batch

        Args:
            raw_batch (object): raw collated items returned by the loader

        Returns:
            Dict: a standardized variant of the raw batch on the XPU

        Example:
            >>> harn = setup_harn().initialize()
            >>> raw_batch = harn._demo_batch(raw=True, tag='train')
            >>> batch = harn.prepare_batch(raw_batch)
        """
        image, label = raw_batch
        batch = {
            'chip': image,
            'nx': label,
        }
        batch = harn.xpu.move(batch)
        batch['cpu_nx'] = label
        batch['cpu_chips'] = image
        return batch

    def run_batch(harn, batch):
        """
        Two - run the batch

        Args:
            batch (object):
                XPU-mounted, standarized, collated, and prepared items

        Returns:
            Tuple: outputs, loss

        SeeAlso:
            ~/code/netharn/netharn/criterions/triplet.py

        Example:
            >>> harn = setup_harn().initialize()
            >>> raw_batch = harn._demo_batch(raw=1)
            >>> batch = harn.prepare_batch(raw_batch)
            >>> harn, outputs = harn.run_batch(batch)
        """
        inputs = batch['chip']
        outputs = harn.model(inputs)

        dvecs = outputs['dvecs']

        try:
            # Takes roughly 30% of the time for batch sizes of 128
            labels = batch['cpu_nx']
            if harn.current_tag == 'train':
                info = harn.criterion.mine_negatives(
                    dvecs, labels,
                    mode='hardest',
                    # mode='moderate'
                )
            else:
                info = harn.criterion.mine_negatives(dvecs, labels,
                                                     mode='consistent')
            pos_dists = info['pos_dists']
            neg_dists = info['neg_dists']
            triples = info['triples']
        except RuntimeError:
            raise nh.exceptions.SkipBatch

        _loss_parts = {}
        _loss_parts['triplet'] = harn.criterion(pos_dists, neg_dists)

        if True and harn.epoch > 0:
            # Experimental extra loss term:
            y_dist = torch.cat([pos_dists, neg_dists], dim=0)
            margin = harn.criterion.margin
            y_probs = (-(y_dist - margin)).sigmoid()

            # Use MSE to encourage the average batch-hard-case prob to be 0.5
            # Note this relies heavilly on the assumption that there are an
            # equal number of pos / neg cases (which is true in triplet loss)
            prob_mean = y_probs.mean()
            prob_std = y_probs.std()

            target_mean = torch.FloatTensor([0.5]).to(prob_mean.device)
            target_std = torch.FloatTensor([0.1]).to(prob_mean.device)

            _loss_parts['pmean'] = 0.1 * torch.nn.functional.mse_loss(prob_mean, target_mean)
            _loss_parts['pstd'] = 0.1 * torch.nn.functional.mse_loss(prob_std.clamp(0, 0.1), target_std)

            # Encourage descriptor vecstors to have a small squared-gradient-magnitude (ref girshik)
            _loss_parts['sgm'] = 0.0003 * (dvecs ** 2).sum(dim=0).mean()

        loss = sum(_loss_parts.values())

        if 0:
            all_grads = harn._check_gradients()
            print(ub.map_vals(torch.norm, all_grads))

        harn._loss_parts = _loss_parts

        outputs['triples'] = triples
        outputs['chip'] = batch['chip']
        outputs['nx'] = batch['nx']
        outputs['distAP'] = pos_dists
        outputs['distAN'] = neg_dists
        return outputs, loss

    def on_batch(harn, batch, outputs, loss):
        """
        custom netharn callback

        Example:
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='vali')
            >>> outputs, loss = harn.run_batch(batch)
            >>> decoded = harn._decode(outputs)
            >>> stacked = harn._draw_batch(decoded, limit=42)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked)
            >>> kwplot.show_if_requested()
        """
        batch_metrics = ub.odict()
        for key, value in harn._loss_parts.items():
            if value is not None and torch.is_tensor(value):
                batch_metrics[key + '_loss'] = float(
                    value.data.cpu().numpy().item())

        bx = harn.bxs[harn.current_tag]

        if bx < 8:
            decoded = harn._decode(outputs)
            stacked = harn._draw_batch(decoded)
            dpath = ub.ensuredir((harn.train_dpath, 'monitor', harn.current_tag))
            fpath = join(dpath, 'batch_{}_epoch_{}.jpg'.format(bx, harn.epoch))
            kwimage.imwrite(fpath, stacked)

        # Record metrics for epoch scores
        n = len(outputs['distAP'])

        harn.confusion_vectors._data['y_true'].extend([harn.POS_LABEL] * n)
        harn.confusion_vectors._data['y_dist'].extend(outputs['distAP'].data.cpu().numpy().tolist())

        harn.confusion_vectors._data['y_true'].extend([harn.NEG_LABEL] * n)
        harn.confusion_vectors._data['y_dist'].extend(outputs['distAN'].data.cpu().numpy().tolist())
        return batch_metrics

    def on_epoch(harn):
        """
        custom netharn callback

        Example:
            >>> harn = setup_harn().initialize()
            >>> harn.before_epochs()
            >>> epoch_metrics = harn._demo_epoch(tag='vali', max_iter=10)
            >>> print('epoch_metrics = {}'.format(ub.repr2(epoch_metrics, precision=4)))
            >>> print('harn.confusion_vectors = {!r}'.format(harn.confusion_vectors._pandas()))
        """
        margin = harn.hyper.criterion_params['margin']
        assert margin == harn.criterion.margin
        epoch_metrics = {}

        if len(harn.confusion_vectors):
            margin = harn.criterion.margin
            y_true = np.array(harn.confusion_vectors['y_true'], dtype=np.uint8)
            y_dist = np.array(harn.confusion_vectors['y_dist'], dtype=np.float32)

            # Transform distance into a probability-like space
            y_probs = torch.Tensor(-(y_dist - margin)).sigmoid().numpy()

            pos_flags = np.where(y_true == harn.POS_LABEL)[0]
            neg_flags = np.where(y_true == harn.NEG_LABEL)[0]

            pos_dists = y_dist[pos_flags]
            neg_dists = y_dist[neg_flags]

            pos_probs = y_probs[pos_flags]
            neg_probs = y_probs[neg_flags]

            y_pred = (pos_probs < neg_probs).astype(np.uint8)

            # How should be choose a threshold here?
            thresh = median_prob = np.median(y_probs)
            # thresh = 0.5
            y_pred = (y_probs > thresh).astype(np.uint8)

            accuracy = (y_true == y_pred).astype(np.uint8).mean()
            brier = ((y_probs - y_true) ** 2).mean()
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'invalid value')
                mcc = metrics.matthews_corrcoef(y_true, y_pred)

            epoch_metrics = {
                'mcc': mcc,
                'brier': brier,
                'accuracy': accuracy,
                'dist_pos': np.nanmean(pos_dists),
                'dist_neg': np.nanmean(neg_dists),

                'prob_pos': np.nanmean(pos_probs),
                'prob_neg': np.nanmean(neg_probs),
                'prob_median': median_prob,
            }

        # Clear scores for the next epoch
        harn.confusion_vectors.clear()
        return epoch_metrics

    def _decode(harn, outputs):
        """
        Convert raw network outputs to something interpretable
        """
        decoded = {}

        triple_idxs = outputs['triples'].T
        A, P, N = triple_idxs

        chips_ = outputs['chip'].data.cpu().numpy()
        nxs_ = outputs['nx'].data.cpu().numpy()

        decoded['triple_idxs'] = triple_idxs
        decoded['triple_imgs'] = [chips_[A], chips_[P], chips_[N]]
        decoded['triple_nxs'] = [nxs_[A], nxs_[P], nxs_[N]]

        decoded['distAP'] = outputs['distAP'].data.cpu().numpy()
        decoded['distAN'] = outputs['distAN'].data.cpu().numpy()
        return decoded

    def _draw_batch(harn, decoded, limit=12):
        """
        Example:
            >>> harn = setup_harn().initialize()
            >>> batch = harn._demo_batch(0, tag='vali')
            >>> outputs, loss = harn.run_batch(batch)
            >>> decoded = harn._decode(outputs)
            >>> stacked = harn._draw_batch(decoded)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(stacked, colorspace='rgb', doclf=True)
            >>> kwplot.show_if_requested()
        """
        tostack = []
        fontkw = {
            'fontScale': 1.0,
            'thickness': 2
        }
        n = min(limit, len(decoded['triple_idxs'][0]))
        dsize = (254, 254)
        for i in range(n):
            ims = [g[i].transpose(1, 2, 0) for g in decoded['triple_imgs']]
            ims = [cv2.resize(g, dsize) for g in ims]
            ims = [kwimage.atleast_3channels(g) for g in ims]
            triple_nxs = [n[i] for n in decoded['triple_nxs']]

            text = 'dAP={:.3g} -- dAN={:.3g} -- {}'.format(
                decoded['distAP'][i],
                decoded['distAN'][i],
                str(triple_nxs),
            )
            color = (
                'dodgerblue' if decoded['distAP'][i] < decoded['distAN'][i]
                else 'orangered')

            img = kwimage.stack_images(
                ims, overlap=-2, axis=1,
                bg_value=(10 / 255, 40 / 255, 30 / 255)
            )
            img = (img * 255).astype(np.uint8)
            img = kwimage.draw_text_on_image(img, text,
                                             org=(2, img.shape[0] - 2),
                                             color=color, **fontkw)
            tostack.append(img)
        stacked = kwimage.stack_images_grid(tostack, overlap=-10,
                                            bg_value=(30, 10, 40),
                                            axis=1, chunksize=3)
        return stacked


def setup_datasets(workdir=None):
    if workdir is None:
        workdir = ub.expandpath('~/data/mnist/')

    # Define your dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    learn_dset = nh.data.MNIST(workdir, transform=transform, train=True,
                               download=True)

    test_dset = nh.data.MNIST(workdir, transform=transform, train=False,
                              download=True)

    # split the learning dataset into training and validation
    # take a subset of data
    factor = .15
    n_vali = int(len(learn_dset) * factor)
    learn_idx = np.arange(len(learn_dset))

    rng = np.random.RandomState(0)
    rng.shuffle(learn_idx)

    reduction = int(ub.argval('--reduction', default=1))
    vali_idx  = torch.LongTensor(learn_idx[:n_vali][::reduction])
    train_idx = torch.LongTensor(learn_idx[n_vali:][::reduction])

    train_dset = torch.utils.data.Subset(learn_dset, train_idx)
    vali_dset = torch.utils.data.Subset(learn_dset, vali_idx)

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }
    if not ub.argflag('--test'):
        del datasets['test']
    for tag, dset in datasets.items():
        # Construct the PCCs (positive connected components)
        # These are groups of item indices which are positive matches
        if isinstance(dset, torch.utils.data.Subset):
            labels = dset.dataset.train_labels[dset.indices]
        else:
            labels = dset.labels
        unique_labels, groupxs = kwarray.group_indices(labels.numpy())
        dset.pccs = [xs.tolist() for xs in groupxs]

    # Give the training dataset an input_id
    datasets['train'].input_id = 'mnist_' + ub.hash_data(train_idx.numpy())[0:8]
    return datasets, workdir


def setup_harn(**kwargs):
    """
    Args:
        nice (str): Custom tag for this run
        workdir (PathLike): path to dump all the intermedate results

        batch_size (int):
            Base batch size. Number of examples in GPU at any time.
        p (int): num individuals per batch
        k (int): num annots-per-individual per batch

        bstep (int): Multiply by batch_size to simulate a larger batches.
        lr (float|str): Base learning rate
        decay (float): Weight decay (L2 regularization)

        workers (int): Number of parallel data loader workers
        xpu (str): Device to train on. Can be either `'cpu'`, `'gpu'`, a number
            indicating a GPU (e.g. `0`), or a list of numbers (e.g. `[0,1,2]`)
            indicating multiple GPUs

        norm_desc (bool): if True normalizes the descriptors
        pretrained (PathLike): path to a compatible pretrained model

        margin (float): margin for loss criterion
        soft (bool): use soft margin
    """
    import ast
    def trycast(x, type):
        try:
            return type(x)
        except Exception:
            return x

    config = {}
    config['init'] = kwargs.get('init', 'kaiming_normal')
    config['pretrained'] = config.get('pretrained',
                                      ub.argval('--pretrained', default=None))
    config['margin'] = kwargs.get('margin', 3.0)
    config['soft'] = kwargs.get('soft', False)
    config['xpu'] = kwargs.get('xpu', 'argv')
    config['nice'] = kwargs.get('nice', 'untitled')
    config['workdir'] = kwargs.get('workdir', None)
    config['workers'] = int(kwargs.get('workers', 1))
    config['bstep'] = int(kwargs.get('bstep', 1))
    config['optim'] = kwargs.get('optim', 'sgd')
    config['scheduler'] = kwargs.get('scheduler', 'onecycle70')
    config['lr'] = trycast(kwargs.get('lr', 0.0001), float)
    config['decay'] = float(kwargs.get('decay', 1e-5))

    config['max_epoch'] = int(kwargs.get('max_epoch', 100))

    config['num_batches'] = trycast(kwargs.get('num_batches', 1000), int)
    config['batch_size'] = int(kwargs.get('batch_size', 128))
    config['p'] = float(kwargs.get('p', 10))
    config['k'] = float(kwargs.get('k', 25))

    config['arch'] = kwargs.get('arch', 'resnet')
    config['hidden'] = trycast(kwargs.get('hidden', [128]), ast.literal_eval)
    config['desc_size'] = kwargs.get('desc_size', 256)

    config['norm_desc'] = kwargs.get('norm_desc', False)
    config['dim'] = 28

    xpu = nh.XPU.coerce(config['xpu'])
    nh.configure_hacks(config)
    datasets, workdir = setup_datasets()

    loaders = {
        tag: torch.utils.data.DataLoader(
            dset,
            batch_sampler=nh.data.batch_samplers.MatchingSamplerPK(
                dset.pccs,
                shuffle=(tag == 'train'),
                batch_size=config['batch_size'],
                num_batches=config['num_batches'],
                k=config['k'],
                p=config['p'],
            ),
            num_workers=config['workers'],
        )
        for tag, dset in datasets.items()
    }

    if config['arch'] == 'simple':
        model_ = (MNISTEmbeddingNet, {
            'input_shape': (1, 1, config['dim'], config['dim']),
            'desc_size': config['desc_size'],
        })
    elif config['arch'] == 'resnet':
        model_ = (nh.models.DescriptorNetwork, {
            'input_shape': (1, 1, config['dim'], config['dim']),
            'norm_desc': config['norm_desc'],
            'hidden_channels': config['hidden'],
            'desc_size': config['desc_size'],
        })
    else:
        raise KeyError(config['arch'])

    if config['scheduler'] == 'steplr':
        from torch.optim import lr_scheduler
        scheduler_ = (lr_scheduler.StepLR,
                      dict(step_size=8, gamma=0.1, last_epoch=-1))
    else:
        scheduler_ = nh.Scheduler.coerce(config, scheduler='onecycle70')

    # Here is the FitHarn magic.
    # They nh.HyperParams object keeps track of and helps log all declarative
    # info related to training a model.
    hyper = nh.hyperparams.HyperParams(
        nice=config['nice'],
        xpu=xpu,
        workdir=workdir,
        datasets=datasets,
        loaders=loaders,
        model=model_,
        initializer=nh.Initializer.coerce(config),
        optimizer=nh.Optimizer.coerce(config),
        scheduler=scheduler_,
        criterion=(nh.criterions.TripletLoss, {
            'margin': config['margin'],
            'soft': config['soft'],
        }),
        monitor=nh.Monitor.coerce(
            config,
            minimize=['loss', 'pos_dist', 'brier'],
            maximize=['accuracy', 'neg_dist', 'mcc'],
            patience=100,
            max_epoch=config['max_epoch'],
            smoothing=0.4,
        ),
        other={
            'batch_size': config['batch_size'],
            'num_batches': config['num_batches'],
        }
    )

    harn = MNIST_MatchingHarness(hyper=hyper)
    harn.preferences
    return harn


def main():
    print('PARSE')
    import xinspect
    parser = xinspect.auto_argparse(setup_harn)

    parser.add_argument(
        '--lrtest', action='store_true',
        help='Run Leslie Smith\'s LR range test')
    parser.add_argument(
        '--interact', action='store_true',
        help='Interact with the range test')
    args, unknown = parser.parse_known_args()
    ns = args.__dict__.copy()

    args.interact |= args.lr == 'interact'
    args.lrtest |= (args.lr == 'test' or args.interact)

    if args.lrtest or args.interact:
        # TODO:
        # - [ ] tweak setup_harn so running the lr-range-test isn't awkward
        from netharn.prefit.lr_tests import lr_range_test
        ns['lr'] = 1e-99

        if args.interact:
            import kwplot
            kwplot.autompl()
            import matplotlib.pyplot as plt

        harn = setup_harn(**ns)
        harn.initialize()
        # TODO: We could cache the result based on the netharn
        # hyperparameters. This would let us integrate with the
        # default fit harness.
        result = lr_range_test(harn)
        print('result.recommended_lr = {!r}'.format(result.recommended_lr))

        if args.interact:
            result.draw()
            plt.show()

        # Seed value with test result
        ns['lr'] = result.recommended_lr
        harn = setup_harn(**ns).initialize()
    else:
        harn = setup_harn(**ns)

    harn.initialize()
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:
        # Show help
        python ~/code/netharn/examples/mnist_matching.py --help

        # Run with default values
        python ~/code/netharn/examples/mnist_matching.py --verbose

        # Try new hyperparam values and run the LR-range-test
        python ~/code/netharn/examples/mnist_matching.py \
                --batch_size=128 \
                --num_batches=1000 \
                --scheduler=onecycle70 \
                --optim=sgd \
                --soft=False \
                --margin=1 \
                --lr=0.0001 \
                --arch=simple \
                --nice=test_bugfix --verbose

        # Train with the simple architecture (to show the harness works)
        python ~/code/netharn/examples/mnist_matching.py \
                --arch=simple --optim=adam \
                --decay=1e-4 \
                --lr=1e-3 --scheduler=steplr --max_epoch=20 \
                --batch_size=250  --num_batches=240 \
                --margin=1 --soft=False \
                --nice=demo_matching_harness_v6

        # Train with netharn's Resnet DescriptorNetwork (to show the model works)
        python ~/code/netharn/examples/mnist_matching.py \
                --arch=resnet --optim=sgd \
                --lr=0.00007 --scheduler=onecycle70 --max_epoch=100 \
                --batch_size=250  --num_batches=250 \
                --margin=1 --soft=True \
                --nice=demo_resnet_descriptor_network_v1
    """
    main()
