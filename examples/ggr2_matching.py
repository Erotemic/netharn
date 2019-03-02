from os.path import join
import os
import torch
import netharn as nh
import graphid
import random
import ubelt as ub
import torchvision


class MatchingNetworkLP(torch.nn.Module):
    """
    Siamese pairwise distance

    Example:
        >>> self = MatchingNetworkLP()
    """

    def __init__(self, p=2, branch=None, input_shape=(1, 3, 416, 416)):
        super(MatchingNetworkLP, self).__init__()
        if branch is None:
            self.branch = torchvision.models.resnet50(pretrained=True)
        else:
            self.branch = branch
        assert isinstance(self.branch, torchvision.models.ResNet)
        prepool_shape = self.resnet_prepool_output_shape(input_shape)
        # replace the last layer of resnet with a linear embedding to learn the
        # LP distance between pairs of images.
        # Also need to replace the pooling layer in case the input has a
        # different size.
        self.prepool_shape = prepool_shape
        pool_channels = prepool_shape[1]
        pool_kernel = prepool_shape[2:4]
        self.branch.avgpool = torch.nn.AvgPool2d(pool_kernel, stride=1)
        self.branch.fc = torch.nn.Linear(pool_channels, 500)

        self.pdist = torch.nn.PairwiseDistance(p=p)

    def resnet_prepool_output_shape(self, input_shape):
        """
        self = MatchingNetworkLP(input_shape=input_shape)
        input_shape = (1, 3, 224, 224)
        self.resnet_prepool_output_shape(input_shape)
        self = MatchingNetworkLP(input_shape=input_shape)
        input_shape = (1, 3, 416, 416)
        self.resnet_prepool_output_shape(input_shape)
        """
        # Figure out how big the output will be and redo the average pool layer
        # to account for it
        branch = self.branch
        shape = input_shape
        shape = nh.OutputShapeFor(branch.conv1)(shape)
        shape = nh.OutputShapeFor(branch.bn1)(shape)
        shape = nh.OutputShapeFor(branch.relu)(shape)
        shape = nh.OutputShapeFor(branch.maxpool)(shape)

        shape = nh.OutputShapeFor(branch.layer1)(shape)
        shape = nh.OutputShapeFor(branch.layer2)(shape)
        shape = nh.OutputShapeFor(branch.layer3)(shape)
        shape = nh.OutputShapeFor(branch.layer4)(shape)
        prepool_shape = shape
        return prepool_shape

    def forward(self, input1, input2):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.

        Example:
            >>> input1 = nh.XPU(None).variable(torch.rand(4, 3, 224, 224))
            >>> input2 = nh.XPU(None).variable(torch.rand(4, 3, 224, 224))
            >>> self = MatchingNetworkLP(input_shape=input2.shape[1:])
            >>> output = self(input1, input2)

        Ignore:
            >>> input1 = nh.XPU(None).variable(torch.rand(1, 3, 416, 416))
            >>> input2 = nh.XPU(None).variable(torch.rand(1, 3, 416, 416))
            >>> input_shape1 = input1.shape
            >>> self = MatchingNetworkLP(input_shape=input2.shape[1:])
            >>> self(input1, input2)
        """
        output1 = self.branch(input1)
        output2 = self.branch(input2)
        output = self.pdist(output1, output2)
        return output

    def output_shape_for(self, input_shape1, input_shape2):
        shape1 = nh.OutputShapeFor(self.branch)(input_shape1)
        shape2 = nh.OutputShapeFor(self.branch)(input_shape2)
        assert shape1 == shape2
        output_shape = (shape1[0], 1)
        return output_shape


class MatchHarness(nh.FitHarn):
    """
    Define how to process a batch, compute loss, and evaluate validation
    metrics.
    """

    def __init__(harn, *args, **kw):
        super(MatchHarness).__init__(*args, **kw)
        harn.batch_confusions = []

    def prepare_batch(harn, raw_batch):
        """
        ensure batch is in a standardized structure
        """
        img1, img2, label = raw_batch
        inputs = harn.xpu.variables(img1, img2)
        label = harn.xpu.variable(label)
        batch = (inputs, label)
        return batch

    def run_batch(harn, batch):
        """
        Connect data -> network -> loss

        Args:
            batch: item returned by the loader
        """
        inputs, label = batch
        output = harn.model(*inputs)
        loss = harn.criterion(output, label).sum()
        return output, loss

    def on_batch(harn, batch, output, loss):
        """ custom callback """
        label = batch[-1]
        l2_dist_tensor = torch.squeeze(output.data.cpu())
        label_tensor = torch.squeeze(label.data.cpu())

        # Distance
        POS_LABEL = 1  # NOQA
        NEG_LABEL = 0  # NOQA
        # is_pos = (label_tensor == POS_LABEL)

        # pos_dists = l2_dist_tensor[is_pos]
        # neg_dists = l2_dist_tensor[~is_pos]

        # Average positive / negative distances
        # pos_dist = pos_dists.sum() / max(1, len(pos_dists))
        # neg_dist = neg_dists.sum() / max(1, len(neg_dists))

        # accuracy
        # margin = harn.hyper.criterion_params['margin']
        # pred_pos_flags = (l2_dist_tensor <= margin).long()

        # pred = pred_pos_flags
        # n_correct = (pred == label_tensor).sum()
        # fraction_correct = n_correct / len(label_tensor)

        # Record metrics for epoch scores
        y_true = label_tensor.cpu().numpy()
        y_dist = l2_dist_tensor.cpu().numpy()
        # y_pred = pred.cpu().numpy()
        harn.batch_confusions.append((y_true, y_dist))

        # metrics = {
        #     'accuracy': float(fraction_correct),
        #     'pos_dist': float(pos_dist),
        #     'neg_dist': float(neg_dist),
        # }
        # return metrics

    def on_epoch(harn):
        """ custom callback """
        from sklearn import metrics
        margin = harn.hyper.criterion_params['margin']

        y_true = np.hstack([p[0] for p in harn.batch_confusions])
        y_dist = np.hstack([p[1] for p in harn.batch_confusions])

        y_pred = (y_dist <= margin).astype(y_true.dtype)

        POS_LABEL = 1  # NOQA
        NEG_LABEL = 0  # NOQA
        pos_dist = np.nanmean(y_dist[y_true == POS_LABEL])
        neg_dist = np.nanmean(y_dist[y_true == NEG_LABEL])

        # Transform distance into a probability-like space
        y_probs = torch.sigmoid(torch.Tensor(-(y_dist - margin))).numpy()

        brier = y_probs - y_true

        accuracy = (y_true == y_pred).mean()
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        brier = ((y_probs - y_true) ** 2).mean()

        epoch_metrics = {
            'mcc': mcc,
            'brier': brier,
            'accuracy': accuracy,
            'pos_dist': pos_dist,
            'neg_dist': neg_dist,
        }

        # Clear scores for next epoch
        harn.batch_confusions.clear()
        return epoch_metrics


class GraphTripletDataset(torch.utils.data.Dataset):
    """
    A pairwise torch dataset
    """

    def __init__(self, infr, coco_dset):
        self.infr = infr
        self.coco_dset = coco_dset

        self._triple_pool = []

        self.dim = 500
        self.letterbox = nh.data.transforms.Resize(
            target_size=(self.dim, self.dim), mode='letterbox')

        self._simple_sample()

    def _simple_sample(self):
        # Simple strategy for creating examples
        infr = self.infr

        self._triple_pool = []

        for aid1, aid2 in self.infr.pos_graph.edges():
            cc = infr.pos_graph.connected_to(aid1)
            neg_edges = graphid.util.edges_outgoing(self.infr.neg_graph, [aid1, aid2])
            neg_aids = []
            for edge in neg_edges:
                neg_aids.append(set(edge) - {aid1, aid2})
            neg_aids = list(ub.flatten(neg_aids))

            if neg_aids:
                aid3 = random.choice(neg_aids)
            else:
                cc2 = next(infr.find_non_neg_redun_pccs(cc=cc, k=1))[1]
                aid3 = random.choice(list(cc2))

            # Check that we actually have the data
            if aid1 in self.coco_dset.anns and aid2 in self.coco_dset.anns and aid3 in self.coco_dset.anns:
                self._triple_pool.append((aid1, aid2, aid3))

    def __len__(self):
        return len(self._triple_pool)

    def __getitem__(self, index):
        aid1, aid2, aid3 = self._triple_pool[index]

        annot1 = self.coco_dset.anns[aid1]
        annot2 = self.coco_dset.anns[aid2]
        annot3 = self.coco_dset.anns[aid3]

        gid1 = annot1['image_id']
        gid2 = annot2['image_id']
        gid3 = annot3['image_id']

        gpath1 = join(self.coco_dset.img_root, self.coco_dset.imgs[gid1]['file_name'])
        gpath2 = join(self.coco_dset.img_root, self.coco_dset.imgs[gid2]['file_name'])
        gpath3 = join(self.coco_dset.img_root, self.coco_dset.imgs[gid3]['file_name'])

        img1 = nh.util.imread(gpath1)
        img2 = nh.util.imread(gpath2)
        img3 = nh.util.imread(gpath3)

        def _sl(bbox):
            x, y, w, h = bbox
            return (slice(y, y + h), slice(x, x + w))

        chip1 = img1[_sl(annot1['bbox'])]
        chip2 = img2[_sl(annot2['bbox'])]
        chip3 = img3[_sl(annot3['bbox'])]

        chip1, chip2, chip3 = self.letterbox.augment_images([chip1, chip2, chip3])

        totensor = torchvision.transforms.ToTensor()
        chip1 = totensor(chip1)
        chip2 = totensor(chip2)
        chip3 = totensor(chip3)

        item = {
            'data': (chip1, chip2, chip3),
            'label': {
                'aids': [aid1, aid2, aid3],
            }
        }
        # x = nh.util.stack_images([chip1, chip2, chip3])[0]
        # chip1.shape
        # chip2.shape
        # chip3.shape
        return item


def prepare_datasets(tags=['train', 'test', 'vali']):
    """
    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/netharn/examples'))
        >>> from ggr2_matching import
        >>> datasets = prepare_datasets(['vali'])
        >>> dset = datasets['train']
        >>> item = dset[0]
        >>> data = item['data']
        >>> img1, img2, img3 = [d.cpu().numpy().transpose(1, 2, 0) for d in data]
        >>> stacked = nh.util.imutil.stack_multiple_images([img1, img2, img3], axis=1)
        >>> # xdoc: +REQUIRES(--show)
        >>> nh.util.autompl()
        >>> nh.util.imshow(stacked)
    """
    import ubelt as ub
    import json
    base = ub.pathlike('~/remote/192.168.222.4/data/ggr2-coco')
    print('Begin prepare_datasets')

    annot_fpaths = {
        'train': os.path.join(base, 'annotations/instances_train2018.json'),
        'test': os.path.join(base, 'annotations/instances_test2018.json'),
        'vali': os.path.join(base, 'annotations/instances_val2018.json'),
    }

    image_dpaths = {
        'train': os.path.join(base, 'images/train2018'),
        'test': os.path.join(base, 'images/test2018'),
        'vali': os.path.join(base, 'images/val2018'),
    }

    datasets = {}

    for tag in tags:
        annot_fpath = annot_fpaths[tag]
        image_dpath = image_dpaths[tag]

        data = json.load(open(annot_fpath, 'r'))
        coco_dset = nh.data.CocoDataset(data=data, tag=tag, img_root=image_dpath)

        graph = graphid.api.GraphID()
        graph.add_annots_from(coco_dset.annots().aids)

        graph.infr.params['inference.enabled'] = False

        for aid1 in ub.ProgIter(coco_dset.annots().aids, desc='construct graph'):
            annot = coco_dset.anns[aid1]
            for review in annot['review_ids']:
                aid2, decision = review
                edge = (aid1, aid2)
                if decision == 'positive':
                    graph.add_edge(edge, graphid.core.POSTV)
                elif decision == 'negative':
                    graph.add_edge(edge, graphid.core.NEGTV)
                elif decision == 'incomparable':
                    graph.add_edge(edge, graphid.core.INCMP)
                else:
                    raise KeyError(decision)

        graph.infr.params['inference.enabled'] = True
        graph.infr.apply_nondynamic_update()

        infr = graph.infr
        print('status = {}' + ub.repr2(infr.status(True)))
        # pccs = list(infr.positive_components())

        torch_dset = GraphTripletDataset(infr, coco_dset)
        datasets[tag] = torch_dset
    return datasets


def setup_harness(**kwargs):
    """
    CommandLine:
        python ~/code/netharn/netharn/examples/siam_ibeis.py setup_harness

    Example:
        >>> harn = setup_harness(dbname='PZ_MTEST')
        >>> harn.initialize()
    """
    nice = kwargs.get('nice', 'untitled')
    bsize = int(kwargs.get('bsize', 6))
    bstep = int(kwargs.get('bstep', 1))
    workers = int(kwargs.get('workers', 0))
    decay = float(kwargs.get('decay', 0.0005))
    lr = float(kwargs.get('lr', 0.001))
    dim = int(kwargs.get('dim', 416))
    xpu = kwargs.get('xpu', 'cpu')
    workdir = kwargs.get('workdir', None)
    dbname = kwargs.get('dbname', 'PZ_MTEST')

    datasets = prepare_datasets(dbname, dim=dim)
    if workdir is None:
        workdir = ub.pathlike(os.path.join('~/work/siam-ibeis3', dbname))
    ub.ensuredir(workdir)

    for k, v in datasets.items():
        print('* len({}) = {}'.format(k, len(v)))

    if workers > 0:
        import cv2
        cv2.setNumThreads(0)

    loaders = {
        key:  torch.utils.data.DataLoader(
            dset, batch_size=bsize, num_workers=workers,
            shuffle=(key == 'train'), pin_memory=True)
        for key, dset in datasets.items()
    }

    xpu = nh.XPU.cast(xpu)

    if False:
        criterion_ = (nh.criterions.ContrastiveLoss, {
            'margin': 4,
            'weight': None,
        })
    else:
        criterion_ = (torch.nn.modules.TripletMarginLoss, {
            'margin': 1,
            'p': 2,
        })

    if True:
        model_ = (MatchingNetworkLP, {
            'p': 2,
            'input_shape': (1, 3, dim, dim),
        })

    hyper = nh.HyperParams(**{
        'nice': nice,
        'workdir': workdir,
        'datasets': datasets,
        'loaders': loaders,

        'xpu': xpu,

        'model': model_,

        'criterion': criterion_,

        'optimizer': (torch.optim.SGD, {
            'lr': lr,
            'weight_decay': decay,
            'momentum': 0.9,
            'nesterov': True,
        }),

        'initializer': (nh.initializers.NoOp, {}),

        'scheduler': (nh.schedulers.Exponential, {
            'gamma': 0.99,
            'stepsize': 2,
        }),
        # 'scheduler': (nh.schedulers.ListedLR, {
        #     'points': {
        #         1:   lr * 1.0,
        #         19:  lr * 1.1,
        #         20:  lr * 0.1,
        #     },
        #     'interpolate': True
        # }),

        'monitor': (nh.Monitor, {
            'minimize': ['loss', 'pos_dist', 'brier'],
            'maximize': ['accuracy', 'neg_dist', 'mcc'],
            'patience': 40,
            'max_epoch': 40,
        }),

        'augment': datasets['train'].augmenter,

        'dynamics': {
            # Controls how many batches to process before taking a step in the
            # gradient direction. Effectively simulates a batch_size that is
            # `bstep` times bigger.
            'batch_step': bstep,
        },

        'other': {
            'n_classes': 2,
        },
    })
    harn = MatchHarness(hyper=hyper)
    harn.config['prog_backend'] = 'progiter'
    harn.intervals['log_iter_train'] = 1
    harn.intervals['log_iter_test'] = None
    harn.intervals['log_iter_vali'] = None

    return harn
