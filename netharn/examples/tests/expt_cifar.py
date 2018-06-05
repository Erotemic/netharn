"""
Experimental code related to cifar
"""
import numpy as np
import ubelt as ub
import torch
# import torchvision
import pandas as pd
from torchvision.datasets import cifar
from netharn import xpu_device
from netharn import monitor
from netharn import initializers
from netharn import hyperparams
from netharn import fit_harness
from netharn.transforms import (ImageCenterScale,)
# from netharn.transforms import (RandomWarpAffine, RandomGamma, RandomBlur,)
import imgaug as ia
import imgaug.augmenters as iaa
from netharn import util
import netharn as nh


class CropTo(iaa.Augmenter):
    def __init__(self, shape,  name=None, deterministic=False, random_state=None):
        super(CropTo, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.shape = shape

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in range(nb_images):
            seed = seeds[i]
            height, width = images[i].shape[0:2]
            top, bot, left, right = self._draw_samples_image(seed, height, width)

            image_cr = images[i][top:bot, left:right]
            image_cr = np.pad(image_cr, ((1, 1), (1, 1)), mode='constant')

            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        nb_images = len(keypoints_on_images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            seed = seeds[i]
            height, width = keypoints_on_image.shape[0:2]
            top, bot, left, right = self._draw_samples_image(seed, height, width)
            shifted = keypoints_on_image.shift(x=-left, y=-top)
            shifted.shape = (
                height - top - bot,
                width - left - right
            ) + shifted.shape[2:]
            result.append(shifted)
        return result

    def _draw_samples_image(self, seed, height, width):
        """
        height = 32
        width = 32
        h, w = shape = (30, 30)
        random_state = np.random
        """
        random_state = ia.new_random_state(seed)
        h, w = self.shape

        assert w <= width, '{} {}'.format(w, width)
        assert h <= height, '{} {}'.format(h, height)
        space_h = height - h
        space_w = width - w

        top = random_state.randint(0, space_h + 1)
        bot = height - (space_h - top)

        left = random_state.randint(0, space_w + 1)
        right = width - (space_w - left)

        sub = [top, bot, left, right]
        return sub

    def get_parameters(self):
        return [self.shape]


class Task(object):
    def __init__(task, labelnames=None, ignore_labelnames=[], alias={}):
        if labelnames is not None:
            task.set_labelnames(labelnames, ignore_labelnames, alias)

    def set_labelnames(task, labelnames, ignore_labelnames=[], alias={}):
        task.labelnames = list(labelnames)
        task.labelname_alias = alias
        task.ignore_labelnames = ignore_labelnames

        # Remove aliased classes
        for k in alias.keys():
            if k in task.labelnames:
                task.labelnames.remove(k)

        # Assign an integer label to each labelname
        task.labelname_to_id = ub.invert_dict(dict(enumerate(task.labelnames)))

        # Map aliased classes to a different label
        for k, v in alias.items():
            task.labelname_to_id[k] = task.labelname_to_id[v]

        task.ignore_labelnames = ignore_labelnames
        task.ignore_labels = np.array(
            list(ub.take(task.labelname_to_id, task.ignore_labelnames)))

        task.labels = np.arange(len(task.labelnames))
        task.relevant_labels = np.setdiff1d(task.labels, task.ignore_labels)


def radial_fourier_mask(img_chw, radius=11, axis=None, clip=None):
    """
    In [1] they use a radius of 11.0 on CIFAR-10.

    Args:
        img_chw (ndarray): assumed to be float 01

    References:
        [1] Jo and Bengio "Measuring the tendency of CNNs to Learn Surface Statistical Regularities" 2017.
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

    CommandLine:
        python examples/cifar.py radial_fourier_mask --show

    Example:
        >>> dset = cifar_training_datasets()['test']
        >>> dset.center_inputs = None
        >>> img_tensor, label = dset[7]
        >>> img_chw = img_tensor.numpy()
        >>> out = radial_fourier_mask(img_chw, radius=11)
        >>> # xdoc: REQUIRES(--show)
        >>> nh.util.qtensure()
        >>> def keepdim(func):
        >>>     def _wrap(im):
        >>>         needs_transpose = (im.shape[0] == 3)
        >>>         if needs_transpose:
        >>>             im = im.transpose(1, 2, 0)
        >>>         out = func(im)
        >>>         if needs_transpose:
        >>>             out = out.transpose(2, 0, 1)
        >>>         return out
        >>>     return _wrap
        >>> @keepdim
        >>> def rgb_to_bgr(im):
        >>>     return util.convert_colorspace(im, src_space='rgb', dst_space='bgr')
        >>> @keepdim
        >>> def bgr_to_lab(im):
        >>>     return util.convert_colorspace(im, src_space='bgr', dst_space='lab')
        >>> @keepdim
        >>> def lab_to_bgr(im):
        >>>     return util.convert_colorspace(im, src_space='lab', dst_space='bgr')
        >>> @keepdim
        >>> def bgr_to_yuv(im):
        >>>     return util.convert_colorspace(im, src_space='bgr', dst_space='yuv')
        >>> @keepdim
        >>> def yuv_to_bgr(im):
        >>>     return util.convert_colorspace(im, src_space='yuv', dst_space='bgr')
        >>> dpath = ub.ensuredir('./fouriertest')
        >>> from matplotlib import pyplot as plt
        >>> for x in ub.ProgIter(range(100)):
        >>>     img_tensor, label = dset[x]
        >>>     img_chw = img_tensor.numpy()
        >>>     bgr_img = rgb_to_bgr(img_chw)
        >>>     nh.util.imshow(bgr_img.transpose(1, 2, 0), fnum=1)
        >>>     pnum_ = nh.util.PlotNums(nRows=4, nCols=5)
        >>>     for r in range(0, 17):
        >>>         imgt = radial_fourier_mask(bgr_img, r, clip=(0, 1))
        >>>         nh.util.imshow(imgt.transpose(1, 2, 0), pnum=pnum_(), fnum=2)
        >>>         plt.gca().set_title('r = {}'.format(r))
        >>>     nh.util.set_figtitle('BGR')
        >>>     plt.gcf().savefig(join(dpath, '{}_{:08d}.png'.format('bgr', x)))
        >>>     pnum_ = nh.util.PlotNums(nRows=4, nCols=5)
        >>>     for r in range(0, 17):
        >>>         imgt = lab_to_bgr(radial_fourier_mask(bgr_to_lab(bgr_img), r)).transpose(1, 2, 0)
        >>>         nh.util.imshow(imgt, pnum=pnum_(), fnum=3)
        >>>         plt.gca().set_title('r = {}'.format(r))
        >>>         #imgt = lab_to_bgr(to_lab(bgr_img)).transpose(1, 2, 0)
        >>>         #nh.util.imshow(lab_to_bgr(to_lab(bgr_img)).transpose(1, 2, 0), pnum=pnum_(), fnum=2)
        >>>     nh.util.set_figtitle('LAB')
        >>>     plt.gcf().savefig(join(dpath, '{}_{:08d}.png'.format('lab', x)))
        >>>     pnum_ = nh.util.PlotNums(nRows=4, nCols=5)
        >>>     for r in range(0, 17):
        >>>         imgt = yuv_to_bgr(radial_fourier_mask(bgr_to_yuv(bgr_img), r, clip=(0., 1.))).transpose(1, 2, 0)
        >>>         nh.util.imshow(imgt, pnum=pnum_(), fnum=4)
        >>>         plt.gca().set_title('r = {}'.format(r))
        >>>     nh.util.set_figtitle('YUV')
        >>>     plt.gcf().savefig(join(dpath, '{}_{:08d}.png'.format('yuv', x)))
        >>> nh.util.show_if_requested()

    Ignore:
        im_chw = bgr_to_lab(bgr_img)
    """
    import cv2
    rows, cols = img_chw.shape[1:3]

    def fourier(s):
        # note: cv2 functions would probably be faster here
        return np.fft.fftshift(np.fft.fft2(s))

    def inv_fourier(f):
        # use real because LAB has negative components
        return np.real(np.fft.ifft2(np.fft.ifftshift(f)))

    diam = radius * 2
    left = int(np.floor((cols - diam) / 2))
    right = int(np.ceil((cols - diam) / 2))
    top = int(np.floor((rows - diam) / 2))
    bot = int(np.ceil((rows - diam) / 2))

    # element = skimage.morphology.disk(radius)
    # mask = np.pad(element, ((top, bot), (left, right)), 'constant')
    if diam > 0:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
        mask = cv2.copyMakeBorder(element, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        mask = 0

    out = np.empty_like(img_chw)
    if axis is None:
        for i, s in enumerate(img_chw):
            # hadamard product (aka simple element-wise multiplication)
            out[i] = inv_fourier(fourier(s) * mask)
    else:
        for i, s in enumerate(img_chw):
            if i in axis:
                # hadamard product (aka simple element-wise multiplication)
                out[i] = inv_fourier(fourier(s) * mask)
            else:
                out[i] = s
    if clip:
        out = np.clip(out, *clip)
    return out

    # nrows = cv2.getOptimalDFTSize(rows)
    # ncols = cv2.getOptimalDFTSize(cols)
    # right = ncols - cols
    # bottom = nrows - rows
    # if right or bottom:
    #     bordertype = cv2.BORDER_CONSTANT  # just to avoid line breakup in PDF file
    #     nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, bordertype, value=0)
    # dft_chans = [cv2.dft(chan, flags=cv2.DFT_COMPLEX_OUTPUT) for chan in img_chw]
    # dft = np.dstack(dft_chans).transpose(2, 0, 1)
    # dft_mag = np.dstack([(c ** 2).sum(axis=-1) for c in dft_chans]).transpose(2, 0, 1)
    # dft_shift = [np.fft.fftshift(c) for c in dft_chans]
    # dft_mag = np.dstack([(c ** 2).sum(axis=-1) for c in dft_shift]).transpose(2, 0, 1)
    # dft_filt_shift = [c * mask[:, :, None] for c in dft_shift]
    # dft_filt = [np.fft.ifftshift(c) for c in dft_filt_shift]
    # idft_filt = [cv2.idft(c) for c in dft_filt]
    # img_filt = np.dstack([np.linalg.norm(c, axis=-1) for c in idft_filt])
    # nh.util.imshow(dft_mag.transpose(1, 2, 0), norm=True)
    # if False:
    #     nh.util.imshow(np.log(dft_mag[0]), norm=True, pnum=(1, 3, 1))
    #     nh.util.imshow(np.log(dft_mag[1]), norm=True, pnum=(1, 3, 2))
    #     nh.util.imshow(np.log(dft_mag[2]), norm=True, pnum=(1, 3, 3))


def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    Args:
        X (ndarray): [M x N] matrix, Rows: Variables, Columns: Observations

    Returns:
        ZCAMatrix: [M x M] matrix

    References:
        https://stackoverflow.com/a/38590790/887074

    Example:
        >>> rng = np.random.RandomState(0)
        >>> # Construct a matrix of observations from grayscale 8x8 images
        >>> gray_images = [rng.rand(8, 8) for _ in range(1000)]
        >>> X = np.array([img.ravel() for img in gray_images]).T
        >>> M = zca_whitening_matrix(X)
        >>> img = gray_images[0]
        >>> norm = M.dot(img.ravel()).reshape(8, 8)
        >>> # ... for the RGB channels of color images
        >>> rgb_images = [rng.rand(3, 8, 8) for _ in range(1000)]
        >>> #X = np.array([img.mean(axis=(1, 2)) for img in rgb_images]).T
        >>> X = np.hstack([img.reshape(3, -1) for img in rgb_images])
        >>> M = zca_whitening_matrix(X)
        >>> img = rgb_images[0]
        >>> norm = M.dot(img.reshape(3, 64)).reshape(3, 8, 8)
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    L = np.diag(1.0 / np.sqrt(S + epsilon))
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(L, U.T))  # [M x M]
    return ZCAMatrix


class CIFAR10_Task(Task):
    """
    task = CIFAR10_Task()
    task._initialize()
    ignore_labelnames = []
    alias = {}
    """
    def __init__(task, root=None):
        if root is None:
            root = ub.ensure_app_cache_dir('netharn')
        task.root = root
        task._initialize()

    def _initialize(task):
        from os.path import join
        import pickle
        train_dset = cifar.CIFAR10(root=task.root, download=False, train=True)

        fpath = join(train_dset.root,
                     cifar.CIFAR10.base_folder, 'batches.meta')
        with open(fpath, 'rb') as fo:
            entry = pickle.load(fo, encoding='latin1')
            labelnames = entry['label_names']
        task.set_labelnames(labelnames)


class CIFAR100_Task(Task):
    """
    task = CIFAR100_Task()
    task._initialize()
    ignore_labelnames = []
    alias = {}
    """
    def __init__(task, root=None):
        if root is None:
            root = ub.ensure_app_cache_dir('netharn')
        task.root = root
        task._initialize()

    def _initialize(task):
        from os.path import join
        import pickle
        train_dset = cifar.CIFAR100(root=task.root, download=False, train=True)

        fpath = join(train_dset.root, cifar.CIFAR100.base_folder, 'meta')
        with open(fpath, 'rb') as fo:
            entry = pickle.load(fo, encoding='latin1')
            labelnames = entry['fine_label_names']
        task.set_labelnames(labelnames)


def mutex_clf_gt_info(gt_labels, task):
    """
    gt_labels = train_dset.train_labels
    """
    index = pd.Index(task.labels, name='label')
    gtstats = pd.DataFrame(0, index=index, columns=['freq'], dtype=np.int)

    label_freq = pd.value_counts(gt_labels)
    gtstats.freq = pd.to_numeric(label_freq)

    gtstats['classname'] = list(ub.take(task.labelnames, gtstats.index))
    gtstats['mf_weight'] = gtstats.freq.median() / gtstats.freq
    gtstats.loc[~np.isfinite(gtstats.mf_weight), 'mf_weight'] = 1

    # Clip weights, so nothing gets crazy high weights, low weights are ok
    gtstats = gtstats.sort_index()
    gtstats.index.name = 'label'
    gtstats = gtstats.reset_index().set_index('classname', drop=False)
    return gtstats


class InMemoryInputs(ub.NiceRepr):
    """
    Change inputs.Inputs to OnDiskInputs
    """
    def __init__(inputs, tag=''):
        inputs.tag = tag
        inputs.im = None
        inputs.gt = None
        inputs.colorspace = None
        inputs.input_id = None

    def __nice__(inputs):
        n = len(inputs)
        return '{} {}'.format(inputs.tag, n)

    def __len__(inputs):
        if inputs.im is not None:
            n = len(inputs.im)
        elif inputs.gt is not None:
            n = len(inputs.gt)
        else:
            n = 0
        return n

    @classmethod
    def from_bhwc_rgb(cls, bhwc, labels=None, **kw):
        # convert to bhwc
        inputs = cls(**kw)
        inputs.im = bhwc
        inputs.gt = labels
        inputs.colorspace = 'rgb'
        return inputs

    def convert_colorspace(inputs, colorspace, inplace=False):
        if colorspace.lower() == inputs.colorspace.lower():
            if not inplace:
                return inputs.im
            return
        im_out = np.empty_like(inputs.im)
        dst = np.ascontiguousarray(np.empty_like(inputs.im[0]))
        for ix, im in enumerate(inputs.im):
            util.convert_colorspace(im, src_space=inputs.colorspace,
                                    dst_space=colorspace, dst=dst)
            im_out[ix] = dst
        if inplace:
            inputs.im = im_out
            inputs.colorspace = colorspace
        else:
            return im_out

    def take(inputs, idxs, **kw):
        new_inputs = inputs.__class__(**kw)
        new_inputs.im = inputs.im.take(idxs, axis=0)
        new_inputs.gt = inputs.gt.take(idxs, axis=0)
        new_inputs.colorspace = inputs.colorspace
        return new_inputs

    def prepare_id(self, force=False):
        if self.input_id is not None and not force:
            return

        depends = []
        depends.append(self.im)
        depends.append(self.gt)

    def _set_id_from_dependency(self, depends):
        """
        Allow for arbitrary representation of dependencies
        (user must ensure that it is consistent)
        """
        print('Preparing id for {} images'.format(self.tag))
        abbrev = 8
        hashid = util.hash_data(depends)[:abbrev]
        n_input = len(self)
        self.input_id = '{}-{}'.format(n_input, hashid)
        print(' * n_input = {}'.format(n_input))
        print(' * input_id = {}'.format(self.input_id))


class CIFAR_Wrapper(torch.utils.data.Dataset):  # cifar.CIFAR10):
    def __init__(dset, inputs, task, workdir, output_colorspace='RGB'):
        dset.inputs = inputs
        dset.task = task

        dset.output_colorspace = output_colorspace

        dset.rng = np.random.RandomState(432432)

        inputs_base = ub.ensuredir((workdir, 'inputs'))
        inputs.base_dpath = inputs_base
        if len(inputs):
            inputs.prepare_id()
            dset.input_id = inputs.input_id
            dset.with_gt = dset.inputs.gt is not None
        else:
            dset.input_id = ''

        # TODO: only use horizontal flipping and translation by 4 pixels to
        # match results from other papers
        # https://arxiv.org/pdf/1603.09382.pdf page 8

        dset.augment = None
        # dset.im_augment = torchvision.transforms.Compose([
        #     RandomGamma(rng=dset.rng),
        #     RandomBlur(rng=dset.rng),
        # ])
        # dset.rand_aff = RandomWarpAffine(dset.rng)

        augmentors = [
            # iaa.Sometimes(.8, iaa.ContrastNormalization((0.2, 1.8))),
            iaa.Fliplr(p=.5),
            iaa.Affine(translate_px={'x': (-1, 1), 'y': (-1, 1)}),

            # CropTo((30, 30)),
            # iaa.Crop(px=(1, 1, 1, 1)),
            # imgaug.Brightness(63),
            # imgaug.RandomCrop((30, 30)),
            # imgaug.MeanVarianceNormalize(all_channel=True)
        ]
        dset.augmenter = iaa.Sequential(augmentors)
        # iaa.Sequential([
        #     iaa.Affine(translate_px={"x":-40}),
        #     iaa.AdditiveGaussianNoise(scale=0.1*255)
        # ])

        # dset.rand_aff = RandomWarpAffine(
        #     dset.rng, tx_pdf=(-2, 2), ty_pdf=(-2, 2), flip_lr_prob=.5,
        #     zoom_pdf=None, shear_pdf=None, flip_ud_prob=None,
        #     enable_stretch=None, default_distribution='uniform')

        dset.center_inputs = None

    def _make_normalizer(dset, mode='independent'):
        """
        Example:
            >>> inputs, task = cifar_inputs(train=True)
            >>> workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))
            >>> dset = CIFAR_Wrapper(inputs, task, workdir, 'RGB')
            >>> center_inputs = dset._make_normalizer('independent')
        """
        if len(dset.inputs):
            # compute normalizers in the output colorspace
            out_im = dset.inputs.convert_colorspace(dset.output_colorspace,
                                                    inplace=False)
            if mode == 'dependant':
                # dependent centering per channel (for RGB)
                im_mean = out_im.mean()
                im_scale = out_im.std()
            elif mode == 'independent':
                # Independent centering per channel (for LAB)
                im_mean = out_im.mean(axis=(0, 1, 2))
                im_scale = out_im.std(axis=(0, 1, 2))

            center_inputs = ImageCenterScale(im_mean, im_scale)

        dset.center_inputs = center_inputs
        return center_inputs

    def __len__(dset):
        return len(dset.inputs)

    def load_inputs(dset, index):
        """
        Ignore:
            >>> inputs, task = cifar_inputs(train=False)
            >>> workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))
            >>> dset = CIFAR_Wrapper(inputs, task, workdir, 'LAB')
            >>> dset._make_normalizer('independent')
            >>> index = 0
            >>> im, gt = dset.load_inputs(index)

        Example:
            >>> inputs, task = cifar_inputs(train=False)
            >>> workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))
            >>> dset = CIFAR_Wrapper(inputs, task, workdir, 'RGB')
            >>> index = 0
            >>> im, gt = dset.load_inputs(index)
            >>> from netharn.util import mplutil
            >>> mplutil.qtensure()
            >>> dset = CIFAR_Wrapper(inputs, task, workdir, 'RGB')
            >>> dset.augment = True
            >>> im, gt = dset.load_inputs(index)
            >>> mplutil.imshow(im, colorspace='rgb')

            >>> dset = CIFAR_Wrapper(inputs, task, workdir, 'LAB')
            >>> dset.augment = True
            >>> im, gt = dset.load_inputs(index)
            >>> mplutil.imshow(im, colorspace='LAB')
        """
        assert dset.inputs.colorspace.lower() == 'rgb', (
            'we must be in rgb for augmentation')
        im = dset.inputs.im[index]

        if dset.inputs.gt is not None:
            gt = dset.inputs.gt[index]
        else:
            gt = None

        if dset.augment:
            # Image augmentation must be done in RGB
            # Augment intensity independently
            # im = dset.im_augment(im)
            # Augment geometry consistently

            # params = dset.rand_aff.random_params()
            # im = dset.rand_aff.warp(im, params, interp='cubic', backend='cv2')

            im = util.convert_colorspace(im, src_space=dset.inputs.colorspace,
                                         dst_space='rgb')
            # Do augmentation in uint8 RGB
            im = (im * 255).astype(np.uint8)
            im = dset.augmenter.augment_image(im)
            im = (im / 255).astype(np.float32)
            im = util.convert_colorspace(im, src_space='rgb',
                                         dst_space=dset.output_colorspace)
        else:
            im = util.convert_colorspace(im, src_space=dset.inputs.colorspace,
                                         dst_space=dset.output_colorspace)
        # Do centering of inputs
        if dset.center_inputs:
            im = dset.center_inputs(im)
        return im, gt

    def __getitem__(dset, index):
        from netharn import im_loaders
        im, gt = dset.load_inputs(index)
        input_tensor = im_loaders.numpy_image_to_float_tensor(im)

        if dset.with_gt:
            # print('gotitem: ' + str(data_tensor.shape))
            # print('gt_tensor: ' + str(gt_tensor.shape))
            return input_tensor, gt
        else:
            return input_tensor

    @property
    def n_channels(dset):
        return 3

    @property
    def n_classes(dset):
        return int(dset.task.labels.max() + 1)

    @property
    def ignore_labels(dset):
        return dset.task.ignore_labels

    def class_weights(dset):
        """
            >>> from netharn.live.sseg_train import *
            >>> dset = load_task_dataset('urban_mapper_3d')['train']
            >>> dset.class_weights()
        """
        # # Handle class weights
        # print('prep class weights')
        # gtstats = dset.inputs.prepare_gtstats(dset.task)
        # gtstats = dset.inputs.gtstats
        # # Take class weights (ensure they are in the same order as labels)
        # mfweight_dict = gtstats['mf_weight'].to_dict()
        # class_weights = np.array(list(ub.take(mfweight_dict, dset.task.classnames)))
        # class_weights[dset.task.ignore_labels] = 0
        # # HACK
        # # class_weights[0] = 1.0
        # # class_weights[1] = 0.7
        # print('class_weights = {!r}'.format(class_weights))
        # print('class_names   = {!r}'.format(dset.task.classnames))
        class_weights = np.ones(dset.n_classes)
        return class_weights


def cifar_inputs(train=False, cifar_num=10):
    root = ub.ensure_app_cache_dir('netharn')

    if cifar_num == 10:
        train_dset = cifar.CIFAR10(root=root, download=True, train=train)
        task = CIFAR10_Task()
    else:
        train_dset = cifar.CIFAR100(root=root, download=True, train=train)
        task = CIFAR100_Task()
    if train:
        bchw = (train_dset.train_data).astype(np.float32) / 255.0
        labels = np.array(train_dset.train_labels)
    else:
        bchw = (train_dset.test_data).astype(np.float32) / 255.0
        labels = np.array(train_dset.test_labels)
    inputs = InMemoryInputs.from_bhwc_rgb(bchw, labels=labels)
    if train:
        inputs.tag = 'learn'
    else:
        inputs.tag = 'test'
    return inputs, task


def cifar_training_datasets(output_colorspace='RGB', norm_mode='independent',
                            cifar_num=10):
    """
    Example:
        >>> datasets = cifar_training_datasets()
    """
    inputs, task = cifar_inputs(train=True, cifar_num=cifar_num)

    # split training into train / validation
    # 45K / 5K validation split was used in densenet and resnet papers.
    # https://arxiv.org/pdf/1512.03385.pdf page 7
    # https://arxiv.org/pdf/1608.06993.pdf page 5

    vali_frac = .1  # 10%  is 5K images
    n_vali = int(len(inputs) * vali_frac)
    # n_vali = 10000  # 10K validation as in http://torch.ch/blog/2015/07/30/cifar.html

    # the gt indexes seem to already be scrambled, I think other papers sample
    # validation from the end, so lets do that
    # The NIN paper https://arxiv.org/pdf/1312.4400.pdf in section 4 mentions
    # that it uses the last 10K images for validation
    input_idxs = np.arange(len(inputs))
    # or just uncomment this line for reproducable random sampling
    # input_idxs = util.random_indices(len(inputs), seed=1184576173)

    train_idxs = sorted(input_idxs[:-n_vali])
    vali_idxs = sorted(input_idxs[-n_vali:])

    train_inputs = inputs.take(train_idxs, tag='train')
    vali_inputs = inputs.take(vali_idxs, tag='vali')
    test_inputs, _ = cifar_inputs(train=False, cifar_num=cifar_num)
    # The dataset name and indices should fully specifiy dependencies
    train_inputs._set_id_from_dependency(
        ['cifar{}-train'.format(cifar_num), train_idxs])
    vali_inputs._set_id_from_dependency(
        ['cifar{}-train'.format(cifar_num), vali_idxs])
    test_inputs._set_id_from_dependency(['cifar{}-test'.format(cifar_num)])

    workdir = ub.ensuredir(ub.truepath('~/data/work/cifar'))

    train_dset = CIFAR_Wrapper(
        train_inputs, task, workdir, output_colorspace=output_colorspace)
    vali_dset = CIFAR_Wrapper(
        vali_inputs, task, workdir, output_colorspace=output_colorspace)
    test_dset = CIFAR_Wrapper(test_inputs, task, workdir,
                              output_colorspace=output_colorspace)
    print('built datasets')

    datasets = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }

    print('computing normalizers')
    datasets['train'].center_inputs = datasets['train']._make_normalizer(
        norm_mode)
    for key in datasets.keys():
        datasets[key].center_inputs = datasets['train'].center_inputs
    print('computed normalizers')

    datasets['train'].augment = True
    return datasets

def train():
    """
    Example:
        >>> train()
    """
    import random
    np.random.seed(1031726816 % 4294967295)
    torch.manual_seed(137852547 % 4294967295)
    random.seed(2497950049 % 4294967295)

    xpu = xpu_device.XPU.from_argv()
    print('Chosen xpu = {!r}'.format(xpu))

    cifar_num = 10

    if ub.argflag('--lab'):
        datasets = cifar_training_datasets(
            output_colorspace='LAB', norm_mode='independent', cifar_num=cifar_num)
    elif ub.argflag('--rgb'):
        datasets = cifar_training_datasets(
            output_colorspace='RGB', norm_mode='independent', cifar_num=cifar_num)
    elif ub.argflag('--rgb-dep'):
        datasets = cifar_training_datasets(
            output_colorspace='RGB', norm_mode='dependant', cifar_num=cifar_num)
    else:
        raise AssertionError('specify --rgb / --lab')

    import netharn.models.densenet

    # batch_size = (128 // 3) * 3
    batch_size = 64

    # initializer_ = (initializers.KaimingNormal, {
    #     'nonlinearity': 'relu',
    # })

    lr = 0.1
    initializer_ = (initializers.LSUV, {})

    hyper = hyperparams.HyperParams(
        model=(netharn.models.densenet.DenseNet, {
            'cifar': True,
            'block_config': (32, 32, 32),  # 100 layer depth
            'num_classes': datasets['train'].n_classes,
            'drop_rate': float(ub.argval('--drop_rate', default=.2)),
            'groups': 1,
        }),
        optimizer=(torch.optim.SGD, {
            # 'weight_decay': .0005,
            'weight_decay': float(ub.argval('--weight_decay', default=.0005)),
            'momentum': 0.9,
            'nesterov': True,
            'lr': 0.1,
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
            'maximize': ['mAP'],
            'patience': 314,
            'max_epoch': 314,
        }),
        initializer=initializer_,
        criterion=(torch.nn.CrossEntropyLoss, {
        }),
        # Specify anything else that is special about your hyperparams here
        # Especially if you make a custom_batch_runner
        augment=str(datasets['train'].augmenter),
        other=ub.dict_union({
            # TODO: type of augmentation as a parameter dependency
            # 'augmenter': str(datasets['train'].augmenter),
            # 'augment': datasets['train'].augment,
            'batch_size': batch_size,
            'colorspace': datasets['train'].output_colorspace,
            'n_classes': datasets['train'].n_classes,
            # 'center_inputs': datasets['train'].center_inputs,
        }, datasets['train'].center_inputs.__dict__),
    )
    # if ub.argflag('--rgb-indie'):
    #     hyper.other['norm'] = 'dependant'
    hyper.input_ids['train'] = datasets['train'].input_id

    xpu = xpu_device.XPU.cast('auto')
    print('xpu = {}'.format(xpu))

    data_kw = {'batch_size': batch_size}
    if xpu.is_gpu():
        data_kw.update({'num_workers': 8, 'pin_memory': True})

    tags = ['train', 'vali', 'test']

    loaders = ub.odict()
    for tag in tags:
        dset = datasets[tag]
        shuffle = tag == 'train'
        data_kw_ = data_kw.copy()
        if tag != 'train':
            data_kw_['batch_size'] = max(batch_size // 4, 1)
        loader = torch.utils.data.DataLoader(dset, shuffle=shuffle, **data_kw_)
        loaders[tag] = loader

    harn = fit_harness.FitHarness(
        hyper=hyper, datasets=datasets, xpu=xpu,
        loaders=loaders,
    )
    # harn.monitor = early_stop.EarlyStop(patience=40)
    harn.monitor = monitor.Monitor(min_keys=['loss'],
                                   max_keys=['global_acc', 'class_acc'],
                                   patience=40)

    # ignore_label = datasets['train'].ignore_label
    # from netharn import metrics

    workdir = ub.ensuredir('train_cifar_work')
    harn.setup_dpath(workdir)

    harn.run()
