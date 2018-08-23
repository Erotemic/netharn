# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub  # NOQA
import cv2
import numpy as np
import netharn as nh
import torch
import torch.utils.data as torch_data
import itertools as it


class SlidingWindow(ub.NiceRepr):
    """
    Generates basis for "sliding window" slices to break a large image into
    smaller pieces. Use it.product to slide across the coordinates.

    Notes:
        This is a simpler version of SlidingSlices

    Args:
        shape (ndarray): shape of source array to slide across.

        window (tuple): shape of window

        overlap (float): a number between 0 and 1 indicating the fraction of
            overlap that parts will have. Must be `0 <= overlap < 1`.

        keepbound (bool): if True, a non-uniform stride will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

    Attributes:
        basis_shape - shape of the grid corresponding to the number of strides
            the sliding window will take.
        basis_slices - slices that will be taken in every dimension

    Yields:
        Tuple[slice]: slices used for numpy indexing

    Example:
        >>> shape = (220, 220)
        >>> window = (10, 10)
        >>> self = SlidingWindow(shape, window, stride=5)
        >>> list(self)[41:45]
        [(slice(0, 10, None), slice(205, 215, None)),
         (slice(0, 10, None), slice(210, 220, None)),
         (slice(5, 15, None), slice(0, 10, None)),
         (slice(5, 15, None), slice(5, 15, None))]
        >>> print('self.overlap = {!r}'.format(self.overlap))
        self.overlap = [0.5, 0.5]

    Example:
        >>> shape = (4, 4)
        >>> window = (3, 3)
        >>> self = SlidingWindow(shape, window, stride=1)
        >>> list(zip(self.centers, self.slices))
        [((1.0, 1.0), (slice(0, 3, None), slice(0, 3, None))),
         ((1.0, 2.0), (slice(0, 3, None), slice(1, 4, None))),
         ((2.0, 1.0), (slice(1, 4, None), slice(0, 3, None))),
         ((2.0, 2.0), (slice(1, 4, None), slice(1, 4, None)))]
        >>> shape = (3, 3)
        >>> window = (2, 2)
        >>> self = SlidingWindow(shape, window, stride=1)
        >>> list(zip(self.centers, self.slices))
        [((0.5, 0.5), (slice(0, 2, None), slice(0, 2, None))),
         ((0.5, 1.5), (slice(0, 2, None), slice(1, 3, None))),
         ((1.5, 0.5), (slice(1, 3, None), slice(0, 2, None))),
         ((1.5, 1.5), (slice(1, 3, None), slice(1, 3, None)))]

    Example:
        >>> shape = (16, 16)
        >>> window = (4, 4)
        >>> self = SlidingWindow(shape, window, overlap=(.5, .25))
        >>> print('self.stride = {!r}'.format(self.stride))
        self.stride = [2, 3]
        >>> list(ub.chunks(self.grid, 5))
        [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
         [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
         [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
         [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
         [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
         [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)],
         [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]]
    """
    def __init__(self, shape, window, overlap=None, stride=None,
                 keepbound=False, allow_overshoot=False):
        assert len(window) == len(shape), (
            'incompatible dims: {} {}'.format(len(window),
                                              len(shape)))
        assert all(d <= D for d, D in zip(window, shape)), (
                'window must be smaller than target')

        stride, overlap = self._compute_stride(overlap, stride, shape,
                                               window)

        if not all(stride):
            raise ValueError(
                'Step must be positive everywhere. Got={}'.format(stride))

        stide_kw = [dict(margin=d, stop=D, step=s, keepbound=keepbound,
                         check=not keepbound and not allow_overshoot)
                      for d, D, s in zip(window, shape, stride)]

        undershot_shape = []
        overshoots = []
        for kw in stide_kw:
            final_pos = (kw['stop'] - kw['margin'])
            n_steps = final_pos // kw['step']
            overshoot = final_pos % kw['step']
            undershot_shape.append(n_steps + 1)
            overshoots.append(overshoot)

        if not allow_overshoot and any(overshoots):
            raise ValueError('overshoot={} stide_kw={}'.format(overshoots,
                                                               stide_kw))

        # make a slice generator for each dimension
        self.stride = stride
        self.overlap = overlap

        self.window = window
        self.input_shape = shape

        # The undershot basis shape, only contains indices that correspond
        # perfectly to the input. It may crop a bit of the ends.  If this is
        # equal to basis_shape, then the self perfectly fits the input.
        self.undershot_shape = undershot_shape

        # NOTE: if we have overshot, then basis shape will not perfectly
        # align to the original image. This shape will be a bit bigger.
        self.basis_slices = [list(nh.util.wide_strides_1d(**kw))
                               for kw in stide_kw]
        self.basis_shape = [len(b) for b in self.basis_slices]
        self.n_total = np.prod(self.basis_shape)

    def __nice__(self):
        return '{}, stride={}'.format(self.basis_shape, self.stride)

    def _compute_stride(self, overlap, stride, shape, window):
        """
        Ensures that stride hasoverlap the correct shape.  If stride is not provided,
        compute stride from desired overlap.
        """
        if not (overlap is None) ^ (stride is None):
            raise ValueError('specify overlap({}) XOR stride ({})'.format(
                overlap, stride))
        if stride is None:
            if not isinstance(overlap, (list, tuple)):
                overlap = [overlap] * len(window)
            if any(frac < 0 or frac >= 1 for frac in overlap):
                raise ValueError((
                    'part overlap was {}, but fractional overlaps must be '
                    'in the range [0, 1)').format(overlap))
            stride = [int(round(d - d * frac))
                      for frac, d in zip(overlap, window)]
        else:
            if not isinstance(stride, (list, tuple)):
                stride = [stride] * len(window)
        # Recompute fractional overlap after integer stride is computed
        overlap = [(d - s) / d for s, d in zip(stride, window)]
        assert len(stride) == len(shape), 'incompatible dims'
        return stride, overlap

    def __len__(self):
        return self.n_total

    def _iter_basis_frac(self):
        for slices in self._iter_slices():
            frac = [sl.start / D for sl, D in zip(slices, self.source.shape)]
            yield frac

    def _iter_basis_idxs(self):
        basis_indices = map(range, self.basis_shape)
        for basis_idxs in it.product(*basis_indices):
            yield basis_idxs

    def _iter_slices(self):
        for slices in it.product(*self.basis_slices):
            yield slices

    def __iter__(self):
        # yield from
        for _ in self._iter_slices():
            yield _

    @property
    def grid(self):
        return self._iter_basis_idxs()

    @property
    def slices(self):
        return self._iter_slices()

    @property
    def centers(self):
        for slices in self._iter_slices():
            center = tuple(sl_.start + (sl_.stop - sl_.start - 1) / 2 for sl_ in slices)
            yield center


class SlidingSlices(ub.NiceRepr):
    """
    Generates basis for "sliding window" slices to break a large image into
    smaller pieces. Use it.product to slide across the coordinates.

    Args:
        source (ndarray): array to slice across. It is typically best to ensure
            this is in CHW or CDHW format for maximum compatibility.

        target_shape (tuple): (chan, depth, height, width) of the window
            (be sure to include channels). CHW or CDHW format.

        overlap (float): a number between 0 and 1 indicating the fraction of
            overlap that parts will have. Must be `0 <= overlap < 1`.

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

    Attributes:
        basis_shape - shape of the grid corresponding to the number of steps
            the sliding window will take.
        basis_slices - slices that will be taken in every dimension

    Yields:
        tuple(slice, slice): row and column slices used for numpy indexing

    Example:
        >>> source = np.zeros((220, 220))
        >>> target_shape = (10, 10)
        >>> slider = SlidingSlices(source, target_shape, step=5)
        >>> list(slider.slices)[41:45]
        [(slice(0, 10, None), slice(205, 215, None)),
         (slice(0, 10, None), slice(210, 220, None)),
         (slice(5, 15, None), slice(0, 10, None)),
         (slice(5, 15, None), slice(5, 15, None))]
        >>> print('slider.overlap = {!r}'.format(slider.overlap))
        slider.overlap = [0.5, 0.5]

    Example:
        >>> source = np.zeros((250, 200, 200))
        >>> target_shape = (10, 10, 10)
        >>> slider = SlidingSlices(source, target_shape, step=(1, 2, 2))
        >>> chip = next(slider.chips)
        >>> print('chip.shape = {!r}'.format(chip.shape))
        chip.shape = (10, 10, 10)

    Example:
        >>> source = np.zeros((16, 16))
        >>> target_shape = (4, 4)
        >>> slider = SlidingSlices(source, target_shape, overlap=(.5, .25))
        >>> print('slider.step = {!r}'.format(slider.step))
        slider.step = [2, 3]
        >>> list(ub.chunks(slider.grid, 5))
        [[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
         [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
         [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
         [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
         [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
         [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4)],
         [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]]
    """
    def __init__(slider, source, target_shape, overlap=None, step=None,
                 keepbound=False, allow_overshoot=False):
        img_shape = source.shape
        assert len(target_shape) == len(img_shape), (
            'incompatible dims: {} {}'.format(len(target_shape),
                                              len(img_shape)))
        assert all(d <= D for d, D in zip(target_shape, img_shape)), (
                'window must be smaller than target')

        step, overlap = slider._compute_step(overlap, step, img_shape,
                                             target_shape)

        if not all(step):
            raise ValueError(
                'Step must be positive everywhere. Got={}'.format(step))

        stide_kw = [dict(margin=d, stop=D, step=s, keepbound=keepbound,
                         check=not keepbound and not allow_overshoot)
                      for d, D, s in zip(target_shape, img_shape, step)]

        undershot_shape = []
        overshoots = []
        for kw in stide_kw:
            final_pos = (kw['stop'] - kw['margin'])
            n_steps = final_pos // kw['step']
            overshoot = final_pos % kw['step']
            undershot_shape.append(n_steps + 1)
            overshoots.append(overshoot)

        if not allow_overshoot and any(overshoots):
            raise ValueError('overshoot={} stide_kw={}'.format(overshoots,
                                                               stide_kw))

        # make a slice generator for each dimension
        slider.step = step
        slider.overlap = overlap
        slider.source = source

        slider.window = target_shape

        # The undershot basis shape, only contains indices that correspond
        # perfectly to the input. It may crop a bit of the ends.  If this is
        # equal to basis_shape, then the slider perfectly fits the input.
        slider.undershot_shape = undershot_shape

        # NOTE: if we have overshot, then basis shape will not perfectly
        # align to the original image. This shape will be a bit bigger.
        slider.basis_slices = [tuple(nh.util.wide_strides_1d(**kw))
                               for kw in stide_kw]
        slider.basis_shape = [len(b) for b in slider.basis_slices]
        slider.n_total = np.prod(slider.basis_shape)

    def __nice__(slider):
        return '{}, step={}'.format(slider.basis_shape, slider.step)

    def _compute_step(slider, overlap, step, img_shape, target_shape):
        """
        Ensures that step hasoverlap the correct shape.  If step is not provided,
        compute step from desired overlap.
        """
        if not (overlap is None) ^ (step is None):
            raise ValueError('specify overlap({}) XOR step ({})'.format(
                overlap, step))
        if step is None:
            if not isinstance(overlap, (list, tuple)):
                overlap = [overlap] * len(target_shape)
            if any(frac < 0 or frac >= 1 for frac in overlap):
                raise ValueError((
                    'part overlap was {}, but fractional overlaps must be '
                    'in the range [0, 1)').format(overlap))
            step = [int(round(d - d * frac))
                    for frac, d in zip(overlap, target_shape)]
        else:
            if not isinstance(step, (list, tuple)):
                step = [step] * len(target_shape)
        # Recompute fractional overlap after integer step is computed
        overlap = [(d - s) / d for s, d in zip(step, target_shape)]
        assert len(step) == len(img_shape), 'incompatible dims'
        return step, overlap

    def __len__(slider):
        return slider.n_total

    def _iter_basis_frac(slider):
        for slices in slider._iter_slices():
            frac = [sl.start / D for sl, D in zip(slices, slider.source.shape)]
            yield frac

    def _iter_basis_idxs(slider):
        basis_indices = map(range, slider.basis_shape)
        for basis_idxs in it.product(*basis_indices):
            yield basis_idxs

    def _iter_slices(slider):
        for slices in it.product(*slider.basis_slices):
            yield slices

    def _iter_chips(slider):
        for slices in slider._iter_slices():
            chip = slider.source[slices]
            yield chip

    def __iter__(slider):
        # yield from zip(slider.slices, slider.chips)
        for _ in zip(slider.slices, slider.chips):
            yield _

    @property
    def grid(self):
        return self._iter_basis_idxs()

    @property
    def slices(self):
        return self._iter_slices()

    @property
    def chips(self):
        return self._iter_chips()

    def to_dataset(self):
        slider_dset = SlidingIndexDataset(self)
        return slider_dset

    def clf_upscale_transform(slider, dims=(-2, -1)):
        """
        Find transformation to upscale a single scalar classification for each
        window back to the spatial resolution of the original data.

        FIXME:
            This contains bugs that will cause slight alignment errors.

            NOTE:
                returned scales are not correct

                * This does work when the window size is 1x1
                * This does work when the step size is 1

        Args:
            dims (tuple): indices of the spatial (height and width) dimensions

        Example:
            >>> source = np.zeros((3, 25, 25))
            >>> window = (3, 5, 5)
            >>> step = 2
            >>> slider = SlidingSlices(source, window, step=step)
            >>> dims = (-2, -1)
            >>> # Make dummy predicted data
            >>> pred_shape = list(ub.take(slider.basis_shape, dims))
            >>> pred = np.arange(slider.n_total).reshape(pred_shape)
            >>> # upscale using computed transforms
            >>> (yscale, xscale), padding, prepad_shape = slider.clf_upscale_transform(dims)
            >>> cv2.resize(pred.astype(np.uint8), prepad_shape)[0].shape
            >>> resized = cv2.resize(pred.astype(np.uint8), prepad_shape)
            >>> resized = np.pad(resized, padding, mode='constant')
            >>> # FIXME: Following scale doesnt work right
            >>> nh.util.imscale(pred.astype(np.uint8), (xscale, yscale))[0].shape
        """
        def slcenter(sl):
            """ center of the window defined by a slice """
            return sl.start + (sl.stop - sl.start - 1) / 2

        def slstep(slices):
            return slices[1].start - slices[0].start

        ydim, xdim = dims

        # Get the height / width of the original data we want to resize to
        orig_h = slider.source.shape[ydim]
        orig_w = slider.source.shape[xdim]

        # Find the windows corresponding to x and y spatial dimensions
        yslices = slider.basis_slices[ydim]
        xslices = slider.basis_slices[xdim]

        # The step size between points corresponds to an integer scale factor?
        # FIXME: This is wrong. Should scale be a function of step and window?
        yscale = slstep(yslices)
        xscale = slstep(xslices)

        # Find padding to account for sliding window boundary effects
        # FIXME: is this really how big the padding should be?
        top  = int(np.floor(slcenter(yslices[0])))
        left = int(np.floor(slcenter(xslices[0])))
        bot   = yslices[-1].stop - int(np.floor(slcenter(yslices[-1]))) - 1
        right = xslices[-1].stop - int(np.floor(slcenter(xslices[-1]))) - 1

        padding = ((top, bot), (left, right))

        # Find the shape we will upscale to before padding
        # updscale + padding should result in the original shape

        # for some reason my initial thought on how to calculate this indirectly failed
        prepad_h = orig_h - left - right
        prepad_w = orig_w - top - bot
        prepad_shape = (prepad_h, prepad_w)

        pred_h, pred_w = list(ub.take(slider.basis_shape, dims))

        # prepad_h / pred_h
        # prepad_w / pred_w

        # Note:
        # when we do this correctly, it is possible padding may be negative
        # if the stride is less than the window size. This is because scale
        # should simply be scaling to the point where the extend of the
        # predicted pixels touches each other but does not overlap.
        # This would mean:
        # * translating by half the window width + .5 (so odd kernels are
        # aligned with center pixels, and even kernels are aligned between
        # boundaries)
        # * scaling by half the stride to make the exent of each pixel touch
        # * padding by half the window size minus half the stride. Or clipping
        # by that amount if it is negative
        return (yscale, xscale), padding, prepad_shape

    def upscale_overlay(slider, pred, dims=(-2, -1)):
        """
        Upscales a prediction computed at each point in the sliding window to
        overlay on top of the original spatial resolution (albiet coarsley)

        TODO:
            handle the case where overshoots happen, should there be an extra
            translation to account for them? Or does this scheme already take
            that into account?

            It does not because the steps might be nonlinear when keepbound=True,
            but when it is False the steps are linear and this does handle it.

        Example:
            >>> source = np.zeros((3, 11, 11))
            >>> window = (3, 5, 5)
            >>> step = 6
            >>> slider = SlidingSlices(source, window, step=step)
            >>> dims = (-2, -1)
            >>> # Make dummy predicted data
            >>> pred_shape = list(ub.take(slider.basis_shape, dims))
            >>> pred = np.arange(1, slider.n_total + 1).reshape(pred_shape).astype(np.float)
            >>> # upscale using computed transforms
            >>> upscaled = slider.upscale_overlay(pred)

        Example:
            >>> source = np.zeros((3, 20, 20))
            >>> window = (3, 3, 3)
            >>> step = 6
            >>> slider = SlidingSlices(source, window, step=step, allow_overshoot=True)
            >>> dims = (-2, -1)
            >>> # Make dummy predicted data
            >>> pred_shape = list(ub.take(slider.basis_shape, dims))
            >>> pred = np.arange(1, slider.n_total + 1).reshape(pred_shape).astype(np.float)
            >>> # upscale using computed transforms
            >>> upscaled = slider.upscale_overlay(pred)
        """
        # We can model this with a simple affine transform.  First allocate the
        # required output size, then construct the transform. Padding and
        # cropping will occur naturally.
        ydim, xdim = dims

        # Get the height / width of the original data we want to resize to
        orig_h = slider.source.shape[ydim]
        orig_w = slider.source.shape[xdim]

        # First scale, then translate
        sy = slider.step[ydim]
        sx = slider.step[xdim]

        ty = slider.window[ydim] / 2 - .5
        tx = slider.window[xdim] / 2 - .5

        aff = np.array([
            [sx,  0, tx],
            [ 0, sy, ty],
        ])
        dsize = (orig_w, orig_h)

        if pred.dtype.kind == 'i':
            upscaled = cv2.warpAffine(pred, aff, dsize, flags=cv2.INTER_NEAREST)
        else:
            upscaled = cv2.warpAffine(pred, aff, dsize, flags=cv2.INTER_LINEAR)
        return upscaled


class SlidingIndexDataset(torch_data.Dataset):
    """
    Faster loading of slices at cost of memory

    slider_dset = SlidingIndexDataset(slider)

    slider_loader = torch_data.DataLoader(slider_dset, shuffle=False, batch_size=128)
    slider_iter = iter(slider_loader)
    batch = next(slider_iter)

    """

    def __init__(slider_dset, slider):
        slider_dset.slider = slider

    def __len__(slider_dset):
        return slider_dset.slider.n_total
        # return np.prod(slider.basis_shape)

    def __getitem__(slider_dset, index):
        slider = slider_dset.slider
        basis_idx = np.unravel_index(index, slider.basis_shape)
        slices = tuple([bdim[i] for bdim, i in zip(slider.basis_slices, basis_idx)])
        chip = slider.source[slices]
        tensor_chip = torch.FloatTensor(chip)
        tensor_basis_idx = torch.LongTensor(np.array(basis_idx))
        return tensor_basis_idx, tensor_chip


class Stitcher(ub.NiceRepr):
    """
    Restitches smaller image patches / pixels into a larger output.  This is
    used to invert the SlidingSlicer.  For semenatic segmentation the patches
    are probability chips. Overlapping chips are averaged together.

    Args:
        shape (tuple): dimensions of the large image that will be created from
            the smaller pixels or patches.

    Example:
        >>> import sys
        >>> # Build a high resolution image and slice it into chips
        >>> highres = np.random.rand(5, 200, 200).astype(np.float32)
        >>> target_shape = (1, 50, 50)
        >>> slider = SlidingSlices(highres, target_shape, overlap=(0, .5, .5))
        >>> # Show how Sticher can be used to reconstruct the original image
        >>> stitcher = Stitcher(slider.source.shape)
        >>> for sl, chip in list(slider):
        ...     stitcher.add(sl, chip)
        >>> assert stitcher.weights.max() == 4, 'some parts should be processed 4 times'
        >>> recon = stitcher.finalize()

    """
    def __init__(stitcher, shape, xpu='numpy'):
        stitcher.shape = shape
        stitcher.xpu = xpu
        if xpu == 'numpy':
            stitcher.sums = np.zeros(shape, dtype=np.float32)
            stitcher.weights = np.zeros(shape, dtype=np.float32)

            stitcher.sumview = stitcher.sums.ravel()
            stitcher.weightview = stitcher.weights.ravel()
        else:
            stitcher.sums = xpu.move(torch.zeros(shape))
            stitcher.weights = xpu.move(torch.zeros(shape))

            stitcher.sumview = stitcher.sums.view(-1)
            stitcher.weightview = stitcher.weights.view(-1)

            stitcher._cumprod = np.cumprod(list(shape[::-1][:-1]))[::-1]
            stitcher._cumprod = torch.LongTensor(np.array(stitcher._cumprod))

    def __nice__(stitcher):
        return str(stitcher.sums.shape)

    def add(stitcher, indices, patch, weight=None):
        """
        Incorporate a new (possibly overlapping) patch or pixel using a
        weighted sum.

        Args:
            indices (slice or tuple): typically a slice of pixels or a single
                pixel, but this can be any numpy fancy index.
            patch (ndarray): data to patch into the bigger image.
            weight (float or ndarray): weight of this patch (default to 1.0)
        """
        if weight is None:
            stitcher.sums[indices] += patch
            stitcher.weights[indices] += 1.0
        else:
            stitcher.sums[indices] += (patch * weight)
            stitcher.weights[indices] += weight

    def add_fast(stitcher, batch_idxs, values, weight=None, assume_order=True):
        """
        new faster version

        Ignore:

            stitcher = velocity_sticher
            values = vel_np

            import ubelt
            for timer in ubelt.Timerit(10, bestof=1):
                with timer:
                    stitcher_add(stitcher, batch_idxs, values, assume_order=False)

            import ubelt
            for timer in ubelt.Timerit(10, bestof=1):
                with timer:
                    stitcher_add(stitcher, batch_idxs, values, assume_order=True)

            import ubelt
            batch_idxs_tuple = list(map(tuple, batch_idxs))
            for timer in ubelt.Timerit(10, bestof=1):
                with timer:
                    for indices, vel in zip(batch_idxs_tuple, vel_np):
                        velocity_sticher.add(indices, vel)

        Example:
            >>> import sys
            >>> # Build a high resolution image and slice it into chips
            >>> frames = np.random.rand(1, 200, 100, 100).astype(np.float32)
            >>> window = (frames.shape[0], 15, 15, 15)
            >>> slider = SlidingSlices(frames, window, step=(1, 1, 1, 1))
            >>> dset = slider.to_dataset()
            >>> n_classes = 2
            >>> xpu = nh.XPU(None)
            >>> stitcher = Stitcher(slider.basis_shape[1:] + [n_classes], xpu=xpu)
            >>> loader = torch.utils.data.DataLoader(dset, batch_size=10)
            >>> batch_iter = iter(loader)
            >>> batch = next(batch_iter)
            >>> batch_idxs_tensors_, chips = batch
            >>> invar = torch.autograd.Variable(chips)
            >>> conv = torch.nn.Conv3d(frames.shape[0], n_classes, window[1:])
            >>> values = conv(invar).data
            >>> # remove channel
            >>> weight = None
            >>> batch_idxs = batch_idxs_tensors_[:, 1:]
            >>> stitcher.add_fast(batch_idxs, values, weight, assume_order=True)

        Time:
            torch.cuda.init()

            weight = None

            import ubelt as ub
            xpu = nh.XPU(0)
            values = xpu.move(values)
            stitcher = Stitcher(slider.basis_shape[1:] + [n_classes], xpu=xpu)
            for timer in ub.Timerit(100, bestof=10, label='gpu'):
                with timer:
                    stitcher.add_fast(batch_idxs, values, weight, assume_order=True)

            stitcher = Stitcher(slider.basis_shape[1:] + [n_classes], xpu='numpy')
            batch_idxs_np = batch_idxs.numpy()
            values_np = values.cpu().numpy()
            for timer in ub.Timerit(100, bestof=10, label='numpy'):
                with timer:
                    stitcher.add_fast(batch_idxs_np, values_np, weight, assume_order=True)

        Benchmark:
            >>> import sys
            >>> # setup benchmark
            >>> frames = np.random.rand(1, 50, 100, 100).astype(np.float32)
            >>> window = (frames.shape[0], 20, 20, 20)
            >>> slider = SlidingSlices(frames, window, step=(1, 1, 1, 1))
            >>> dset = slider.to_dataset()
            >>> loader = torch.utils.data.DataLoader(dset, batch_size=1024)
            >>> n_classes = 2
            >>> xpu = nh.XPU(1)
            >>> conv = torch.nn.Conv3d(window[0], n_classes, window[1:])
            >>> conv = xpu.move(conv)
            >>> #weight = torch.rand(n_classes, 1, 1, 1)[None, :]
            >>> #weight = xpu.move(weight)
            >>> #weight_np = weight.cpu().numpy()
            >>> weight = weight_np = None
            >>> # do dummy computation to warm up gpu
            >>> conv(xpu.variable(dset[0][1][None, :]))
            >>> torch.set_grad_enabled(False)
            >>> conv.train(False)
            >>> base_shape = slider.basis_shape[1:]
            >>> # ---------------------------------------
            >>> # Benchmark on-gpu stitching with pytorch
            >>> import tqdm
            >>> t1 = ub.Timerit(3, bestof=3, label='gpu')
            >>> for timer in tqdm.tqdm(t1, total=3, leave=True):
            >>>     with timer:
            >>>         stitcher = Stitcher(base_shape + [n_classes], xpu=xpu)
            >>>         for batch in loader:
            >>>             batch_idxs_tensors_, chips = batch
            >>>             invar = xpu.variable(chips, async=True)
            >>>             values = conv(invar).data
            >>>             batch_idxs = batch_idxs_tensors_[:, 1:].numpy()
            >>>             stitcher.add_fast(batch_idxs, values, weight,
            >>>                               assume_order=True)
            >>> # ---------------------------------------
            >>> # Benchmark on-cpu stitching with numpy
            >>> t2 = ub.Timerit(3, bestof=3, label='numpy')
            >>> for timer in tqdm.tqdm(t2, total=3, leave=True):
            >>>     with timer:
            >>>         stitcher = Stitcher(base_shape + [n_classes], xpu='numpy')
            >>>         for batch in iter(loader):
            >>>             batch_idxs_tensors_, chips = batch
            >>>             invar = xpu.variable(chips, async=True)
            >>>             values_np = conv(invar).data.cpu().numpy()
            >>>             batch_idxs_np = batch_idxs_tensors_[:, 1:].numpy()
            >>>             stitcher.add_fast(batch_idxs_np, values_np,
            >>>                               weight_np, assume_order=True)
            >>> # VERDICT:
            >>> # Async GPU stitching gives a minor but insignificant speedup
            >>> # GPU:   time per loop: best=4.394 s, mean=4.394 ± 0.0 s
            >>> # NUMPY: time per loop: best=4.876 s, mean=4.876 ± 0.0 s
        """
        if stitcher.xpu != 'numpy':
            # ON GPU STITCHING
            n_classes = stitcher.shape[-1]
            end = batch_idxs.shape[0] - 1
            t_base_multi_idxs = batch_idxs[[0, end]]

            # we dont need a trailing 1 because we arent padding extra zeros
            cumprod = stitcher._cumprod[None :]
            ravel_idxs_range = (t_base_multi_idxs * cumprod).sum(dim=1)
            first = ravel_idxs_range[0]
            last = ravel_idxs_range[-1] + n_classes
            ravel_sl = slice(first, last)
            ravel_index = ravel_sl

            if weight is None:
                stitcher.sumview[ravel_index] += values.view(-1)
                stitcher.weightview[ravel_index] += 1.0
            else:
                stitcher.sumview[ravel_index] += (values * weight).view(-1)
                stitcher.weightview[ravel_index] += weight.view(-1)
        else:
            # TODO: maybe check if the input is a tensor?

            shape = stitcher.shape
            n_classes = shape[-1]
            # if we assume we get data in order, its even faster
            if assume_order:
                last = batch_idxs.shape[0] - 1
                base_multi_idxs = tuple(batch_idxs[[0, last]].T)
                # Add extra dimension for output classes
                extra_multi_idxs = np.zeros(2, dtype=np.int)
                multi_idxs_range = base_multi_idxs + (extra_multi_idxs,)
                ravel_idxs_range = np.ravel_multi_index(multi_idxs_range, dims=shape)
                first = ravel_idxs_range[0]
                last = ravel_idxs_range[-1] + n_classes
                ravel_sl = slice(first, last)
                ravel_index = ravel_sl
            else:
                base_multi_idxs = tuple(batch_idxs.T)
                extra_multi_idxs = np.zeros(len(batch_idxs), dtype=np.int)
                # The indices for the 0-th class (which should be the last dimension)
                multi_idxs_first = base_multi_idxs + (extra_multi_idxs,)
                ravel_idxs_first = np.ravel_multi_index(multi_idxs_first, dims=shape)

                # The indices for the next classes should be sequentially after
                all_ravel_idxs = [ravel_idxs_first[None, :]]
                for i in range(1, n_classes):
                    all_ravel_idxs.append((ravel_idxs_first + i)[None, :])
                # raveled indices that correspond with raveled data
                ravel_idxs = np.vstack(all_ravel_idxs).T.ravel()
                # assert np.sum(1 - np.diff(ravel_idxs)), 'we cant assume order'
                ravel_index = ravel_idxs

            if weight is None:
                stitcher.sumview[ravel_index] += values.ravel()
                stitcher.weightview[ravel_index] += 1.0
            else:
                stitcher.sumview[ravel_index] += (values * weight).ravel()
                stitcher.weightview[ravel_index] += np.ravel(weight)

    def average(stitcher):
        """
        Averages out contributions from overlapping adds using weighted average

        Returns:
            out: ndarray: the stitched image
        """
        out = stitcher.sums / stitcher.weights
        return out

    def finalize(stitcher, frame_ids=None):
        """
        Averages out contributions from overlapping adds

        Args:
            frame_ids(None or slice or tuple): if subset is not None, this is
                done for only a region of the larger tensor, otherwise it is
                done for the entire tensor.
                TODO: rename frame_ids subset

        Returns:
            final: ndarray: the stitched image
        """
        if frame_ids is None:
            final = stitcher.sums / stitcher.weights
        else:
            final = stitcher.sums[frame_ids] / stitcher.weights[frame_ids]

        if stitcher.xpu != 'numpy':
            final = final.cpu().numpy()

        final = np.nan_to_num(final)
        return final


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_slider all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
