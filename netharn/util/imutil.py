# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
from os.path import expanduser, exists, join, basename
import ubelt as ub  # NOQA
import warnings
import numpy as np
import cv2
import six
# try:
#     # Don't import skimage.io immediately because it imports pyplot
#     # See GH Issue https://github.com/scikit-image/scikit-image/issues/3347
#     import skimage.io
# except ImportError:
#     pass


CV2_INTERPOLATION_TYPES = {
    'nearest': cv2.INTER_NEAREST,
    'linear':  cv2.INTER_LINEAR,
    'area':    cv2.INTER_AREA,
    'cubic':   cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}


def _rectify_interpolation(interp, default=cv2.INTER_LANCZOS4):
    """
    Converts interpolation into flags suitable cv2 functions

    Args:
        interp (int or str): string or cv2-style interpolation type
        default (int): cv2 interpolation flag to use if `interp` is None

    Returns:
        int: flag specifying interpolation type that can be passed to
           functions like cv2.resize, cv2.warpAffine, etc...
    """
    if interp is None:
        return default
    elif isinstance(interp, six.text_type):
        try:
            return CV2_INTERPOLATION_TYPES[interp]
        except KeyError:
            print('Valid values for interpolation are {}'.format(
                list(CV2_INTERPOLATION_TYPES.keys())))
            raise
    else:
        return interp


def imscale(img, scale, interpolation=None):
    """
    Resizes an image by a scale factor.

    Because the result image must have an integer number of pixels, the scale
    factor is rounded, and the rounded scale factor is returnedG

    Args:
        dsize (ndarray): an image
        scale (float or tuple): desired floating point scale factor
    """
    dsize = img.shape[0:2][::-1]
    try:
        sx, sy = scale
    except TypeError:
        sx = sy = scale
    w, h = dsize
    new_w = int(round(w * sx))
    new_h = int(round(h * sy))
    new_scale = new_w / w, new_h / h
    new_dsize = (new_w, new_h)

    interpolation = _rectify_interpolation(interpolation)
    new_img = cv2.resize(img, new_dsize, interpolation=interpolation)
    return new_img, new_scale


def adjust_gamma(img, gamma=1.0):
    """
    gamma correction function

    References:
        http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    Ignore:
        >>> from netharn import util
        >>> fpath = util.grab_test_image()
        >>> img = util.imread(fpath)
        >>> gamma = .5
        >>> imgf = ensure_float01(img)
        >>> img2 = adjust_gamma(img, gamma)
        >>> img3 = adjust_gamma(imgf, gamma)
        >>> import plottool as pt
        >>> pt.qtensure()
        >>> pt.imshow(img, pnum=(3, 3, 1), fnum=1)
        >>> pt.imshow(img2, pnum=(3, 3, 2), fnum=1)
        >>> pt.imshow(img3, pnum=(3, 3, 3), fnum=1)
        >>> pt.imshow(adjust_gamma(img, 1), pnum=(3, 3, 5), fnum=1)
        >>> pt.imshow(adjust_gamma(imgf, 1), pnum=(3, 3, 6), fnum=1)
        >>> pt.imshow(adjust_gamma(img, 2), pnum=(3, 3, 8), fnum=1)
        >>> pt.imshow(adjust_gamma(imgf, 2), pnum=(3, 3, 9), fnum=1)
    """
    if img.dtype.kind in ('i', 'u'):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        table = (((np.arange(0, 256) / 255.0) ** (1 / gamma)) * 255).astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)
        ]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)
    else:
        np_img = ensure_float01(img, copy=False)
        gain = 1
        np_img = gain * (np_img ** (1 / gamma))
        np_img = np.clip(np_img, 0, 1)
        return np_img


def ensure_float01(img, dtype=np.float32, copy=True):
    """ Ensure that an image is encoded using a float properly """
    if img.dtype.kind in ('i', 'u'):
        assert img.max() <= 255
        img_ = img.astype(dtype, copy=copy) / 255.0
    else:
        img_ = img.astype(dtype, copy=copy)
    return img_


def make_channels_comparable(img1, img2):
    """
    Broadcasts image arrays so they can have elementwise operations applied

    CommandLine:
        python -m netharn.util.imutil make_channels_comparable

    Example:
        >>> import itertools as it
        >>> wh_basis = [(5, 5), (3, 5), (5, 3), (1, 1), (1, 3), (3, 1)]
        >>> for w, h in wh_basis:
        >>>     shape_basis = [(w, h), (w, h, 1), (w, h, 3)]
        >>>     # Test all permutations of shap inputs
        >>>     for shape1, shape2 in it.product(shape_basis, shape_basis):
        >>>         print('*    input shapes: %r, %r' % (shape1, shape2))
        >>>         img1 = np.empty(shape1)
        >>>         img2 = np.empty(shape2)
        >>>         img1, img2 = make_channels_comparable(img1, img2)
        >>>         print('... output shapes: %r, %r' % (img1.shape, img2.shape))
        >>>         elem = (img1 + img2)
        >>>         print('... elem(+) shape: %r' % (elem.shape,))
        >>>         assert elem.size == img1.size, 'outputs should have same size'
        >>>         assert img1.size == img2.size, 'new imgs should have same size'
        >>>         print('--------')
    """
    if img1.shape != img2.shape:
        c1 = get_num_channels(img1)
        c2 = get_num_channels(img2)
        if len(img1.shape) == 2 and len(img2.shape) == 2:
            raise AssertionError('UNREACHABLE: Both are 2-grayscale')
        elif len(img1.shape) == 3 and len(img2.shape) == 2:
            # Image 2 is grayscale
            if c1 == 3:
                img2 = np.tile(img2[..., None], 3)
            else:
                img2 = img2[..., None]
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            # Image 1 is grayscale
            if c2 == 3:
                img1 = np.tile(img1[..., None], 3)
            else:
                img1 = img1[..., None]
        elif len(img1.shape) == 3 and len(img2.shape) == 3:
            # Both images have 3 dims.
            # Check if either have color, then check for alpha
            if c1 == 1 and c2 == 1:
                # raise AssertionError('UNREACHABLE: Both are 3-grayscale')
                pass
            elif c1 == 4 and c2 == 4:
                # raise AssertionError('UNREACHABLE: Both are 3-alpha')
                pass
            elif c1 == 3 and c2 == 3:
                # raise AssertionError('UNREACHABLE: Both are 3-color')
                pass
            elif c1 == 1 and c2 == 3:
                img1 = np.tile(img1, 3)
            elif c1 == 3 and c2 == 1:
                img2 = np.tile(img2, 3)
            elif c1 == 1 and c2  == 4:
                img1 = np.dstack((np.tile(img1, 3), _alpha_fill_for(img1)))
            elif c1 == 4 and c2  == 1:
                img2 = np.dstack((np.tile(img2, 3), _alpha_fill_for(img2)))
            elif c1 == 3 and c2  == 4:
                img1 = np.dstack((img1, _alpha_fill_for(img1)))
            elif c1 == 4 and c2  == 3:
                img2 = np.dstack((img2, _alpha_fill_for(img2)))
            else:
                raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
        else:
            raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
    return img1, img2


def _alpha_fill_for(img):
    """ helper for make_channels_comparable """
    fill_value = (255 if img.dtype.kind in ('i', 'u') else 1)
    alpha_chan = np.full(img.shape[0:2], dtype=img.dtype,
                         fill_value=fill_value)
    return alpha_chan


def get_num_channels(img):
    """ Returns the number of color channels """
    ndims = len(img.shape)
    if ndims == 2:
        nChannels = 1
    elif ndims == 3 and img.shape[2] == 3:
        nChannels = 3
    elif ndims == 3 and img.shape[2] == 4:
        nChannels = 4
    elif ndims == 3 and img.shape[2] == 1:
        nChannels = 1
    else:
        raise ValueError('Cannot determine number of channels '
                         'for img.shape={}'.format(img.shape))
    return nChannels


def overlay_alpha_images(img1, img2, keepalpha=True):
    """
    places img1 on top of img2 respecting alpha channels

    Args:
        img1 (ndarray): top image to overlay over img2
        img2 (ndarray): base image to superimpose on
        keepalpha (bool): if False, the alpha channel is removed after blending

    References:
        http://stackoverflow.com/questions/25182421/overlay-numpy-alpha
        https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
    """
    # img1, img2 = make_channels_comparable(img1, img2)
    rgb1, alpha1 = _prep_rgb_alpha(img1)
    rgb2, alpha2 = _prep_rgb_alpha(img2)

    # Perform the core alpha blending algorithm
    # rgb3, alpha3 = _alpha_blend_core(rgb1, alpha1, rgb2, alpha2)
    rgb3, alpha3 = _alpha_blend_fast1(rgb1, alpha1, rgb2, alpha2)

    if keepalpha:
        img3 = np.dstack([rgb3, alpha3[..., None]])
        # Note: if we want to output a 255 img we could do something like this
        # out = np.zeros_like(img1)
        # out[..., :3] = rgb3
        # out[..., 3] = alpha3
    else:
        img3 = rgb3
    return img3


def _prep_rgb_alpha(img):
    img = ensure_float01(img, copy=False)
    img = atleast_3channels(img, copy=False)

    c = get_num_channels(img)

    if c == 4:
        # rgb = np.ascontiguousarray(img[..., 0:3])
        # alpha = np.ascontiguousarray(img[..., 3])
        rgb = img[..., 0:3]
        alpha = img[..., 3]
    else:
        rgb = img
        alpha = np.ones_like(img[..., 0])
    return rgb, alpha


def _alpha_blend_fast1(rgb1, alpha1, rgb2, alpha2):
    """
    Uglier but faster version of the core alpha blending algorithm using
    preallocation and in-place computation where possible.

    Example:
        >>> rng = np.random.RandomState(0)
        >>> rgb1, rgb2 = rng.rand(10, 10, 3), rng.rand(10, 10, 3)
        >>> alpha1, alpha2 = rng.rand(10, 10), rng.rand(10, 10)
        >>> f1, f2 = _alpha_blend_fast1(rgb1, alpha1, rgb2, alpha2)
        >>> s1, s2 = _alpha_blend_core(rgb1, alpha1, rgb2, alpha2)
        >>> assert np.all(f1 == s1) and np.all(f2 == s2)
        >>> alpha1, alpha2 = np.zeros((10, 10)), np.zeros((10, 10))
        >>> f1, f2 = _alpha_blend_fast1(rgb1, alpha1, rgb2, alpha2)
        >>> s1, s2 = _alpha_blend_core(rgb1, alpha1, rgb2, alpha2)
        >>> assert np.all(f1 == s1) and np.all(f2 == s2)

    _ = profiler.profile_onthefly(overlay_alpha_images)(img1, img2)
    _ = profiler.profile_onthefly(_prep_rgb_alpha)(img1)
    _ = profiler.profile_onthefly(_prep_rgb_alpha)(img2)

    _ = profiler.profile_onthefly(_alpha_blend_core)(rgb1, alpha1, rgb2, alpha2)
    _ = profiler.profile_onthefly(_alpha_blend_fast1)(rgb1, alpha1, rgb2, alpha2)
    """
    rgb3 = np.empty_like(rgb1)
    temp_rgb = np.empty_like(rgb1)
    alpha3 = np.empty_like(alpha1)
    temp_alpha = np.empty_like(alpha1)

    # hold (1 - alpha1)
    np.subtract(1, alpha1, out=temp_alpha)

    # alpha3
    np.copyto(dst=alpha3, src=temp_alpha)
    np.multiply(alpha2, alpha3, out=alpha3)
    np.add(alpha1, alpha3, out=alpha3)

    # numer1
    np.multiply(rgb1, alpha1[..., None], out=rgb3)

    # numer2
    np.multiply(alpha2, temp_alpha, out=temp_alpha)
    np.multiply(rgb2, temp_alpha[..., None], out=temp_rgb)

    # (numer1 + numer2)
    np.add(rgb3, temp_rgb, out=rgb3)

    # removing errstate is actually a significant speedup
    # with np.errstate(invalid='ignore'):
    np.divide(rgb3, alpha3[..., None], out=rgb3)
    if not np.all(alpha3):
        rgb3[alpha3 == 0] = 0
    return rgb3, alpha3


def _alpha_blend_core(rgb1, alpha1, rgb2, alpha2):
    """
    Core alpha blending algorithm

    SeeAlso:
        _alpha_blend_fast1
    """
    c_alpha1 = (1.0 - alpha1)
    alpha3 = alpha1 + alpha2 * c_alpha1

    numer1 = (rgb1 * alpha1[..., None])
    numer2 = (rgb2 * (alpha2 * c_alpha1)[..., None])
    with np.errstate(invalid='ignore'):
        rgb3 = (numer1 + numer2) / alpha3[..., None]
    rgb3[alpha3 == 0] = 0
    return rgb3, alpha3


def ensure_alpha_channel(img, alpha=1.0):
    img = ensure_float01(img, copy=False)
    c = get_num_channels(img)
    if c == 4:
        return img
    else:
        alpha_channel = np.full(img.shape[0:2], fill_value=alpha, dtype=img.dtype)
        if c == 3:
            return np.dstack([img, alpha_channel])
        elif c == 1:
            return np.dstack([img, img, img, alpha_channel])
        else:
            raise ValueError('unknown dim')


def atleast_3channels(arr, copy=True):
    r"""
    Ensures that there are 3 channels in the image

    Args:
        arr (ndarray[N, M, ...]): the image
        copy (bool): Always copies if True, if False, then copies only when the
            size of the array must change.

    Returns:
        ndarray: with shape (N, M, C), where C in {3, 4}

    Doctest:
        >>> assert atleast_3channels(np.zeros((10, 10))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 1))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 3))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 4))).shape[-1] == 4
    """
    ndims = len(arr.shape)
    if ndims == 2:
        res = np.tile(arr[:, :, None], 3)
        return res
    elif ndims == 3:
        h, w, c = arr.shape
        if c == 1:
            res = np.tile(arr, 3)
        elif c in [3, 4]:
            res = arr.copy() if copy else arr
        else:
            raise ValueError('Cannot handle ndims={}'.format(ndims))
    else:
        raise ValueError('Cannot handle arr.shape={}'.format(arr.shape))
    return res


def ensure_grayscale(img, colorspace_hint='BGR'):
    img = ensure_float01(img, copy=False)
    c = get_num_channels(img)
    if c == 1:
        return img
    else:
        return convert_colorspace(img, 'gray', colorspace_hint)


def convert_colorspace(img, dst_space, src_space='BGR', copy=False, dst=None):
    """
    Converts colorspace of img.
    Convinience function around cv2.cvtColor

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        colorspace (str): RGB, LAB, etc
        dst_space (unicode): (default = u'BGR')

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    Note:
        Note the LAB and HSV colorspaces in float do not go into the 0-1 range.

        For HSV the floating point range is:
            0:360, 0:1, 0:1
        For LAB the floating point range is:
            0:100, -86.1875:98.234375, -107.859375:94.46875
            (Note, that some extreme combinations of a and b are not valid)

    Example:
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[0, 1, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[1, 0, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[1, 1, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'HSV', src_space='RGB')

    Ignore:
        # Check LAB output ranges
        import itertools as it
        s = 1
        _iter = it.product(range(0, 256, s), range(0, 256, s), range(0, 256, s))
        minvals = np.full(3, np.inf)
        maxvals = np.full(3, -np.inf)
        for r, g, b in ub.ProgIter(_iter, total=(256 // s) ** 3):
            img255 = np.array([[[r, g, b]]], dtype=np.uint8)
            img01 = (img255 / 255.0).astype(np.float32)
            lab = convert_colorspace(img01, 'lab', src_space='rgb')
            np.minimum(lab[0, 0], minvals, out=minvals)
            np.maximum(lab[0, 0], maxvals, out=maxvals)
        print('minvals = {}'.format(ub.repr2(minvals, nl=0)))
        print('maxvals = {}'.format(ub.repr2(maxvals, nl=0)))
    """
    dst_space = dst_space.upper()
    src_space = src_space.upper()
    if src_space == dst_space:
        img2 = img
        if dst is not None:
            dst[...] = img[...]
            img2 = dst
        elif copy:
            img2 = img2.copy()
    else:
        code = _lookup_colorspace_code(dst_space, src_space)
        # Note the conversion to colorspaces like LAB and HSV in float form
        # do not go into the 0-1 range. Instead they go into
        # (0-100, -111-111ish, -111-111ish) and (0-360, 0-1, 0-1) respectively
        img2 = cv2.cvtColor(img, code, dst=dst)
    return img2


def _lookup_colorspace_code(dst_space, src_space='BGR'):
    src = src_space.upper()
    dst = dst_space.upper()
    convert_attr = 'COLOR_{}2{}'.format(src, dst)
    if not hasattr(cv2, convert_attr):
        prefix = 'COLOR_{}2'.format(src)
        valid_dst_spaces = [
            key.replace(prefix, '')
            for key in cv2.__dict__.keys() if key.startswith(prefix)]
        raise KeyError(
            '{} does not exist, valid conversions from {} are to {}'.format(
                convert_attr, src_space, valid_dst_spaces))
    else:
        code = getattr(cv2, convert_attr)
    return code


def overlay_colorized(colorized, orig, alpha=.6, keepcolors=False):
    """
    Overlays a color segmentation mask on an original image

    Args:
        colorized (ndarray): the color mask to be overlayed on top of the original image
        orig (ndarray): the original image to superimpose on
        alpha (float): blend level to use if colorized is not an alpha image

    """
    color_mask = ensure_alpha_channel(colorized, alpha=alpha)
    if not keepcolors:
        orig = ensure_grayscale(orig)
    color_blend = overlay_alpha_images(color_mask, orig)
    color_blend = (color_blend * 255).astype(np.uint8)
    return color_blend


def load_image_paths(dpath, ext=('.png', '.tiff', 'tif')):
    dpath = expanduser(dpath)
    if not exists(dpath):
        raise ValueError('dpath = {} does not exist'.format(dpath))
    if not isinstance(ext, (list, tuple)):
        ext = [ext]

    image_paths = []
    for ext_ in ext:
        image_paths.extend(list(glob.glob(join(dpath, '*' + ext_))))
    # potentially non-general
    # (utilfname solves this though)
    image_paths = sorted(image_paths, key=basename)
    return image_paths


def wide_strides_1d(margin, stop, step=None, start=0, keepbound=False,
                    check=True):
    """
    Helper for `image_slices`. Generates slices in a single dimension.

    Args:
        start (int): starting point (in most cases set this to 0)

        margin (int): the length of the slice (window)

        stop (int): the length of the image dimension

        step (int): the length of each step / distance between slices

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

        check (bool): if True an error will be raised if the window does not
            cover the entire extent from start to stop, even if keepbound is
            True.

    Yields:
        slice : slice in one dimension of size (margin)

    Example:
        >>> stop, margin, step = 2000, 360, 360
        >>> keepbound = True
        >>> strides = list(wide_strides_1d(margin, stop, step, keepbound, check=False))
        >>> assert all([(s.stop - s.start) == margin for s in strides])

    Example:
        >>> stop, margin, step = 200, 46, 7
        >>> keepbound = True
        >>> strides = list(wide_strides_1d(margin, stop, step, keepbound=False, check=True))
        >>> starts = np.array([s.start for s in strides])
        >>> stops = np.array([s.stop for s in strides])
        >>> widths = stops - starts
        >>> assert np.all(np.diff(starts) == step)
        >>> assert np.all(widths == margin)

    Example:
        >>> import pytest
        >>> stop, margin, step = 200, 36, 7
        >>> with pytest.raises(ValueError):
        ...     list(wide_strides_1d(margin, stop, step))
    """
    if step is None:
        step = margin

    if check:
        # see how far off the end we would fall if we didnt check bounds
        perfect_final_pos = (stop - start - margin)
        overshoot = perfect_final_pos % step
        if overshoot > 0:
            raise ValueError(
                ('margin={} and step={} overshoot endpoint={} '
                 'by {} units when starting from={}').format(
                     margin, step, stop, overshoot, start))
    pos = start
    # probably could be more efficient with numpy here
    while True:
        endpos = pos + margin
        yield slice(pos, endpos)
        # Stop once we reached the end
        if endpos == stop:
            break
        pos += step
        if pos + margin > stop:
            if keepbound:
                # Ensure the boundary is always used even if steps
                # would overshoot Could do some other strategy here
                pos = stop - margin
            else:
                break


def image_slices(img_shape, target_shape, overlap=0, keepbound=False):
    """
    Generates "sliding window" slices to break a large image into smaller
    pieces.

    Args:
        img_shape (tuple): height and width of the image

        target_shape (tuple): (height, width) of the

        overlap (float): a number between 0 and 1 indicating the fraction of
            overlap that parts will have. Must be `0 <= overlap < 1`.

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

    Yields:
        tuple(slice, slice): row and column slices used for numpy indexing

    Example:
        >>> img_shape = (2000, 2000)
        >>> target_shape = (360, 480)
        >>> overlap = 0
        >>> keepbound = True
        >>> list(image_slices(img_shape, target_shape, overlap, keepbound))
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError(('part overlap was {}, but it must be '
                          'in the range [0, 1)').format(overlap))
    ph, pw = target_shape
    sy = int(ph - ph * overlap)
    sx = int(pw - pw * overlap)
    orig_h, orig_w = img_shape
    kw = dict(keepbound=keepbound, check=False, start=0)
    for rslice in wide_strides_1d(ph, orig_h, sy, **kw):
        for cslice in wide_strides_1d(pw, orig_w, sx, **kw):
            yield rslice, cslice


def run_length_encoding(img):
    """
    Run length encoding.

    Parameters
    ----------
    img : 2D image

    Example:
        >>> lines = ub.codeblock(
        >>>     '''
        >>>     ..........
        >>>     ......111.
        >>>     ..2...111.
        >>>     .222..111.
        >>>     22222.....
        >>>     .222......
        >>>     ..2.......
        >>>     ''').replace('.', '0').splitlines()
        >>> img = np.array([list(map(int, line)) for line in lines])
        >>> (w, h), runlen = run_length_encoding(img)
        >>> target = np.array([0,16,1,3,0,3,2,1,0,3,1,3,0,2,2,3,0,2,1,3,0,1,2,5,0,6,2,3,0,8,2,1,0,7])
        >>> assert np.all(target == runlen)
    """
    flat = img.ravel()
    diff_idxs = np.flatnonzero(np.abs(np.diff(flat)) > 0)
    pos = np.hstack([[0], diff_idxs + 1])

    values = flat[pos]
    lengths = np.hstack([np.diff(pos), [len(flat) - pos[-1]]])

    runlen = np.hstack([values[:, None], lengths[:, None]]).ravel()

    h, w = img.shape[0:2]
    return (w, h), runlen


def imread(fpath, **kw):
    """
    reads image data in BGR format

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> from os.path import splitext  # NOQA
        >>> fpath = ub.grabdata('https://i.imgur.com/oHGsmvF.png', fname='carl.png')
        >>> fpath = ub.grabdata('http://www.topcoder.com/contest/problem/UrbanMapper3D/JAX_Tile_043_DTM.tif')
        >>> ext = splitext(fpath)[1]
        >>> img1 = imread(fpath)
        >>> # Check that write + read preserves data
        >>> tmp = tempfile.NamedTemporaryFile(suffix=ext)
        >>> imwrite(tmp.name, img1)
        >>> img2 = imread(tmp.name)
        >>> assert np.all(img2 == img1)

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> img1 = imread(ub.grabdata('http://i.imgur.com/iXNf4Me.png', fname='ada.png'))
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> import tempfile
        >>> #img1 = (np.arange(0, 12 * 12 * 3).reshape(12, 12, 3) % 255).astype(np.uint8)
        >>> tif_fpath = ub.grabdata('https://ghostscript.com/doc/tiff/test/images/rgb-3c-16b.tiff')
        >>> img1 = imread(tif_fpath)
        >>> tmp_tif = tempfile.NamedTemporaryFile(suffix='.tif')
        >>> tmp_png = tempfile.NamedTemporaryFile(suffix='.png')
        >>> imwrite(tmp_tif.name, img1)
        >>> imwrite(tmp_png.name, img1)
        >>> tif_im = imread(tmp_tif.name)
        >>> png_im = imread(tmp_png.name)
        >>> assert np.all(tif_im == png_im)

        import plottool as pt
        pt.qtensure()
        pt.imshow(tif_im / 2 ** 16, pnum=(1, 2, 1), fnum=1)
        pt.imshow(png_im / 2 ** 16, pnum=(1, 2, 2), fnum=1)

    Ignore:
        from PIL import Image
        pil_img = Image.open(tif_fpath)
        assert int(Image.PILLOW_VERSION.split('.')[0]) > 4
    """
    try:
        if fpath.lower().endswith(('.ntf', '.nitf')):
            try:
                import gdal
            except ImportError:
                raise Exception('cannot read NITF images without gdal')
            try:
                gdal_dset = gdal.Open(fpath)
                if gdal_dset.RasterCount == 1:
                    band = gdal_dset.GetRasterBand(1)
                    image = np.array(band.ReadAsArray())
                elif gdal_dset.RasterCount == 3:
                    bands = [
                        gdal_dset.GetRasterBand(i)
                        for i in [1, 2, 3]
                    ]
                    channels = [np.array(band.ReadAsArray()) for band in bands]
                    image = np.dstack(channels)
                else:
                    raise NotImplementedError(
                        'Can only read 1 or 3 channel NTF images. '
                        'Got {}'.format(gdal_dset.RasterCount))
            except Exception:
                raise
            finally:
                gdal_dset = None
        elif fpath.lower().endswith(('.tif', '.tiff')):
            import skimage.io
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # skimage reads in RGB, convert to BGR
                image = skimage.io.imread(fpath, **kw)
                if get_num_channels(image) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif get_num_channels(image) == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        else:
            image = cv2.imread(fpath, flags=cv2.IMREAD_UNCHANGED)
            if image is None:
                if exists(fpath):
                    raise IOError('OpenCV cannot read this image: "{}", '
                                  'but it exists'.format(fpath))
                else:
                    raise IOError('OpenCV cannot read this image: "{}", '
                                  'because it does not exist'.format(fpath))
        return image
    except Exception as ex:
        print('Error reading fpath = {!r}'.format(fpath))
        raise


def imwrite(fpath, image, **kw):
    """
    writes image data in BGR format
    """
    if fpath.endswith(('.tif', '.tiff')):
        import skimage.io
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skimage writes in RGB, convert from BGR
            if get_num_channels(image) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif get_num_channels(image) == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return skimage.io.imsave(fpath, image)
    else:
        return cv2.imwrite(fpath, image)


def stack_multiple_images(images, axis=0, resize=None, interpolation=None, overlap=0):
    img1 = images[0]
    for img2 in images[1:]:
        img1 = stack_images(img1, img2, axis=axis, resize=resize,
                            interpolation=interpolation, overlap=overlap)[0]
    return img1


def stack_images(img1, img2, axis=0, resize=None, interpolation=None,
                 overlap=0):
    """
    Make a new image with the input images side-by-side

    Args:
        img1 (ndarray[ndim=2]):  image data
        img2 (ndarray[ndim=2]):  image data
        axis (int): axis to stack on (either 0 or 1)
        resize (int, str, or None): if None image sizes are not modified,
            otherwise resize resize can be either 0 or 1.  We resize the
            `resize`-th image to match the `1 - resize`-th image. Can
            also be strings "larger" or "smaller".
        interpolation (int or str): string or cv2-style interpolation type.
            only used if resize or overlap > 0
        overlap (int): number of pixels to overlap. Using a negative
            number results in a border.

    Returns:
        Tuple[ndarray, Tuple, Tuple]: imgB, offset_tup, sf_tup

    Example:
        >>> from netharn import util
        >>> img1 = util.grab_test_image('carl', space='bgr')
        >>> img2 = util.grab_test_image('astro', space='bgr')
        >>> imgB, offs, sfs = stack_images(img1, img2, axis=0, resize=0,
        >>>                                overlap=-10)
        >>> woff, hoff = offs
        >>> # verify results
        >>> result = str((imgB.shape, woff, hoff))
        >>> print(result)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import netharn as nh
        >>> nh.util.autompl()
        >>> nh.util.imshow(imgB, colorspace='bgr')
        >>> wh1 = np.multiply(img1.shape[0:2][::-1], sfs[0])
        >>> wh2 = np.multiply(img2.shape[0:2][::-1], sfs[1])
        >>> nh.util.draw_boxes([(0, 0, wh1[0], wh1[1])], 'xywh', color=(1.0, 0, 0))
        >>> nh.util.draw_boxes([(woff[1], hoff[1], wh2[0], wh2[0])], 'xywh', color=(0, 1.0, 0))
        >>> nh.util.show_if_requested()
        ((662, 512, 3), (0.0, 0.0), (0, 150))
    """

    def _rectify_axis(img1, img2, axis):
        """ determine if we are stacking in horzontally or vertically """
        (h1, w1) = img1.shape[0: 2]  # get chip dimensions
        (h2, w2) = img2.shape[0: 2]
        woff, hoff = 0, 0
        vert_wh  = max(w1, w2), h1 + h2
        horiz_wh = w1 + w2, max(h1, h2)
        if axis is None:
            # Display the orientation with the better (closer to 1) aspect ratio
            vert_ar  = max(vert_wh) / min(vert_wh)
            horiz_ar = max(horiz_wh) / min(horiz_wh)
            axis = 0 if vert_ar < horiz_ar else 1
        if axis == 0:  # vertical stack
            wB, hB = vert_wh
            hoff = h1
        elif axis == 1:
            wB, hB = horiz_wh
            woff = w1
        else:
            raise ValueError('axis can only be 0 or 1')
        return axis, h1, h2, w1, w2, wB, hB, woff, hoff

    def _round_dsize(dsize, scale):
        """
        Returns an integer size and scale that best approximates
        the floating point scale on the original size

        Args:
            dsize (tuple): original width height
            scale (float or tuple): desired floating point scale factor
        """
        try:
            sx, sy = scale
        except TypeError:
            sx = sy = scale
        w, h = dsize
        new_w = int(round(w * sx))
        new_h = int(round(h * sy))
        new_scale = new_w / w, new_h / h
        new_dsize = (new_w, new_h)
        return new_dsize, new_scale

    def _ramp(shape, axis):
        """ nd ramp function """
        newshape = [1] * len(shape)
        reps = list(shape)
        newshape[axis] = -1
        reps[axis] = 1
        basis = np.linspace(0, 1, shape[axis])
        data = basis.reshape(newshape)
        return np.tile(data, reps)

    def _blend(part1, part2, alpha):
        """ blending based on an alpha mask """
        part1, alpha = make_channels_comparable(part1, alpha)
        part2, alpha = make_channels_comparable(part2, alpha)
        partB = (part1 * (1.0 - alpha)) + (part2 * (alpha))
        return partB

    interpolation = _rectify_interpolation(interpolation, default=cv2.INTER_NEAREST)

    img1, img2 = make_channels_comparable(img1, img2)
    nChannels = get_num_channels(img1)

    assert img1.dtype == img2.dtype, (
        'img1.dtype=%r, img2.dtype=%r' % (img1.dtype, img2.dtype))

    axis, h1, h2, w1, w2, wB, hB, woff, hoff = _rectify_axis(img1, img2, axis)

    # allow for some overlap / blending of the images
    if overlap:
        if axis == 0:
            hB -= overlap
        else:
            wB -= overlap
    # Rectify both images to they are the same dimension
    if resize:
        # Compre the lengths of the width and height
        length1 = img1.shape[1 - axis]
        length2 = img2.shape[1 - axis]
        if resize == 'larger':
            resize = 0 if length1 > length2 else 1
        elif resize == 'smaller':
            resize = 0 if length1 < length2 else 1
        if resize == 0:
            tonew_sf2 = (1., 1.)
            scale = length2 / length1
            dsize, tonew_sf1 = _round_dsize(img1.shape[0:2][::-1], scale)
            img1 = cv2.resize(img1, dsize, interpolation=interpolation)
        elif resize == 1:
            tonew_sf1 = (1., 1.)
            scale = length1 / length2
            dsize, tonew_sf2 = _round_dsize(img2.shape[0:2][::-1], scale)
            img2 = cv2.resize(img2, dsize, interpolation=interpolation)
        else:
            raise ValueError('resize can only be 0 or 1')
        axis, h1, h2, w1, w2, wB, hB, woff, hoff = _rectify_axis(img1, img2, axis)
    else:
        tonew_sf1 = (1., 1.)
        tonew_sf2 = (1., 1.)

    # Do image concatentation
    if nChannels > 1 or len(img1.shape) > 2:
        newshape = (hB, wB, nChannels)
    else:
        newshape = (hB, wB)
    # Allocate new image for both
    imgB = np.zeros(newshape, dtype=img1.dtype)

    # Insert the images in the larger frame
    if overlap:
        if axis == 0:
            hoff -= overlap
        elif axis == 1:
            woff -= overlap
        # Insert the images
        imgB[0:h1, 0:w1] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2
        if overlap > 0:
            # Blend the overlapping part
            if axis == 0:
                part1 = img1[-overlap:, :]
                part2 = imgB[hoff:(hoff + overlap), 0:w1]
                alpha = _ramp(part1.shape[0:2], axis=axis)
                blended = _blend(part1, part2, alpha)
                imgB[hoff:(hoff + overlap), 0:w1] = blended
            elif axis == 1:
                part1 = img1[:, -overlap:]
                part2 = imgB[0:h1, woff:(woff + overlap)]
                alpha = _ramp(part1.shape[0:2], axis=axis)
                blended = _blend(part1, part2, alpha)
                imgB[0:h1, woff:(woff + overlap)] = blended
    else:
        imgB[0:h1, 0:w1] = img1
        imgB[hoff:(hoff + h2), woff:(woff + w2)] = img2

    offset1 = (0.0, 0.0)
    offset2 = (woff, hoff)
    offset_tup = (offset1, offset2)
    sf_tup = (tonew_sf1, tonew_sf2)
    return imgB, offset_tup, sf_tup


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.util.imutil all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
