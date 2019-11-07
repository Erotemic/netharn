# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
from os.path import expanduser, exists, join, basename
import ubelt as ub
import warnings
import numpy as np
import cv2
import six
try:
    import skimage.io
except ImportError:
    pass


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


def grab_test_imgpath(key='carl'):
    assert key == 'carl'
    fpath = ub.grabdata('https://i.imgur.com/oHGsmvF.png', fname='carl.png')
    return fpath


def adjust_gamma(img, gamma=1.0):
    """
    gamma correction function

    References:
        http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    Ignore:
        >>> from netharn import util
        >>> fpath = grab_test_imgpath()
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
        >>> # DISABLE_DOCTEST
        >>> wh_basis = [(5, 5), (3, 5), (5, 3), (1, 1), (1, 3), (3, 1)]
        >>> for w, h in wh_basis:
        >>>     shape_basis = [(w, h), (w, h, 1), (w, h, 3)]
        >>>     # Test all permutations of shap inputs
        >>>     for shape1, shape2 in ut.product(shape_basis, shape_basis):
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
                raise AssertionError('UNREACHABLE: Both are 3-grayscale')
            elif c1 == 3 and c2 == 3:
                raise AssertionError('UNREACHABLE: Both are 3-color')
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

    with np.errstate(invalid='ignore'):
        np.divide(rgb3, alpha3[..., None], out=rgb3)
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
    r"""
    Converts colorspace of img.
    Convinience function around cv2.cvtColor

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data
        colorspace (str): RGB, LAB, etc
        dst_space (unicode): (default = u'BGR')

    Returns:
        ndarray[uint8_t, ndim=2]: img -  image data

    Example:
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[0, 1, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[1, 0, 0]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[1, 1, 1]]], dtype=np.float32), 'LAB', src_space='RGB')
        >>> convert_colorspace(np.array([[[0, 0, 1]]], dtype=np.float32), 'HSV', src_space='RGB')
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
        # (0-100, -111-111hs, -111-111is) and (0-360, 0-1, 0-1) respectively
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
        if fpath.endswith(('.tif', '.tiff')):
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.util.imutil all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
