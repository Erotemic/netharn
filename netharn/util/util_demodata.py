# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import ubelt as ub


_TEST_IMAGES = {
    'astro': {
        'url': 'https://i.imgur.com/KXhKM72.png',
        'sha1': 'de64fcb37e67d5b5946ee45eb659436',
    },
    'carl': {
        'url': 'https://i.imgur.com/flTHWFD.png',
        'sha1': '12501af6b0e49567fc3df4c5316f99',
    },
    'stars': {
        'url': 'https://i.imgur.com/kCi7C1r.png',
        'sha1': 'e19e0c0c28c67441700cf272cb6ae',
    },
    'paraview': {
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/46/ParaView_splash1.png',
        'sha1': '25e92fe7661c0d9caf8eb919f6a9e76',
    },
}


def grab_test_image(key='astro', space='rgb'):
    """
    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky

        space (str): which colorspace to return in (defaults to RGB)

    Example:
        >>> for key in grab_test_image.keys():
        ...     grab_test_image(key)
    """
    from netharn.util import convert_colorspace
    import cv2
    fpath = grab_test_image_fpath(key)
    bgr = cv2.imread(fpath)
    image = convert_colorspace(bgr, space, src_space='bgr')
    return image


def grab_test_image_fpath(key='astro'):
    """
    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky

    Example:
        >>> for key in grab_test_image.keys():
        ...     grab_test_image_fpath(key)
    """
    try:
        item = _TEST_IMAGES[key]
    except KeyError:
        valid_keys = sorted(_TEST_IMAGES.keys())
        raise KeyError(
            'Unknown key={!r}. Valid keys are {!r}'.format(
                key, valid_keys))
    if not isinstance(item, dict):
        item = {'url': item}

    if 'sha1' in item:
        fpath = ub.grabdata(item['url'], hash_prefix=item['sha1'],
                            appname='netharn')
    else:
        fpath = ub.grabdata(item['url'], appname='netharn')
    return fpath

grab_test_image.keys = lambda: _TEST_IMAGES.keys()
grab_test_image_fpath.keys = lambda: _TEST_IMAGES.keys()


# def grab_test_image_fpath(key='carl'):
#     assert key == 'carl'
#     fpath = ub.grabdata('https://i.imgur.com/oHGsmvF.png', fname='carl.png')
#     return fpath
