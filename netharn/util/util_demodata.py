import ubelt as ub


def grab_test_image(key='astro', space='rgb'):
    from netharn.util import convert_colorspace
    import cv2
    if key == 'astro':
        url = 'https://i.imgur.com/KXhKM72.png'
    elif key == 'carl':
        url = 'https://i.imgur.com/oHGsmvF.png'
    else:
        raise KeyError(key)
    fpath = ub.grabdata(url)
    bgr = cv2.imread(fpath)
    image = convert_colorspace(bgr, space, src_space='bgr')
    return image
