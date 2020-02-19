import ubelt as ub
from os.path import exists
from os.path import join
from os.path import dirname
from os.path import relpath


def convert_voc_to_coco():
    # TODO: convert segmentation information

    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    devkit_dpath = ensure_voc_data()
    root = out_dpath = dirname(devkit_dpath)

    dsets = []

    d = _convert_voc_split(devkit_dpath, classes, 'train', 2012, root)
    dsets.append(d)

    d = _convert_voc_split(devkit_dpath, classes, 'train', 2007, root)
    dsets.append(d)

    d = _convert_voc_split(devkit_dpath, classes, 'val', 2012, root)
    dsets.append(d)

    d = _convert_voc_split(devkit_dpath, classes, 'val', 2007, root)
    dsets.append(d)

    d = _convert_voc_split(devkit_dpath, classes, 'test', 2007, root)
    dsets.append(d)

    if 0:
        import xdev
        xdev.view_directory(out_dpath)

    def reroot_imgs(dset, root):
        for img in dset.imgs.values():
            img['file_name'] = relpath(img['file_name'], root)

    import ndsampler
    t1 = ndsampler.CocoDataset(join(out_dpath, 'voc-train-2007.mscoco.json'))
    t2 = ndsampler.CocoDataset(join(out_dpath, 'voc-train-2012.mscoco.json'))

    v1 = ndsampler.CocoDataset(join(out_dpath, 'voc-val-2007.mscoco.json'))
    v2 = ndsampler.CocoDataset(join(out_dpath, 'voc-val-2012.mscoco.json'))

    t = ndsampler.CocoDataset.union(t1, t2)
    t.tag = 'voc-train'
    reroot_imgs(t, root)
    t.fpath = join(root, t.tag + '.mscoco.json')
    print('t.fpath = {!r}'.format(t.fpath))
    t.dump(t.fpath, newlines=True)

    v = ndsampler.CocoDataset.union(v1, v2)
    v.tag = 'voc-val'
    reroot_imgs(v, root)
    v.fpath = join(root, v.tag + '.mscoco.json')
    print('v.fpath = {!r}'.format(v.fpath))
    v.dump(v.fpath, newlines=True)

    tv = ndsampler.CocoDataset.union(t1, t2, v1, v2)
    tv.tag = 'voc-trainval'
    reroot_imgs(tv, root)
    tv.fpath = join(root, tv.tag + '.mscoco.json')
    print('tv.fpath = {!r}'.format(tv.fpath))
    tv.dump(tv.fpath, newlines=True)
    if 0:
        tv.img_root = root
        import kwplot
        kwplot.autompl()
        tv.show_image(2)

    dsets = {
        'train': t,
        'vali': v,
        'trainval': tv,
    }
    return dsets


def _convert_voc_split(devkit_dpath, classes, split, year, root):
    """
    split, year = 'train', 2012
    split, year = 'train', 2007
    """
    import ndsampler
    import xml.etree.ElementTree as ET
    dset = ndsampler.CocoDataset(tag='voc-{}-{}'.format(split, year))

    for catname in classes:
        dset.add_category(catname)

    gpaths, apaths = _read_split_paths(devkit_dpath, split, year)

    KNOWN = {'object', 'segmented', 'size', 'source', 'filename', 'folder',
             'owner'}

    for gpath, apath in ub.ProgIter(zip(gpaths, apaths)):
        tree = ET.parse(apath)
        troot = tree.getroot()

        top_level = troot.getchildren()

        unknown = {e.tag for e in top_level} - KNOWN
        assert not unknown

        img = {
            'file_name': relpath(gpath, root),
            'width': int(tree.find('size').find('width').text),
            'height': int(tree.find('size').find('height').text),
            'depth': int(tree.find('size').find('depth').text),
            'segmented': int(tree.find('segmented').text),
            'source': {
                elem.tag: elem.text
                for elem in tree.find('source').getchildren()
            },
        }

        assert img.pop('depth') == 3

        owner = tree.find('owner')
        if owner is not None:
            img['owner'] = {
                elem.tag: elem.text
                for elem in owner.getchildren()
            }

        gid = dset.add_image(**img)

        for obj in tree.findall('object'):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc is None else int(diffc.text)

            catname = obj.find('name').text.lower().strip()
            w = x2 - x1
            h = y2 - y1
            ann = {
                'bbox': [x1, y1, w, h],
                'category_name': catname,
                'difficult': difficult,
                'weight': 1.0 - difficult,
            }
            cid = dset._alias_to_cat(ann.pop('category_name'))['id']
            dset.add_annotation(image_id=gid, category_id=cid, **ann)

    dset.dump(join(root, dset.tag + '.mscoco.json'), newlines=True)
    return dset


def _read_split_paths(devkit_dpath, split, year):
    """
    split = 'train'
    self = VOCDataset('test')
    year = 2007
    year = 2012
    """
    import glob
    import re
    split_idstrs = []
    data_dpath = join(devkit_dpath, 'VOC{}'.format(year))
    split_dpath = join(data_dpath, 'ImageSets', 'Main')
    pattern = join(split_dpath, '*_' + split + '.txt')
    for p in sorted(glob.glob(pattern)):
        rows = [list(re.split(' +', t)) for t in ub.readfrom(p).split('\n') if t]
        # code = -1 if the image does not contain the object
        # code = 1 if the image contains at least one instance
        # code = 0 if the image contains only hard instances of the object
        idstrs = [idstr for idstr, code in rows if int(code) == 1]
        split_idstrs.extend(idstrs)
    split_idstrs = sorted(set(split_idstrs))

    image_dpath = join(data_dpath, 'JPEGImages')
    annot_dpath = join(data_dpath, 'Annotations')
    gpaths = [join(image_dpath, '{}.jpg'.format(idstr))
              for idstr in split_idstrs]
    apaths = [join(annot_dpath, '{}.xml'.format(idstr))
              for idstr in split_idstrs]
    return gpaths, apaths


def ensure_voc_data(dpath=None, force=False, years=[2007, 2012]):
    """
    Download the Pascal VOC 2007 data if it does not already exist.

    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> devkit_dpath = ensure_voc_data()
    """
    if dpath is None:
        dpath = ub.expandpath('~/data/VOC')
    devkit_dpath = join(dpath, 'VOCdevkit')
    # if force or not exists(devkit_dpath):
    ub.ensuredir(dpath)

    fpath1 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar', dpath=dpath)
    if force or not exists(join(dpath, 'VOCdevkit', 'VOCcode')):
        ub.cmd('tar xvf "{}" -C "{}"'.format(fpath1, dpath), verbout=1)

    if 2007 in years:
        # VOC 2007 train+validation data
        fpath2 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', dpath=dpath)
        if force or not exists(join(dpath, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'bird_trainval.txt')):
            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath2, dpath), verbout=1)

        # VOC 2007 test data
        fpath3 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar', dpath=dpath)
        if force or not exists(join(dpath, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'bird_test.txt')):
            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath3, dpath), verbout=1)

    if 2012 in years:
        # VOC 2012 train+validation data
        fpath4 = ub.grabdata('https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar', dpath=dpath)
        if force or not exists(join(dpath, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'bird_trainval.txt')):
            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath4, dpath), verbout=1)
    return devkit_dpath


def main():
    convert_voc_to_coco()


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.data.grab_voc
    """
    main()
