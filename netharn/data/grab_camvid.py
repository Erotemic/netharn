from os.path import relpath
from os.path import exists
from os.path import join
import ubelt as ub


def _devcheck_sample_full_image():
    """
    """
    import kwimage
    import numpy as np

    sampler = grab_camvid_sampler()

    cid_to_cidx = sampler.catgraph.id_to_idx
    classes = sampler.catgraph

    # Try loading an entire image
    img, annots = sampler.load_image_with_annots(1)

    file = img['imdata']
    imdata = file[:]

    aids = [ann['id'] for ann in annots]
    _annots = sampler.dset.annots(aids)

    sseg_list = []
    for s in _annots.lookup('segmentation'):
        m = kwimage.MultiPolygon.coerce(s)
        sseg_list.append(m)

    aids = _annots.aids
    cids = _annots.cids
    boxes = _annots.boxes
    segmentations = kwimage.PolygonList(sseg_list)
    class_idxs = np.array([cid_to_cidx[cid] for cid in cids])

    dets = kwimage.Detections(
        aids=aids,
        boxes=boxes,
        class_idxs=class_idxs,
        segmentations=segmentations,
        classes=classes,
        datakeys=['aids'],
    )

    if 1:
        print('dets = {!r}'.format(dets))
        print('dets.data = {!r}'.format(dets.data))
        print('dets.meta = {!r}'.format(dets.meta))

    if ub.argflag('--show'):
        import kwplot

        with ub.Timer('dets.draw_on'):
            canvas = imdata.copy()
            canvas = dets.draw_on(canvas)
            kwplot.imshow(canvas, pnum=(1, 2, 1), title='dets.draw_on')

        with ub.Timer('dets.draw'):
            kwplot.imshow(imdata, pnum=(1, 2, 2), docla=True, title='dets.draw')
            dets.draw()


def _devcheck_load_sub_image():
    import kwimage
    import numpy as np

    sampler = grab_camvid_sampler()

    cid_to_cidx = sampler.catgraph.id_to_idx
    classes = sampler.catgraph

    # Try loading a subregion of an image
    sample = sampler.load_positive(2)
    imdata = sample['im']
    annots = sample['annots']
    aids = annots['aids']
    cids = annots['cids']
    boxes = annots['rel_boxes']
    class_idxs = np.array([cid_to_cidx[cid] for cid in cids])
    segmentations = annots['rel_ssegs']

    raw_dets = kwimage.Detections(
        aids=aids,
        boxes=boxes,
        class_idxs=class_idxs,
        segmentations=segmentations,
        classes=classes,
        datakeys=['aids'],
    )

    # Clip boxes to the image boundary
    input_dims = imdata.shape[0:2]
    raw_dets.data['boxes'] = raw_dets.boxes.clip(0, 0, input_dims[1], input_dims[0])

    keep = []
    for i, s in enumerate(raw_dets.data['segmentations']):
        # TODO: clip polygons
        m = s.to_mask(input_dims)
        if m.area > 0:
            keep.append(i)
    dets = raw_dets.take(keep)

    heatmap = dets.rasterize(bg_size=(1, 1), input_dims=input_dims)

    if 1:
        print('dets = {!r}'.format(dets))
        print('dets.data = {!r}'.format(dets.data))
        print('dets.meta = {!r}'.format(dets.meta))

    if ub.argflag('--show'):
        import kwplot

        kwplot.autompl()
        heatmap.draw()

        draw_boxes = 1

        kwplot.figure(doclf=True)
        with ub.Timer('dets.draw_on'):
            canvas = imdata.copy()
            # TODO: add logic to color by class
            canvas = dets.draw_on(canvas, boxes=draw_boxes, color='random')
            kwplot.imshow(canvas, pnum=(1, 2, 1), title='dets.draw_on')

        with ub.Timer('dets.draw'):
            kwplot.imshow(imdata, pnum=(1, 2, 2), docla=True, title='dets.draw')
            dets.draw(boxes=draw_boxes, color='random')


def grab_camvid_train_test_val_splits(coco_dset, mode='segnet'):
    # Use the split from SegNet: https://github.com/alexgkendall/SegNet-Tutorial
    split_files = {
        'train': ub.grabdata('https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/train.txt'),
        'vali': ub.grabdata('https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/val.txt'),
        'test': ub.grabdata('https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/master/CamVid/test.txt'),
    }
    gid_subsets = {}
    for tag, fpath in split_files.items():
        text = open(fpath, 'r').read()
        parts = text.replace('\n', ' ').split(' ')
        parts = [p for p in parts if p]
        from os.path import basename
        names = sorted(set(basename(p) for p in parts))
        gids = [coco_dset.index.file_name_to_img['701_StillsRaw_full/' + name]['id'] for name in names]
        gid_subsets[tag] = gids
    return gid_subsets


def grab_camvid_sampler():
    """
    Grab a ndsampler.CocoSampler object for the CamVid dataset.

    Returns:
        ndsampler.CocoSampler: sampler

    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> sampler = grab_camvid_sampler()
        >>> print('sampler = {!r}'.format(sampler))
        >>> # sampler.load_sample()
        >>> for gid in ub.ProgIter(sampler.image_ids, desc='load image'):
        >>>     img = sampler.load_image(gid)
    """
    import ndsampler
    dset = grab_coco_camvid()
    workdir = ub.ensure_app_cache_dir('camvid')
    sampler = ndsampler.CocoSampler(dset, workdir=workdir)
    return sampler


def grab_coco_camvid():
    """
    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> dset = grab_coco_camvid()
        >>> print('dset = {!r}'.format(dset))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> plt.clf()
        >>> dset.show_image(gid=1)

    Ignore:
        import xdev
        gid_list = list(dset.imgs)
        for gid in xdev.InteractiveIter(gid_list):
            dset.show_image(gid)
            xdev.InteractiveIter.draw()
    """
    import ndsampler
    cache_dpath = ub.ensure_app_cache_dir('netharn', 'camvid')
    coco_fpath = join(cache_dpath, 'camvid.mscoco.json')

    # Need to manually bump this if you make a change to loading
    SCRIPT_VERSION = 'v4'

    # Ubelt's stamp-based caches are super cheap and let you take control of
    # the data format.
    stamp = ub.CacheStamp('camvid_coco', cfgstr=SCRIPT_VERSION,
                          dpath=cache_dpath, product=coco_fpath, hasher='sha1',
                          verbose=3)
    if stamp.expired():
        camvid_raw_info = grab_raw_camvid()
        dset = convert_camvid_raw_to_coco(camvid_raw_info)
        with ub.Timer('dumping MS-COCO dset to: {}'.format(coco_fpath)):
            dset.dump(coco_fpath)
        # Mark this process as completed by saving a small file containing the
        # hash of the "product" you are stamping.
        stamp.renew()

    # We can also cache the index build step independently. This uses
    # ubelt.Cacher, which is pickle based, and writes the actual object to
    # disk. Each type of caching has its own uses and tradeoffs.
    cacher = ub.Cacher('prebuilt-coco', cfgstr=SCRIPT_VERSION,
                       dpath=cache_dpath, verbose=3)
    dset = cacher.tryload()
    if dset is None:
        print('Reading coco_fpath = {!r}'.format(coco_fpath))
        dset = ndsampler.CocoDataset(coco_fpath, tag='camvid')
        # Directly save the file to disk.
        dset._build_index()
        dset._build_hashid()
        cacher.save(dset)

    camvid_dset = dset
    print('Loaded camvid_dset = {!r}'.format(camvid_dset))
    return camvid_dset


def grab_raw_camvid():
    """
    Grab the raw camvid data.
    """
    import zipfile
    dpath = ub.get_app_cache_dir('netharn', 'camvid')

    # url = 'http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/LabeledApproved_full.zip'
    # url = 'https://github.com/mostafaizz/camvid/archive/master.zip'
    url = 'https://data.kitware.com/api/v1/item/5cc0adce8d777f072b643503/download'
    zip_fpath = ub.grabdata(url, fname='camvid-master.zip', dpath=dpath)

    dset_root = join(dpath, 'camvid-master')
    image_dpath = join(dset_root, '701_StillsRaw_full')
    mask_dpath = join(dset_root, 'LabeledApproved_full')
    label_path = join(dset_root, 'label_colors.txt')

    if not exists(image_dpath):
        zip_ref = zipfile.ZipFile(zip_fpath, 'r')
        zip_ref.extractall(dpath)
        zip_ref.close()

    import glob
    img_paths = sorted([relpath(fpath, dset_root)
                        for fpath in glob.glob(join(image_dpath, '*.png'))])
    mask_paths = sorted([relpath(fpath, dset_root)
                         for fpath in glob.glob(join(mask_dpath, '*.png'))])

    camvid_raw_info = {
        'img_paths': img_paths,
        'mask_paths': mask_paths,
        'dset_root': dset_root,
        'label_path': label_path,
    }
    return camvid_raw_info


def rgb_to_cid(r, g, b):
    cid = (r << 16) + (g << 8) + (b << 0)
    return cid


def cid_to_rgb(cid):
    mask_b = (int(2 ** 8) - 1) << 0
    mask_g = (int(2 ** 8) - 1) << 8
    mask_r = (int(2 ** 8) - 1) << 16

    r = (cid & mask_b) >> 0
    g = (cid & mask_g) >> 8
    b = (cid & mask_r) >> 16
    rgb = (r, g, b)
    return rgb


def convert_camvid_raw_to_coco(camvid_raw_info):
    """
    Converts the raw camvid format to an MSCOCO based format, ( which lets use
    use ndsampler's COCO backend).

    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> camvid_raw_info = grab_raw_camvid()
        >>> # test with a reduced set of data
        >>> del camvid_raw_info['img_paths'][2:]
        >>> del camvid_raw_info['mask_paths'][2:]
        >>> dset = convert_camvid_raw_to_coco(camvid_raw_info)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, pnum=(1, 2, 1))
        >>> dset.show_image(gid=1)
        >>> kwplot.figure(fnum=1, pnum=(1, 2, 2))
        >>> dset.show_image(gid=2)
    """
    import re
    import kwimage
    import ndsampler
    print('Converting CamVid to MS-COCO format')

    dset_root, img_paths, label_path, mask_paths = ub.take(
        camvid_raw_info, 'dset_root, img_paths, label_path, mask_paths'.split(', '))

    img_infos = {
        'img_fname': img_paths,
        'mask_fname': mask_paths,
    }
    keys = list(img_infos.keys())
    next_vals = list(zip(*img_infos.values()))
    image_items = [{k: v for k, v in zip(keys, vals)} for vals in next_vals]

    dataset = {
        'img_root': dset_root,
        'images': [],
        'categories': [],
        'annotations': [],
    }

    lines = ub.readfrom(label_path).split('\n')
    lines = [line for line in lines if line]
    for line in lines:
        color_text, name = re.split('\t+', line)
        r, g, b = map(int, color_text.split(' '))
        color = (r, g, b)

        # Parse the special camvid format
        cid = (r << 16) + (g << 8) + (b << 0)
        cat = {
            'id': cid,
            'name': name,
            'color': color,
        }
        dataset['categories'].append(cat)

    for gid, img_item in enumerate(image_items, start=1):
        img = {
            'id': gid,
            'file_name': img_item['img_fname'],
            # nonstandard image field
            'segmentation': img_item['mask_fname'],
        }
        dataset['images'].append(img)

    dset = ndsampler.CocoDataset(dataset)
    dset.rename_categories({'Void': 'background'})

    assert dset.name_to_cat['background']['id'] == 0
    dset.name_to_cat['background'].setdefault('alias', []).append('Void')

    if False:
        _define_camvid_class_hierarcy(dset)

    if 1:
        # TODO: Binarize CCs (and efficiently encode if possible)
        import numpy as np

        bad_info = []
        once = False

        # Add images
        dset.remove_all_annotations()
        for gid, img in ub.ProgIter(dset.imgs.items(), desc='parse label masks'):
            mask_fpath = join(dset_root, img['segmentation'])

            rgb_mask = kwimage.imread(mask_fpath, space='rgb')
            r, g, b  = rgb_mask.T.astype(np.int64)
            cid_mask = np.ascontiguousarray(rgb_to_cid(r, g, b).T)

            cids = set(np.unique(cid_mask)) - {0}

            for cid in cids:
                if cid not in dset.cats:
                    if gid == 618:
                        # Handle a known issue with image 618
                        c_mask = (cid == cid_mask).astype(np.uint8)
                        total_bad = c_mask.sum()
                        if total_bad < 32:
                            if not once:
                                print('gid 618 has a few known bad pixels, ignoring them')
                                once = True
                            continue
                        else:
                            raise Exception('more bad pixels than expected')
                    else:
                        raise Exception('UNKNOWN cid = {!r} in gid={!r}'.format(cid, gid))

                    # bad_rgb = cid_to_rgb(cid)
                    # print('bad_rgb = {!r}'.format(bad_rgb))
                    # print('WARNING UNKNOWN cid = {!r} in gid={!r}'.format(cid, gid))
                    # bad_info.append({
                    #     'gid': gid,
                    #     'cid': cid,
                    # })
                else:
                    ann = {
                        'category_id': cid,
                        'image_id': gid
                        # 'segmentation': mask.to_coco()
                    }
                    assert cid in dset.cats
                    c_mask = (cid == cid_mask).astype(np.uint8)
                    mask = kwimage.Mask(c_mask, 'c_mask')

                    box = kwimage.Boxes([mask.get_xywh()], 'xywh')
                    # box = mask.to_boxes()

                    ann['bbox'] = ub.peek(box.to_coco())
                    ann['segmentation'] = mask.to_coco()
                    dset.add_annotation(**ann)

        if 0:
            bad_cids = [i['cid'] for i in bad_info]
            print(sorted([c['color'] for c in dataset['categories']]))
            print(sorted(set([cid_to_rgb(i['cid']) for i in bad_info])))

            gid = 618
            img = dset.imgs[gid]
            mask_fpath = join(dset_root, img['segmentation'])
            rgb_mask = kwimage.imread(mask_fpath, space='rgb')
            r, g, b  = rgb_mask.T.astype(np.int64)
            cid_mask = np.ascontiguousarray(rgb_to_cid(r, g, b).T)
            cid_hist = ub.dict_hist(cid_mask.ravel())

            bad_cid_hist = {}
            for cid in bad_cids:
                bad_cid_hist[cid] = cid_hist.pop(cid)

            import kwplot
            kwplot.autompl()
            kwplot.imshow(rgb_mask)

    if 0:
        import kwplot
        plt = kwplot.autoplt()
        plt.clf()
        dset.show_image(1)

        import xdev
        gid_list = list(dset.imgs)
        for gid in xdev.InteractiveIter(gid_list):
            dset.show_image(gid)
            xdev.InteractiveIter.draw()

    dset._build_index()
    dset._build_hashid()
    return dset


def _define_camvid_class_hierarcy(dset):
    # add extra supercategories
    # NOTE: life-conscious, and life-inanimate are disjoint in this
    # forumlation because we are restricted to a tree structure.  If
    # this changse, then we can try rencoding with multiple parents.
    extra_structure = {
        # Break down the image into things that are part of the system, and
        # things that aren't
        'background': 'root',
        'system': 'root',

        # The system is made up of environmental components and actor
        # components.
        'environment': 'system',
        'actor': 'system',

        # Break actors (things with complex movement) into subtypes
        'life-conscious': 'actor',
        'vehicle-land': 'actor',
        'actor-other': 'actor',

        # Break the environment (things with simple movement) info subtypes
        'life-inanimate': 'environment',
        'civil-structure': 'environment',
        'civil-notice': 'environment',
        'transport-way': 'environment',

        # Subclassify transport mediums
        'drive-way': 'transport-way',
        'walk-way': 'transport-way',
    }

    for child, parent in extra_structure.items():
        if child in dset.name_to_cat:
            dset.name_to_cat[child]['supercategory'] = parent
        else:
            dset.add_category(name=child, supercategory=parent)

    dset.name_to_cat['background']['supercategory'] = 'root'

    dset.name_to_cat['Sky']['supercategory'] = 'environment'

    dset.name_to_cat['Animal']['supercategory'] = 'life-conscious'
    dset.name_to_cat['Bicyclist']['supercategory'] = 'life-conscious'
    dset.name_to_cat['Pedestrian']['supercategory'] = 'life-conscious'
    dset.name_to_cat['Child']['supercategory'] = 'life-conscious'

    dset.name_to_cat['OtherMoving']['supercategory'] = 'actor-other'
    dset.name_to_cat['CartLuggagePram']['supercategory'] = 'actor-other'

    dset.name_to_cat['Car']['supercategory'] = 'vehicle-land'
    dset.name_to_cat['Train']['supercategory'] = 'vehicle-land'
    dset.name_to_cat['Truck_Bus']['supercategory'] = 'vehicle-land'
    dset.name_to_cat['SUVPickupTruck']['supercategory'] = 'vehicle-land'
    dset.name_to_cat['MotorcycleScooter']['supercategory'] = 'vehicle-land'

    dset.name_to_cat['VegetationMisc']['supercategory'] = 'life-inanimate'
    dset.name_to_cat['Tree']['supercategory'] = 'life-inanimate'

    dset.name_to_cat['Column_Pole']['supercategory'] = 'civil-structure'
    dset.name_to_cat['Fence']['supercategory'] = 'civil-structure'
    dset.name_to_cat['Wall']['supercategory'] = 'civil-structure'
    dset.name_to_cat['Building']['supercategory'] = 'civil-structure'
    dset.name_to_cat['Archway']['supercategory'] = 'civil-structure'
    dset.name_to_cat['Bridge']['supercategory'] = 'civil-structure'
    dset.name_to_cat['Tunnel']['supercategory'] = 'civil-structure'

    dset.name_to_cat['TrafficCone']['supercategory'] = 'civil-notice'
    dset.name_to_cat['TrafficLight']['supercategory'] = 'civil-notice'
    dset.name_to_cat['LaneMkgsDriv']['supercategory'] = 'civil-notice'
    dset.name_to_cat['LaneMkgsNonDriv']['supercategory'] = 'civil-notice'
    dset.name_to_cat['SignSymbol']['supercategory'] = 'civil-notice'
    dset.name_to_cat['ParkingBlock']['supercategory'] = 'civil-notice'
    dset.name_to_cat['Misc_Text']['supercategory'] = 'civil-notice'

    dset.name_to_cat['Road']['supercategory'] = 'drive-way'
    dset.name_to_cat['RoadShoulder']['supercategory'] = 'drive-way'
    dset.name_to_cat['Sidewalk']['supercategory'] = 'walk-way'

    for cat in list(dset.cats.values()):
        parent = cat.get('supercategory', None)
        if parent is not None:
            if parent not in dset.name_to_cat:
                print('Missing parent = {!r}'.format(parent))
                dset.add_category(name=parent, supercategory=parent)

    if 0:
        graph = dset.category_graph()
        import graphid
        graphid.util.show_nx(graph)

    # Add in some hierarcy information
    if 0:
        for x in dset.name_to_cat:
            print("dset.name_to_cat[{!r}]['supercategory'] = 'object'".format(x))

    if 0:
        example_cat_aids = []
        for cat in dset.cats.values():
            cname = cat['name']
            aids = dset.index.cid_to_aids[dset.name_to_cat[cname]['id']]
            if len(aids):
                aid = ub.peek(aids)
                example_cat_aids.append(aid)
            else:
                print('No examples of cat = {!r}'.format(cat))

        import xdev
        import kwplot
        kwplot.autompl()
        for aid in xdev.InteractiveIter(example_cat_aids):
            print('aid = {!r}'.format(aid))
            ann = dset.anns[aid]
            cat = dset.cats[ann['category_id']]
            print('cat = {!r}'.format(cat))
            dset.show_image(aid=aid)
            xdev.InteractiveIter.draw()

        if 0:
            cname = 'CartLuggagePram'
            cname = 'ParkingBlock'
            cname = 'LaneMkgsDriv'
            aids = dset.index.cid_to_aids[dset.name_to_cat[cname]['id']]
            if len(aids):
                aid = ub.peek(aids)
                print('aid = {!r}'.format(aid))
                ann = dset.anns[aid]
                cat = dset.cats[ann['category_id']]
                print('cat = {!r}'.format(cat))
                dset.show_image(aid=aid)


def main():
    """
    Dump the paths to the coco file to stdout

    By default these will go to in the path:
        ~/.cache/netharn/camvid/camvid-master

    The four files will be:
        ~/.cache/netharn/camvid/camvid-master/camvid-full.mscoco.json
        ~/.cache/netharn/camvid/camvid-master/camvid-train.mscoco.json
        ~/.cache/netharn/camvid/camvid-master/camvid-vali.mscoco.json
        ~/.cache/netharn/camvid/camvid-master/camvid-test.mscoco.json
    """
    coco_dset = grab_coco_camvid()

    # Use the same train/test/vali splits used in segnet
    gid_subsets = grab_camvid_train_test_val_splits(coco_dset, mode='segnet')
    dpath = coco_dset.dataset['img_root']

    # Dump the full dataset
    fpath = join(dpath, 'camvid-full.mscoco.json')
    print(fpath)
    coco_dset.dump(open(fpath, 'w'))

    # Dump the train/vali/test splits
    for tag, gids in gid_subsets.items():
        subset = coco_dset.subset(gids)
        fpath = join(dpath, 'camvid-{}.mscoco.json'.format(tag))
        print(fpath)
        subset.dump(open(fpath, 'w'))

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.data.grab_camvid
    """
    main()
