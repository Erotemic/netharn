from os.path import relpath
from os.path import exists
from os.path import join
import ubelt as ub


def grab_camvid():
    import zipfile

    dpath = ub.get_app_cache_dir('netharn', 'camvid')
    dset_root = join(dpath, 'camvid-master')
    image_dpath = join(dset_root, '701_StillsRaw_full')
    mask_dpath = join(dset_root, 'LabeledApproved_full')
    label_path = join(dset_root, 'label_colors.txt')
    # url = 'http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/LabeledApproved_full.zip'

    url = 'https://github.com/mostafaizz/camvid/archive/master.zip'
    zip_fpath = ub.grabdata(url, dpath=dpath)

    if not exists(image_dpath):
        zip_ref = zipfile.ZipFile(zip_fpath, 'r')
        zip_ref.extractall(dpath)
        zip_ref.close()

    import glob
    img_paths = sorted([relpath(fpath, dset_root) for fpath in glob.glob(join(image_dpath, '*.png'))])
    mask_paths = sorted([relpath(fpath, dset_root) for fpath in glob.glob(join(mask_dpath, '*.png'))])

    dataset = {
        'img_root': dset_root,
        'images': [],
        'categories': [],
        'annotations': [],
    }

    import re
    for line in (_ for _ in ub.readfrom(label_path).split('\n') if _):
        color, name = re.split('\t+', line)
        r, g, b = map(int, color.split(' '))
        cid = (r << 16) + (g << 8) + (b << 0)
        cat = {
            'id': cid,
            'name': name,
        }
        dataset['categories'].append(cat)

    for gid, (g, m) in enumerate(zip(img_paths, mask_paths), start=1):
        img = {
            'id': gid,
            'file_name': g,
            'segmentation': m,
        }
        dataset['images'].append(img)

    import kwil
    kwil.autompl()

    import ndsampler
    dset = ndsampler.CocoDataset(dataset)
    dset._build_index()
    dset._build_hashid()

    # dset.show_image(gid=1)
    # ndsampler.abstract_frames.SimpleFrames()

    workdir = ub.ensure_app_cache_dir('camvid')
    sampler = ndsampler.CocoSampler(dset, workdir=workdir)

    # sampler.load_sample()

    for gid in ub.ProgIter(sampler.image_ids, desc='load image'):
        sampler.load_image(gid)
