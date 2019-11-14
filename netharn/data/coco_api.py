# -*- coding: utf-8 -*-
"""
DEPRECATED

NOTE:
    THIS IS DEPRECATED IN FAVOR OF COCO_DATASET IN NDSAMPLER


Extended MS-COCO API. Currently only supports keypoints and bounding boxes.

Extends the format to also include line annotations.

Dataset Spec:
    dataset = {
        'categories': [
            {
                'id': int, 'name': str,
                'supercategory': str  # optional
            },
            ...
        ],
        'images': [
            {'id': int, 'file_name': str},
            ...
        ],
        'annotations': [
            {
                'id': int,
                'image_id': int,
                'category_id': int,
                'bbox': [tl_x, tl_y, w, h],  # optional (xywh format)
            },
            ...
        ],
        'licenses': [],
        'info': [],
    }

Extras:
    We allow each annotation to specify a fine_category_id which specifies the
    most specific category that the annotation can be labeled as. Assumes
    categories are tree-structures.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
from os.path import splitext
from os.path import basename
from os.path import join
from collections import OrderedDict
from six.moves import cStringIO as StringIO
import json
import numpy as np
import ubelt as ub
import six
import itertools as it
import copy

__all__ = [
    'CocoDataset',
]

_dict = OrderedDict


INT_TYPES = (int, np.integer)


def annot_type(ann):
    """
    Returns what type of annotation `ann` is.
    """
    return tuple(sorted(set(ann) & {'bbox', 'line', 'keypoints'}))


class ObjectList1D(ub.NiceRepr):
    """
    Lightweight reference to a set of annotations that allows for convenient
    property access.

    Similar to ibeis._ibeis_object.ObjectList1D
    """
    def __init__(self, ids, dset, key):
        self._key = key
        self._ids = ids
        self._dset = dset

    def __nice__(self):
        return 'num={!r}'.format(len(self))

    def __iter__(self):
        return iter(self._ids)

    def __len__(self):
        return len(self._ids)

    def take(self, idxs):
        """
        Example:
            >>> self = CocoDataset.demo().annots()
            >>> assert len(self.take([0, 2, 3])) == 3
        """
        subids = list(ub.take(self._ids, idxs))
        newself = self.__class__(subids, self._dset)
        return newself

    def compress(self, flags):
        """
        Example:
            >>> self = CocoDataset.demo().images()
            >>> assert len(self.compress([True, False, True])) == 2
        """
        subids = list(ub.compress(self._ids, flags))
        newself = self.__class__(subids, self._dset)
        return newself

    def _lookup(self, key):
        raise NotImplementedError('must be implemented')

    def _ilookup(self, key):
        raise NotImplementedError('must be implemented')


class ObjectGroups(ub.NiceRepr):
    def __init__(self, groups, dset):
        self._groups = groups

    def _lookup(self, key):
        return [group._lookup(key) for group in self._groups]

    def _ilookup(self, key):
        return (group._lookup(key) for group in self._groups)

    def __nice__(self):
        # import timerit
        # mu = timerit.core._trychar('μ', 'm')
        # sigma = timerit.core._trychar('σ', 's')
        mu = 'm'
        sigma = 's'
        len_list = list(map(len, self._groups))
        num = len(self._groups)
        mean = np.mean(len_list)
        std = np.std(len_list)
        nice = 'n={!r}, {}={:.1f}, {}={:.1f}'.format(
            num, mu, mean, sigma, std)
        return nice


class Images(ObjectList1D):
    """
    """
    def __init__(self, ids, dset):
        super(Images, self).__init__(ids, dset, 'images')

    @property
    def gids(self):
        return self._ids

    def _lookup(self, key):
        return [img[key] for img in ub.take(self._dset.imgs, self._ids)]

    def _ilookup(self, key):
        return (img[key] for img in ub.take(self._dset.imgs, self._ids))

    @property
    def width(self):
        return self._lookup('width')

    @property
    def height(self):
        return self._lookup('height')

    @property
    def size(self):
        """
        Example:
            >>> from netharn.data.coco_api import *
            >>> self = CocoDataset.demo().images()
            >>> self._dset._ensure_imgsize()
            >>> print(self.size)
            [(512, 512), (300, 250), (256, 256)]
        """
        return list(zip(self._ilookup('width'), self._ilookup('height')))

    @property
    def area(self):
        """
        Example:
            >>> from netharn.data.coco_api import *
            >>> self = CocoDataset.demo().images()
            >>> self._dset._ensure_imgsize()
            >>> print(self.area)
            [262144, 75000, 65536]
        """
        return [w * h for w, h in zip(self._ilookup('width'), self._ilookup('height'))]

    @property
    def n_annots(self):
        """
        Example:
            >>> self = CocoDataset.demo().images()
            >>> print(ub.repr2(self.n_annots, nl=0))
            [9, 2, 0]
        """
        return list(map(len, ub.take(self._dset.gid_to_aids, self._ids)))

    @property
    def aids(self):
        """
        Example:
            >>> self = CocoDataset.demo().images()
            >>> print(ub.repr2(list(map(list, self.aids)), nl=0))
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11], []]
        """
        return list(ub.take(self._dset.gid_to_aids, self._ids))

    @property
    def annots(self):
        """
        Example:
            >>> self = CocoDataset.demo().images()
            >>> print(self.annots)
            <AnnotGroups(n=3, m=3.7, s=3.9)>
        """
        return AnnotGroups([self._dset.annots(aids) for aids in self.aids],
                           self._dset)


class Annots(ObjectList1D):
    """
    """
    def __init__(self, ids, dset):
        super(Annots, self).__init__(ids, dset, 'annotations')

    @property
    def aids(self):
        return self._ids

    @property
    def images(self):
        return self._dset.images(self.gids)

    @property
    def gids(self):
        return self._lookup('image_id')

    @property
    def cids(self):
        return self._lookup('category_id')

    def _lookup(self, key):
        return [ann[key] for ann in ub.take(self._dset.anns, self._ids)]

    def _ilookup(self, key):
        return (ann[key] for ann in ub.take(self._dset.anns, self._ids))

    @property
    def boxes(self):
        """
        Returns netharn style boxes

        Example:
            >>> self = CocoDataset.demo().annots([1, 2, 11])
            >>> print(self.boxes)
            <Boxes(xywh,
                array([[ 10,  10, 360, 490],
                       [350,   5, 130, 290],
                       [124,  96,  45,  18]]))>
        """
        import netharn as nh
        xywh = self._lookup('bbox')
        boxes = nh.util.Boxes(xywh, 'xywh')
        return boxes

    @property
    def xywh(self):
        """
        Returns raw boxes

        Example:
            >>> self = CocoDataset.demo().annots([1, 2, 11])
            >>> print(self.boxes)
            <Boxes(xywh,
                array([[ 10,  10, 360, 490],
                       [350,   5, 130, 290],
                       [124,  96,  45,  18]]))>
        """
        xywh = self._lookup('bbox')
        return xywh


class AnnotGroups(ObjectGroups):
    @property
    def cids(self):
        return self._lookup('category_id')


class ImageGroups(ObjectGroups):
    pass


class MixinCocoDepricate(object):
    """
    These functions are marked for deprication and may be removed at any time
    """

    def lookup_imgs(self, filename=None):
        """
        Linear search for an images with specific attributes

        # DEPRICATE

        Ignore:
            filename = '201503.20150525.101841191.573975.png'
            list(self.lookup_imgs(filename))
            gid = 64940
            img = self.imgs[gid]
            img['file_name'] = filename
        """
        for img in self.imgs.values():
            if filename is not None:
                fpath = img['file_name']
                fname = basename(fpath)
                fname_noext = splitext(fname)[0]
                if filename in [fpath, fname, fname_noext]:
                    print('img = {!r}'.format(img))
                    yield img

    def lookup_anns(self, has=None):
        """
        Linear search for an annotations with specific attributes

        # DEPRICATE

        Ignore:
            list(self.lookup_anns(has='radius'))
            gid = 112888
            img = self.imgs[gid]
            img['file_name'] = filename
        """
        for ann in self.anns.values():
            if has is not None:
                if hasattr(ann, has):
                    print('ann = {!r}'.format(ann))
                    yield ann

    def _mark_annotated_images(self):
        """
        Mark any image that explicitly has annotations.

        # DEPRICATE
        """
        for gid, img in self.imgs.items():
            aids = self.gid_to_aids.get(gid, [])
            # If there is at least one annotation, always mark as has_annots
            if len(aids) > 0:
                assert img.get('has_annots', ub.NoParam) in [ub.NoParam, True], (
                    'image with annots was explictly labeled as non-True!')
                img['has_annots'] = True
            else:
                # Otherwise set has_annots to null if it has not been
                # explicitly labeled
                if 'has_annots' not in img:
                    img['has_annots'] = None

    def _find_bad_annotations(self):
        to_remove = []
        for ann in self.dataset['annotations']:
            if ann['image_id'] is None or ann['category_id'] is None:
                to_remove.append(ann)
            else:
                if ann['image_id'] not in self.imgs:
                    to_remove.append(ann)
                if ann['category_id'] not in self.cats:
                    to_remove.append(ann)
        return to_remove

    def _remove_keypoint_annotations(self, rebuild=True):
        """
        Remove annotations with keypoints only

        Example:
            >>> self = CocoDataset.demo()
            >>> self._remove_keypoint_annotations()
        """
        to_remove = []
        for ann in self.dataset['annotations']:
            roi_shape = ann.get('roi_shape', None)
            if roi_shape is None:
                if 'keypoints' in ann and ann.get('bbox', None) is None:
                    to_remove.append(ann)
            elif roi_shape == 'keypoints':
                to_remove.append(ann)
        print('Removing {} keypoint annotations'.format(len(to_remove)))
        self.remove_annotations(to_remove)
        if rebuild:
            self._build_index()

    def _remove_bad_annotations(self, rebuild=True):
        # DEPRICATE
        to_remove = []
        for ann in self.dataset['annotations']:
            if ann['image_id'] is None or ann['category_id'] is None:
                to_remove.append(ann)
        print('Removing {} bad annotations'.format(len(to_remove)))
        self.remove_annotations(to_remove)
        if rebuild:
            self._build_index()

    def _remove_radius_annotations(self, rebuild=False):
        # DEPRICATE
        to_remove = []
        for ann in self.dataset['annotations']:
            if 'radius' in ann:
                to_remove.append(ann)
        print('Removing {} radius annotations'.format(len(to_remove)))
        self.remove_annotations(to_remove)
        if rebuild:
            self._build_index()

    def _remove_empty_images(self):
        to_remove = []
        for gid in self.imgs.keys():
            aids = self.gid_to_aids.get(gid, [])
            if not aids:
                to_remove.append(self.imgs[gid])
        print('Removing {} empty images'.format(len(to_remove)))
        for img in to_remove:
            self.dataset['images'].remove(img)
        self._build_index()


class MixinCocoExtras(object):
    """
    Misc functions for coco
    """

    @classmethod
    def demo(cls):
        dataset = demo_coco_data()
        self = cls(dataset, tag='demo')
        return self

    def _build_hashid(self, hash_pixels=True, verbose=0):
        """
        Construct a hash that uniquely identifies the state of this dataset.

        Args:
            hash_pixels (bool): If False the image data is not included in the
                hash, which can speed up computation. Defaults to True.
            verbose (int): verbosity level

        Example:
            >>> self = CocoDataset.demo()
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            ...
            >>> print('self.hashid_parts = ' + ub.repr2(self.hashid_parts))
            >>> print('self.hashid = {!r}'.format(self.hashid))
            self.hashid_parts = {
                'annotations': {
                    'json': 'c6e2a55613b...',
                    'num': 11,
                },
                'images': {
                    'pixels': '67d741fefc8...',
                    'json': 'd52eeb0c4...',
                    'num': 3,
                },
                'categories': {
                    'json': '9a92615e2b...',
                    'num': 7,
                },
            }
            self.hashid = 'b45256273...

        Doctest:
            >>> self = CocoDataset.demo()
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            >>> self.hashid_parts
            >>> # Test that when we modify the dataset only the relevant
            >>> # hashid parts are recomputed.
            >>> orig = self.hashid_parts['categories']['json']
            >>> self.add_category('foobar')
            >>> assert 'categories' not in self.hashid_parts
            >>> self.hashid_parts
            >>> self.hashid_parts['images']['json'] = 'should not change'
            >>> self._build_hashid(hash_pixels=True, verbose=3)
            >>> assert self.hashid_parts['categories']['json']
            >>> assert self.hashid_parts['categories']['json'] != orig
            >>> assert self.hashid_parts['images']['json'] == 'should not change'
        """
        # Construct nested container that we will populate with hashable
        # info corresponding to each type of data that we track.
        hashid_parts = self.hashid_parts
        if hashid_parts is None:
            hashid_parts = OrderedDict()

        # Ensure hashid_parts has the proper root structure
        parts = ['annotations', 'images', 'categories']
        for part in parts:
            if not hashid_parts.get(part, None):
                hashid_parts[part] = OrderedDict()

        rebuild_parts = []
        reuse_parts = []

        gids = None
        if hash_pixels:
            if not hashid_parts['images'].get('pixels', None):
                gids = sorted(self.imgs.keys())
                gpaths = [join(self.img_root, gname)
                          for gname in self.images(gids)._lookup('file_name')]
                gpath_sha512s = [
                    ub.hash_file(gpath, hasher='sha512')
                    for gpath in ub.ProgIter(gpaths, desc='hashing images',
                                             verbose=verbose)
                ]
                hashid_parts['images']['pixels'] = ub.hash_data(gpath_sha512s)
                rebuild_parts.append('images.pixels')
            else:
                reuse_parts.append('images.pixels')

        # Hash individual components
        with ub.Timer(label='hash coco parts', verbose=verbose > 1):
            # Dumping annots to json takes the longest amount of time
            # However, its faster than hashing the data directly
            def _ditems(d):
                # return sorted(d.items())
                return list(d.items()) if isinstance(d, OrderedDict) else sorted(d.items())

            if not hashid_parts['annotations'].get('json', None):
                aids = sorted(self.anns.keys())
                _anns_ordered = (self.anns[aid] for aid in aids)
                anns_ordered = [_ditems(ann) for ann in _anns_ordered]
                anns_text = json.dumps(anns_ordered)
                hashid_parts['annotations']['json'] = ub.hash_data(anns_text, hasher='sha512')
                hashid_parts['annotations']['num'] = len(aids)
                rebuild_parts.append('annotations.json')
            else:
                reuse_parts.append('annotations.json')

            if not hashid_parts['images'].get('json', None):
                if gids is None:
                    gids = sorted(self.imgs.keys())
                imgs_text = json.dumps([_ditems(self.imgs[gid]) for gid in gids])
                hashid_parts['images']['json'] = ub.hash_data(imgs_text, hasher='sha512')
                hashid_parts['images']['num'] = len(gids)
                rebuild_parts.append('images.json')
            else:
                reuse_parts.append('images.json')

            if not hashid_parts['categories'].get('json', None):
                cids = sorted(self.cats.keys())
                cats_text = json.dumps([_ditems(self.cats[cid]) for cid in cids])
                hashid_parts['categories']['json'] = ub.hash_data(cats_text, hasher='sha512')
                hashid_parts['categories']['num'] = len(cids)
                rebuild_parts.append('categories.json')
            else:
                reuse_parts.append('categories.json')

        if verbose > 1:
            if reuse_parts:
                print('Reused hashid_parts: {}'.format(reuse_parts))
                print('Rebuilt hashid_parts: {}'.format(rebuild_parts))

        hashid = ub.hash_data(hashid_parts)
        self.hashid = hashid
        self.hashid_parts = hashid_parts
        return hashid

    def _invalidate_hashid(self, parts=None):
        """
        Called whenever the coco dataset is modified. It is possible to specify
        which parts were modified so unmodified parts can be reused next time
        the hash is constructed.
        """
        self.hashid = None
        if parts is not None and self.hashid_parts is not None:
            for part in parts:
                self.hashid_parts.pop(part, None)
        else:
            self.hashid_parts = None

    def _ensure_imgsize(self, verbose=1):
        """
        Populate the imgsize field if it does not exist.

        Example:
            >>> self = CocoDataset.demo()
            >>> self._ensure_imgsize()
            >>> assert self.imgs[1]['width'] == 512
            >>> assert self.imgs[2]['width'] == 300
            >>> assert self.imgs[3]['width'] == 256
        """
        if any('width' not in img for img in self.dataset['images']):
            from PIL import Image
            if self.tag:
                desc = 'populate imgsize for ' + self.tag
            else:
                desc = 'populate imgsize for untagged coco dataset'
            for img in ub.ProgIter(self.dataset['images'], desc=desc,
                                   verbose=verbose):
                gpath = join(self.img_root, img['file_name'])
                if 'width' not in img:
                    pil_img = Image.open(gpath)
                    w, h = pil_img.size
                    pil_img.close()
                    img['width'] = w
                    img['height'] = h

    def _resolve_to_id(self, id_or_dict):
        """
        Ensures output is an id
        """
        if isinstance(id_or_dict, INT_TYPES):
            resolved_id = id_or_dict
        else:
            resolved_id = id_or_dict['id']
        return resolved_id

    def _resolve_to_cid(self, id_or_name_or_dict):
        """
        Ensures output is an category id
        """
        if isinstance(id_or_name_or_dict, INT_TYPES):
            resolved_id = id_or_name_or_dict
        elif isinstance(id_or_name_or_dict, six.string_types):
            resolved_id = self.index.name_to_cat[id_or_name_or_dict]['id']
        else:
            resolved_id = id_or_name_or_dict['id']
        return resolved_id

    def _resolve_to_ann(self, aid_or_ann):
        """
        Ensures output is an annotation dictionary
        """
        if isinstance(aid_or_ann, INT_TYPES):
            resolved_ann = None
            if self.anns is not None:
                resolved_ann = self.anns[aid_or_ann]
            else:
                for ann in self.dataset['annotations']:
                    if ann['id'] == aid_or_ann:
                        resolved_ann = ann
                        break
                if not resolved_ann:
                    raise IndexError('aid {} not in dataset'.format(aid_or_ann))
        else:
            resolved_ann = aid_or_ann
        return resolved_ann

    def category_graph(self):
        """
            >>> self = CocoDataset.demo()
            >>> graph = self.category_graph()

            import graphid
            graphid.util.show_nx(graph)
        """
        import networkx as nx
        graph = nx.DiGraph()
        for cat in self.dataset['categories']:
            graph.add_node(cat['name'], **cat)
            if 'supercategory' in cat:
                graph.add_edge(cat['supercategory'], cat['name'])
        return graph

    def missing_images(self):
        import os
        bad_paths = []
        for index in ub.ProgIter(range(len(self.dataset['images']))):
            img = self.dataset['images'][index]
            gpath = join(self.img_root, img['file_name'])
            if not os.path.exists(gpath):
                bad_paths.append((index, gpath))
        return bad_paths
        # if bad_paths:
        #     print('bad paths:')
        #     print(ub.repr2(bad_paths, nl=1))
        # raise AssertionError('missing images')

    def rename_categories(self, mapper, strict=False, preserve=False,
                          rebuild=True, simple=False):
        """
        Create a coarser categorization

        Args:
            mapper (dict or Function): maps old names to new names.
            strict (bool): if True, fails if mapper doesnt map all classes
            preserve (bool): if True, preserve old categories as supercatgories
                FIXME: Broken

        Example:
            >>> # DISABLE_DOCTEST
            >>> self = CocoDataset.demo()
            >>> self.rename_categories({'astronomer': 'person', 'astronaut': 'person', 'mouth': 'person', 'helmet': 'hat'}, preserve=0)
            >>> self.rename_categories({'person': 'obj', 'hat': 'obj'}, preserve=0)
            >>> assert 'hat' in self.name_to_cat
            >>> assert 'helmet' not in self.name_to_cat
        """
        old_cats = self.dataset['categories']

        if simple:
            # In the simple case we are just changing the labels, so nothing
            # special needs to happen.
            old_cat_names = {cat['name'] for cat in old_cats}
            assert set(mapper.keys()).issubset(set(old_cat_names))
            assert not set(mapper.values()).intersection(set(old_cat_names))
            assert not set(mapper.values()).intersection(set(mapper.keys()))

            for key, value in mapper.items():
                for cat in self.dataset['categories']:
                    if cat['name'] == key:
                        cat['name'] = value
        else:
            new_cats = []
            old_cats = self.dataset['categories']
            new_name_to_cat = {}
            old_to_new_id = {}

            if not callable(mapper):
                mapper = mapper.__getitem__

            for old_cat in old_cats:
                try:
                    new_name = mapper(old_cat['name'])
                except KeyError:
                    if strict:
                        raise
                    new_name = old_cat['name']

                old_cat['supercategory'] = new_name

                if new_name in new_name_to_cat:
                    # Multiple old categories are mapped to this new one
                    new_cat = new_name_to_cat[new_name]
                else:
                    if old_cat['name'] == new_name:
                        # new name is an existing category
                        new_cat = old_cat.copy()
                        new_cat['id'] = len(new_cats) + 1
                    else:
                        # new name is a entirely new category
                        new_cat = _dict([
                            ('id', len(new_cats) + 1),
                            ('name', new_name),
                        ])
                    new_name_to_cat[new_name] = new_cat
                    new_cats.append(new_cat)

                old_to_new_id[old_cat['id']] = new_cat['id']

            if preserve:
                raise NotImplementedError
                # for old_cat in old_cats:
                #     # Ensure all old cats are preserved
                #     if old_cat['name'] not in new_name_to_cat:
                #         new_cat = old_cat.copy()
                #         new_cat['id'] = len(new_cats) + 1
                #         new_name_to_cat[new_name] = new_cat
                #         new_cats.append(new_cat)
                #         old_to_new_id[old_cat['id']] = new_cat['id']

            # self.dataset['fine_categories'] = old_cats
            self.dataset['categories'] = new_cats

            # Fix annotations of modified categories
            # (todo: if the index is built, we can use that to only modify
            #  a potentially smaller subset of annotations)
            for ann in self.dataset['annotations']:
                old_id = ann['category_id']
                new_id = old_to_new_id[old_id]

                if old_id != new_id:
                    ann['category_id'] = new_id
                    # See if the annotation already has a fine-grained category If
                    # not, then use the old id as its current fine-grained
                    # granularity
                    fine_id = ann.get('fine_category_id', None)
                    if fine_id is None:
                        ann['fine_category_id'] = old_id
        if rebuild:
            self._build_index()
        else:
            self.index.clear()
        self._invalidate_hashid()

    def _aspycoco(self):
        # Converts to the official pycocotools.coco.COCO object
        from pycocotools import coco
        pycoco = coco.COCO()
        pycoco.dataset = self.dataset
        pycoco.createIndex()
        return pycoco


class MixinCocoAttrs(object):
    """
    Expose methods to construct object lists / groups
    """
    def annots(self, aids=None, gid=None):
        """
        Return boxes for annotations

        Example:
            >>> self = CocoDataset.demo()
            >>> annots = self.annots()
            >>> print(annots)
            <Annots(num=11)>
        """
        if aids is None and gid is not None:
            aids = self.gid_to_aids[gid]
        if aids is None:
            aids = sorted(self.anns.keys())
        return Annots(aids, self)

    def images(self, gids=None):
        """
        Return boxes for annotations

        Example:
            >>> self = CocoDataset.demo()
            >>> images = self.images()
            >>> print(images)
            <Images(num=3)>
        """
        if gids is None:
            gids = sorted(self.imgs.keys())
        return Images(gids, self)


class MixinCocoStats(object):
    """
    Methods for getting stats about the dataset
    """

    @property
    def n_annots(self):
        return len(self.dataset['annotations'])

    @property
    def n_images(self):
        return len(self.dataset['images'])

    @property
    def n_cats(self):
        return len(self.dataset['categories'])

    def category_annotation_frequency(self):
        """
        Reports the number of annotations of each category

        Example:
            >>> self = CocoDataset.demo()
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.repr2(hist))
            {
                'astroturf': 0,
                'astronaut': 1,
                'astronomer': 1,
                'helmet': 1,
                'rocket': 1,
                'mouth': 2,
                'star': 5,
            }
        """
        catname_to_nannots = ub.map_keys(lambda x: self.cats[x]['name'],
                                         ub.map_vals(len, self.cid_to_aids))
        catname_to_nannots = ub.odict(sorted(catname_to_nannots.items(),
                                             key=lambda kv: (kv[1], kv[0])))
        return catname_to_nannots

    def category_annotation_type_frequency(self):
        """
        Reports the number of annotations of each type for each category

        Example:
            >>> self = CocoDataset.demo()
            >>> hist = self.category_annotation_frequency()
            >>> print(ub.repr2(hist))
        """
        catname_to_nannot_types = {}
        for cid, aids in self.cid_to_aids.items():
            name = self.cats[cid]['name']
            hist = ub.dict_hist(map(annot_type, ub.take(self.anns, aids)))
            catname_to_nannot_types[name] = ub.map_keys(
                lambda k: k[0] if len(k) == 1 else k, hist)
        return catname_to_nannot_types

    def basic_stats(self):
        """
        Reports number of images, annotations, and categories.

        Example:
            >>> self = CocoDataset.demo()
            >>> print(ub.repr2(self.basic_stats()))
            {
                'n_anns': 11,
                'n_imgs': 3,
                'n_cats': 7,
            }
        """
        return ub.odict([
            ('n_anns', self.n_annots),
            ('n_imgs', self.n_images),
            ('n_cats', self.n_cats),
        ])

    def extended_stats(self):
        """
        Reports number of images, annotations, and categories.

        Example:
            >>> self = CocoDataset.demo()
            >>> print(ub.repr2(self.extended_stats()))
        """
        def mapping_stats(xid_to_yids):
            import netharn as nh
            n_yids = list(ub.map_vals(len, xid_to_yids).values())
            return nh.util.stats_dict(n_yids, n_extreme=True)
        return ub.odict([
            ('annots_per_img', mapping_stats(self.gid_to_aids)),
            # ('cats_per_img', mapping_stats(self.cid_to_gids)),
            ('cats_per_annot', mapping_stats(self.cid_to_aids)),
        ])


class _NextId(object):
    """ Helper class to tracks unused ids for new items """
    def __init__(self, parent):
        self.parent = parent
        self.unused = {
            'cid': None,
            'gid': None,
            'aid': None,
        }

    def set(self, key):
        # Determines what the next safe id can be
        key2 = {'aid': 'annotations', 'gid': 'images',
                'cid': 'categories'}[key]
        item_list = self.parent.dataset[key2]
        max_id = max(item['id'] for item in item_list) if item_list else 0
        next_id = max(max_id + 1, len(item_list))
        self.unused[key] = next_id
        # for i in it.count(len(self.cats) + 1):
        #     if i not in self.cats:
        #         return i

    def get(self, key):
        """ Get the next safe item id """
        if self.unused[key] is None:
            self.set(key)
        new_id = self.unused[key]
        self.unused[key] += 1
        return new_id


class MixinCocoDraw(object):
    """
    Matplotlib / display functionality
    """
    def show_image(self, gid=None, aids=None, aid=None):
        """
        Use matplotlib to show an image with annotations overlaid

        Args:
            gid (int): image to show
            aids (list): aids to highlight within the image
            aid (int): a specific aid to focus on. If gid is not give,
                look up gid based on this aid.
        """
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from PIL import Image

        if gid is None:
            primary_ann = self.anns[aid]
            gid = primary_ann['image_id']

        highlight_aids = set()
        if aid is not None:
            highlight_aids.add(aid)
        if aids is not None:
            highlight_aids.update(aids)

        img = self.imgs[gid]
        aids = self.gid_to_aids.get(img['id'], [])

        # Collect annotation overlays
        colored_segments = ub.ddict(list)
        keypoints = []
        rects = []
        texts = []
        for aid in aids:
            ann = self.anns[aid]
            # Note standard coco bbox is [x,y,width,height]
            if 'bbox' in ann:
                x1, y1 = ann['bbox'][0:2]
            elif 'line' in ann:
                x1, y1 = ann['line'][0:2]
            elif 'keypoints' in ann:
                kpts = np.array(ann['keypoints']).reshape(-1, 3)
                xys = kpts.T[0:2].T
                x1, y1 = xys.min(axis=0)

            catname = self.cats[ann['category_id']]['name']
            textkw = {
                'horizontalalignment': 'left',
                'verticalalignment': 'top',
                'backgroundcolor': (0, 0, 0, .3),
                'color': 'white',
                'fontproperties': mpl.font_manager.FontProperties(
                    size=6, family='monospace'),
            }
            texts.append((x1, y1, catname, textkw))

            color = 'orange' if aid in highlight_aids else 'blue'
            if 'bbox' in ann:
                [x, y, w, h] = ann['bbox']
                rect = mpl.patches.Rectangle((x, y), w, h, facecolor='none',
                                             edgecolor=color)
                rects.append(rect)
            if 'obox' in ann:
                # Oriented bounding box
                segs = np.array(ann['obox']).reshape(-1, 3)[:, 0:2]
                for pt1, pt2 in ub.iter_window(segs, wrap=True):
                    colored_segments[color].append([pt1, pt2])
            if 'line' in ann:
                x1, y1, x2, y2 = ann['line']
                pt1, pt2 = (x1, y1), (x2, y2)
                colored_segments[color].append([pt1, pt2])
            if 'keypoints' in ann:
                kpts = np.array(ann['keypoints']).reshape(-1, 3)
                xys = kpts.T[0:2].T
                keypoints.append(xys)

        # Show image
        gpath = join(self.img_root, img['file_name'])
        with Image.open(gpath) as pil_img:
            np_img = np.array(pil_img)

        fig = plt.gcf()
        ax = fig.gca()
        ax.cla()
        plt.imshow(np_img)

        # Show all annotations inside it
        for (x1, y1, catname, textkw) in texts:
            ax.text(x1, y1, catname, **textkw)

        for color, segments in colored_segments.items():
            line_col = mpl.collections.LineCollection(segments, 2, color=color)
            ax.add_collection(line_col)

        rect_col = mpl.collections.PatchCollection(rects, match_original=True)
        ax.add_collection(rect_col)
        if keypoints:
            xs, ys = np.vstack(keypoints).T
            ax.plot(xs, ys, 'bo')


class MixinCocoAddRemove(object):
    """
    Mixin functions to dynamically add / remove annotations images and
    categories while maintaining lookup indexes.
    """

    def add_image(self, gname, gid=None, **kw):
        """
        Add an image to the dataset (dynamically updates the index)

        Args:
            gname (str): image name
            gid (None or int): ADVANCED. Force using this image id.

        Example:
            >>> self = CocoDataset.demo()
            >>> import netharn as nh
            >>> gname = nh.util.grab_test_image_fpath('paraview')
            >>> gid = self.add_image(gname)
            >>> assert self.imgs[gid]['file_name'] == gname
        """
        if gid is None:
            gid = self._next_ids.get('gid')
        elif self.imgs and gid in self.imgs:
            raise IndexError('Image id={} already exists'.format(gid))

        img = _dict()
        img['id'] = int(gid)
        img['file_name'] = str(gname)
        img.update(**kw)
        self.dataset['images'].append(img)
        self.index._add_image(gid, img)
        self._invalidate_hashid()
        return gid

    def add_annotation(self, gid, cid, bbox=None, aid=None, **kw):
        """
        Add an annotation to the dataset (dynamically updates the index)

        Args:
            gid (int): image_id to add to
            cid (int): category_id to add to
            bbox (list or nh.util.Boxes): bounding box in xywh format
            aid (None or int): ADVANCED. Force using this annotation id.

        Example:
            >>> self = CocoDataset.demo()
            >>> gid = 1
            >>> cid = 1
            >>> bbox = [10, 10, 20, 20]
            >>> aid = self.add_annotation(gid, cid, bbox)
            >>> assert self.anns[aid]['bbox'] == bbox
        """
        if aid is None:
            aid = self._next_ids.get('aid')
        elif self.anns and aid in self.anns:
            raise IndexError('Annot id={} already exists'.format(aid))

        ann = _dict()
        ann['id'] = int(aid)
        ann['image_id'] = int(gid)
        ann['category_id'] = int(cid)
        if bbox:
            import netharn as nh
            if isinstance(bbox, nh.util.Boxes):
                bbox = bbox.to_xywh().data.tolist()
            ann['bbox'] = bbox
        # assert not set(kw).intersection(set(ann))
        ann.update(**kw)
        self.dataset['annotations'].append(ann)
        self.index._add_annotation(aid, gid, cid, ann)
        self._invalidate_hashid(['annotations'])
        return aid

    def add_annotations(self, anns):
        """
        Faster less-safe multi-item alternative

        Args:
            anns (List[Dict]): list of annotation dictionaries

        Example:
            >>> self = CocoDataset.demo()
            >>> anns = [self.anns[aid] for aid in [2, 3, 5, 7]]
            >>> self.remove_annotations(anns)
            >>> assert self.n_annots == 7 and self._check_index()
            >>> self.add_annotations(anns)
            >>> assert self.n_annots == 11 and self._check_index()
        """
        self.dataset['annotations'].extend(anns)
        self.index._add_annotations(anns)
        self._invalidate_hashid(['annotations'])

    def add_images(self, imgs):
        """
        Faster less-safe multi-item alternative

        Args:
            imgs (List[Dict]): list of image dictionaries

        Example:
            >>> imgs = CocoDataset.demo().dataset['images']
            >>> self = CocoDataset()
            >>> self.add_images(imgs)
            >>> assert self.n_images == 3 and self._check_index()
        """
        self.dataset['images'].extend(imgs)
        self.index._add_images(imgs)
        self._invalidate_hashid(['images'])

    def add_category(self, name, supercategory=None, cid=None):
        """
        Adds a category

        Args:
            name (str): name of the new category
            supercategory (str, optional): parent of this category
            cid (int, optional): use this category id, if it was not taken

        CommandLine:
            xdoctest -m netharn.data.coco_api MixinCocoAddRemove.add_category

        Example:
            >>> self = CocoDataset.demo()
            >>> cid = self.add_category('dog', supercategory='object')
            >>> assert self.cats[cid]['name'] == 'dog'
            >>> assert self.n_cats == 8
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     self.add_category('dog', supercategory='object')
        """
        index = self.index
        if index.cats and name in index.name_to_cat:
            raise ValueError('Category name={!r} already exists'.format(name))

        if cid is None:
            cid = self._next_ids.get('cid')
        elif index.cats and cid in index.cats:
            raise IndexError('Category id={} already exists'.format(cid))

        cat = _dict()
        cat['id'] = int(cid)
        cat['name'] = str(name)
        if supercategory:
            cat['supercategory'] = supercategory

        # Add to raw data structure
        self.dataset['categories'].append(cat)

        # And add to the indexes
        index._add_category(cid, name, cat)
        self._invalidate_hashid(['categories'])
        return cid

    def clear_images(self):
        """
        Removes all images and annotations (but not categories)

        Example:
            >>> self = CocoDataset.demo()
            >>> self.clear_images()
            >>> print(ub.repr2(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 0, n_cats: 7
        """
        # self.dataset['images'].clear()
        # self.dataset['annotations'].clear()
        del self.dataset['images'][:]
        del self.dataset['annotations'][:]
        self.index._remove_all_images()
        self._invalidate_hashid(['images', 'annotations'])

    def clear_annotations(self):
        """
        Removes all annotations (but not images and categories)

        Example:
            >>> self = CocoDataset.demo()
            >>> self.clear_annotations()
            >>> print(ub.repr2(self.basic_stats(), nobr=1, nl=0, si=1))
            n_anns: 0, n_imgs: 3, n_cats: 7
        """
        # self.dataset['annotations'].clear()
        del self.dataset['annotations'][:]
        self.index._remove_all_annotations()
        self._invalidate_hashid(['annotations'])

    remove_all_images = clear_images
    remove_all_annotations = clear_annotations

    def remove_annotation(self, aid_or_ann):
        """
        Remove a single annotation from the dataset

        If you have multiple annotations to remove its more efficient to remove
        them in batch with `self.remove_annotations`

        Example:
            >>> self = CocoDataset.demo()
            >>> aids_or_anns = [self.anns[2], 3, 4, self.anns[1]]
            >>> self.remove_annotations(aids_or_anns)
            >>> assert len(self.dataset['annotations']) == 7
            >>> self._check_index()
        """
        # Do the simple thing, its O(n) anyway,
        remove_ann = self._resolve_to_ann(aid_or_ann)
        self.dataset['annotations'].remove(remove_ann)
        self.index.clear()
        self._invalidate_hashid(['annotations'])

    def remove_annotations(self, aids_or_anns, verbose=0):
        """
        Remove multiple annotations from the dataset.

        Args:
            anns_or_aids (List): list of annotation dicts or ids

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> self = CocoDataset.demo()
            >>> aids_or_anns = [self.anns[2], 3, 4, self.anns[1]]
            >>> self.remove_annotations(aids_or_anns)
            >>> assert len(self.dataset['annotations']) == 7
            >>> self._check_index()
        """
        remove_info = {'annotations': None}
        # Do nothing if given no input
        if aids_or_anns:
            # build mapping from aid to index O(n)
            # TODO: it would be nice if this mapping was as part of the index.
            aid_to_index = {
                ann['id']: index
                for index, ann in enumerate(self.dataset['annotations'])
            }
            remove_aids = list(map(self._resolve_to_id, aids_or_anns))
            remove_info['annotations'] = len(remove_aids)

            # Lookup the indices to remove, sort in descending order
            if verbose > 1:
                print('Removing {} annotations'.format(len(remove_aids)))

            remove_idxs = list(ub.take(aid_to_index, remove_aids))
            delitems(self.dataset['annotations'], remove_idxs)

            self.index._remove_annotations(remove_aids, verbose=verbose)
            self._invalidate_hashid(['annotations'])
        return remove_info

    def remove_categories(self, cids_or_cats, verbose=0):
        """
        Remove categories and all annotations in those categories.
        Currently does not change any hierarcy information

        Args:
            cids_or_cats (List): list of category dicts, names, or ids

        Returns:
            Dict: num_removed: information on the number of items removed

        Example:
            >>> self = CocoDataset.demo()
            >>> cids_or_cats = [self.cats[1], 'rocket', 3]
            >>> self.remove_categories(cids_or_cats)
            >>> assert len(self.dataset['categories']) == 4
            >>> self._check_index()
        """
        remove_info = {'annotations': None, 'categories': None}
        if cids_or_cats:

            if verbose > 1:
                print('Removing annots of removed categories')

            remove_cids = list(map(self._resolve_to_cid, cids_or_cats))
            # First remove any annotation that belongs to those categories
            if self.cid_to_aids:
                remove_aids = list(it.chain(*[self.cid_to_aids[cid]
                                              for cid in remove_cids]))
            else:
                remove_aids = [ann['id'] for ann in self.dataset['annotations']
                               if ann['category_id'] in remove_cids]

            rminfo = self.remove_annotations(remove_aids, verbose=verbose)
            remove_info.update(rminfo)

            remove_info['categories'] = len(remove_cids)
            if verbose > 1:
                print('Removing {} category entries'.format(len(remove_cids)))
            cid_to_index = {
                cat['id']: index
                for index, cat in enumerate(self.dataset['categories'])
            }
            # Lookup the indices to remove, sort in descending order
            remove_idxs = list(ub.take(cid_to_index, remove_cids))
            delitems(self.dataset['categories'], remove_idxs)

            self.index._remove_categories(remove_cids, verbose=verbose)
            self._invalidate_hashid(['categories', 'annotations'])

        return remove_info


class CocoIndex(object):
    """
    Fast lookup index for the COCO dataset with dynamic modification
    """

    # _set = ub.oset  # many operations are much slower for oset
    _set = set

    def __init__(self):
        self.anns = None
        self.imgs = None
        self.cats = None
        self.gid_to_aids = None
        self.cid_to_aids = None
        self.name_to_cat = None
        self.file_name_to_img = None

    def __bool__(self):
        return self.anns is not None

    __nonzero__ = __bool__  # python 2 support

    def _add_image(self, gid, img):
        if self.imgs is not None:
            self.imgs[gid] = img
            self.gid_to_aids[gid] = self._set()
            self.file_name_to_img[img['file_name']] = img

    def _add_images(self, imgs):
        if self.imgs is not None:
            gids = [img['id'] for img in imgs]
            new_imgs = dict(zip(gids, imgs))
            self.imgs.update(new_imgs)
            self.file_name_to_img.update(
                {img['file_name']: img for img in imgs})
            for gid in gids:
                self.gid_to_aids[gid] = self._set()

    def _add_annotation(self, aid, gid, cid, ann):
        if self.anns is not None:
            self.anns[aid] = ann
            self.gid_to_aids[gid].add(aid)
            self.cid_to_aids[cid].add(aid)

    def _add_annotations(self, anns):
        if self.anns is not None:
            aids = [ann['id'] for ann in anns]
            gids = [ann['image_id'] for ann in anns]
            cids = [ann['category_id'] for ann in anns]
            new_anns = dict(zip(aids, anns))
            self.anns.update(new_anns)
            for gid, cid, aid in zip(gids, cids, aids):
                self.gid_to_aids[gid].add(aid)
                self.cid_to_aids[cid].add(aid)

    def _add_category(self, cid, name, cat):
        if self.cats is not None:
            self.cats[cid] = cat
            self.cid_to_aids[cid] = self._set()
            self.name_to_cat[name] = cat

    def _remove_all_annotations(self):
        # Keep the category and image indexes alive
        if self.anns is not None:
            self.anns.clear()
            for _ in self.gid_to_aids.values():
                _.clear()
            for _ in self.cid_to_aids.values():
                _.clear()

    def _remove_all_images(self):
        # Keep the category indexes alive
        if self.imgs is not None:
            self.imgs.clear()
            self.anns.clear()
            self.gid_to_aids.clear()
            self.file_name_to_img.clear()
            for _ in self.cid_to_aids.values():
                _.clear()

    def _remove_annotations(self, remove_aids, verbose=0):
        if self.anns is not None:
            if verbose > 1:
                print('Updating annotation index')
            # This is faster for simple set cid_to_aids
            for aid in remove_aids:
                ann = self.anns.pop(aid)
                gid = ann['image_id']
                cid = ann['category_id']
                self.cid_to_aids[cid].remove(aid)
                self.gid_to_aids[gid].remove(aid)

    def _remove_categories(self, remove_cids, verbose=0):
        # dynamically update the category index
        if self.cats is not None:
            for cid in remove_cids:
                cat = self.cats.pop(cid)
                del self.cid_to_aids[cid]
                del self.name_to_cat[cat['name']]
            if verbose > 2:
                print('Updated category index')

    def clear(self):
        self.anns = None
        self.imgs = None
        self.cats = None
        self.gid_to_aids = None
        self.cid_to_aids = None
        self.name_to_cat = None
        self.file_name_to_img = None

    def build(self, parent):
        """
        build reverse indexes

        Notation:
            aid - Annotation ID
            gid - imaGe ID
            cid - Category ID
        """
        # create index
        anns, cats, imgs = {}, {}, {}
        gid_to_aids = ub.ddict(self._set)
        cid_to_aids = ub.ddict(self._set)

        # Build one-to-one self-lookup maps
        for cat in parent.dataset.get('categories', []):
            cid = cat['id']
            if cid in cat:
                warnings.warn(
                    'Categories have the same id in {}:\n{} and\n{}'.format(
                        parent, cats[cid], cat))
            cats[cid] = cat

        for img in parent.dataset.get('images', []):
            gid = img['id']
            if gid in imgs:
                warnings.warn(
                    'Images have the same id in {}:\n{} and\n{}'.format(
                        parent, imgs[gid], img))
            imgs[gid] = img

        for ann in parent.dataset.get('annotations', []):
            aid = ann['id']
            if aid in anns:
                warnings.warn(
                    'Annotations at index {} and {} '
                    'have the same id in {}:\n{} and\n{}'.format(
                        parent.dataset['annotations'].index(anns[aid]),
                        parent.dataset['annotations'].index(ann),
                        parent, anns[aid], ann))
            anns[aid] = ann

        # Build one-to-many lookup maps
        for ann in anns.values():
            try:
                aid = ann['id']
                gid = ann['image_id']
                cid = ann['category_id']
            except KeyError:
                raise KeyError('Annotation does not have ids {}'.format(ann))

            if not isinstance(aid, INT_TYPES):
                raise TypeError('bad aid={} type={}'.format(aid, type(aid)))
            if not isinstance(gid, INT_TYPES):
                raise TypeError('bad gid={} type={}'.format(gid, type(gid)))
            if not isinstance(cid, INT_TYPES):
                raise TypeError('bad cid={} type={}'.format(cid, type(cid)))

            gid_to_aids[gid].add(aid)
            cid_to_aids[cid].add(aid)
            if gid not in imgs:
                warnings.warn('Annotation {} in {} references '
                              'unknown image_id'.format(ann, parent))
            if cid not in cats:
                warnings.warn('Annotation {} in {} references '
                              'unknown category_id'.format(ann, parent))

        # Fix one-to-zero cases
        for cid in cats.keys():
            if cid not in cid_to_aids:
                cid_to_aids[cid] = self._set()

        for gid in imgs.keys():
            if gid not in gid_to_aids:
                gid_to_aids[gid] = self._set()

        # create class members
        self.anns = anns
        self.imgs = imgs
        self.cats = cats
        self.gid_to_aids = gid_to_aids
        self.cid_to_aids = cid_to_aids
        self.name_to_cat = {cat['name']: cat for cat in self.cats.values()}
        self.file_name_to_img = {img['file_name']: img for img in self.imgs.values()}


class MixinCocoIndex(object):
    """
    Give the dataset top level access to index attributes
    """
    @property
    def anns(self):
        return self.index.anns

    @property
    def imgs(self):
        return self.index.imgs

    @property
    def cats(self):
        return self.index.cats

    @property
    def gid_to_aids(self):
        return self.index.gid_to_aids

    @property
    def cid_to_aids(self):
        return self.index.cid_to_aids

    @property
    def name_to_cat(self):
        return self.index.name_to_cat


class CocoDataset(ub.NiceRepr, MixinCocoAddRemove, MixinCocoStats,
                  MixinCocoAttrs, MixinCocoDraw, MixinCocoExtras,
                  MixinCocoIndex, MixinCocoDepricate):
    """
    Notes:
        A keypoint annotation
            {
                "image_id" : int,
                "category_id" : int,
                "keypoints" : [x1,y1,v1,...,xk,yk,vk],
                "score" : float,
            }
            Note that `vi` is a visibility flag.

        A bounding box annotation
            {
                "image_id" : int,
                "category_id" : int,
                "bbox" : [x,y,width,height],
                "score" : float,
            }

        We also define a non-standard "line" annotation (which
            our fixup scripts will interpret as the diameter of a circle to
            convert into a bounding box)

        A line* annotation (note this is a non-standard field)
            {
                "image_id" : int,
                "category_id" : int,
                "line" : [x1,y1,x2,y2],
                "score" : float,
            }

        Lastly, note that our datasets will sometimes specify multiple bbox,
        line, and/or, keypoints fields. In this case we may also specify a
        field roi_shape, which denotes which field is the "main" annotation
        type.

    References:
        http://cocodataset.org/#format
        http://cocodataset.org/#download

    CommandLine:
        python -m netharn.data.coco_api CocoDataset --show

    Example:
        >>> dataset = demo_coco_data()
        >>> self = CocoDataset(dataset, tag='demo')
        >>> # xdoctest: +REQUIRES(--show)
        >>> self.show_image(gid=2)
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
    """
    def __init__(self, data=None, tag=None, img_root=None, autobuild=True):
        if data is None:
            data = {
                'categories': [],
                'images': [],
                'annotations': [],
                'licenses': [],
                'info': [],
            }
        if isinstance(data, six.string_types):
            # HACK: rework this
            fpath = data
            key = basename(fpath).split('.')[0]
            data = json.load(open(fpath, 'r'))
            if tag is None:
                tag = key
            # if img_root is None:
            #     img_root = join('.', key)
        else:
            if not isinstance(data, dict):
                raise TypeError('data must be a dict or path to json file')

        if img_root is None:
            if 'img_root' in data:
                # allow image root to be specified in the dataset
                img_root = data['img_root']
            else:
                img_root = '.'

        self.index = CocoIndex()

        self.hashid = None
        self.hashid_parts = None

        self.tag = tag
        self.dataset = data
        self.img_root = img_root

        # Keep track of an unused id we may use
        self._next_ids = _NextId(self)

        if autobuild:
            self._build_index()

    def copy(self):
        """
        Example:
            >>> from netharn.data.coco_api import *
            >>> self = CocoDataset.demo()
            >>> new = self.copy()
            >>> assert new.imgs[1] is new.dataset['images'][0]
            >>> assert new.imgs[1] == self.dataset['images'][0]
            >>> assert new.imgs[1] is not self.dataset['images'][0]
        """
        new = copy.copy(self)
        new.index = CocoIndex()
        new.dataset = copy.deepcopy(self.dataset)
        new._build_index()
        return new

    def __nice__(self):
        parts = []
        parts.append('tag={}'.format(self.tag))
        if self.dataset is not None:
            info = ub.repr2(self.basic_stats(), kvsep='=', si=1, nobr=1, nl=0)
            parts.append(info)
        return ', '.join(parts)

    def dumps(self, indent=None, newlines=False):
        """
        Writes the dataset out to the json format

        Args:
            newlines (bool) :
                if True, each annotation, image, category gets its own line

        Notes:
            Using newlines=True is similar to:
                print(ub.repr2(dset.dataset, nl=2, trailsep=False))
                However, the above may not output valid json if it contains
                ndarrays.

        Example:
            >>> from netharn.data.coco_api import *
            >>> self = CocoDataset.demo()
            >>> text = self.dumps()
            >>> print(text)
            >>> self2 = CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

            >>> text = self.dumps(newlines=True)
            >>> print(text)
            >>> self2 = CocoDataset(json.loads(text), tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset
        """
        def _json_dumps(data, indent=None):
            fp = StringIO()
            json.dump(data, fp, indent=indent)
            fp.seek(0)
            text = fp.read()
            return text

        # Instead of using json to dump the whole thing make the text a bit
        # more pretty.
        if newlines:
            if indent is None:
                indent = ''
            if isinstance(indent, int):
                indent = ' ' * indent
            dict_lines = []
            main_keys = ['info', 'licenses', 'categories', 'images',
                         'annotations']
            other_keys = sorted(set(self.dataset.keys()) - set(main_keys))
            for key in main_keys:
                if key not in self.dataset:
                    continue
                # We know each main entry is a list, so make it such that
                # Each entry gets its own line
                value = self.dataset[key]
                value_lines = [_json_dumps(v) for v in value]
                if value_lines:
                    value_body = (',\n' + indent).join(value_lines)
                    value_repr = '[\n' + indent + value_body + '\n]'
                else:
                    value_repr = '[]'
                item_repr = '{}: {}'.format(_json_dumps(key), value_repr)
                dict_lines.append(item_repr)

            for key in other_keys:
                # Dont assume anything about other data
                value = self.dataset.get(key, [])
                value_repr = _json_dumps(value)
                item_repr = '{}: {}'.format(_json_dumps(key), value_repr)
                dict_lines.append(item_repr)
            text = '{\n' + ',\n'.join(dict_lines) + '\n}'
        else:
            text = _json_dumps(self.dataset, indent=indent)

        return text

    def dump(self, file, indent=None, newlines=False):
        """
        Writes the dataset out to the json format

        Args:
            file (PathLike | FileLike):
                Where to write the data.  Can either be a path to a file or an
                open file pointer / stream.

            newlines (bool) : if True, each annotation, image, category gets
                its own line.

        Example:
            >>> import tempfile
            >>> from netharn.data.coco_api import *
            >>> self = CocoDataset.demo()
            >>> file = tempfile.NamedTemporaryFile('w')
            >>> self.dump(file)
            >>> file.seek(0)
            >>> text = open(file.name, 'r').read()
            >>> print(text)
            >>> file.seek(0)
            >>> dataset = json.load(open(file.name, 'r'))
            >>> self2 = CocoDataset(dataset, tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset

            >>> file = tempfile.NamedTemporaryFile('w')
            >>> self.dump(file, newlines=True)
            >>> file.seek(0)
            >>> text = open(file.name, 'r').read()
            >>> print(text)
            >>> file.seek(0)
            >>> dataset = json.load(open(file.name, 'r'))
            >>> self2 = CocoDataset(dataset, tag='demo2')
            >>> assert self2.dataset == self.dataset
            >>> assert self2.dataset is not self.dataset
        """
        if isinstance(file, six.string_types):
            with open(file, 'w') as fp:
                self.dump(fp, indent=indent, newlines=newlines)
        else:
            if newlines:
                file.write(self.dumps(indent=indent, newlines=newlines))
            else:
                json.dump(self.dataset, file, indent=indent)

    def _check_index(self):
        # We can verify our index invariants by copying the raw dataset and
        # checking if the newly constructed index is the same as this index.
        new = copy.copy(self)
        new.dataset = copy.deepcopy(self.dataset)
        new._build_index()
        assert self.index.anns == new.index.anns
        assert self.index.imgs == new.index.imgs
        assert self.index.cats == new.index.cats
        assert self.index.gid_to_aids == new.index.gid_to_aids
        assert self.index.cid_to_aids == new.index.cid_to_aids
        assert self.index.name_to_cat == new.index.name_to_cat
        assert self.index.file_name_to_img == new.index.file_name_to_img
        return True

    def _check_pointers(self, verbose=1):
        """
        Check that all category and image ids referenced by annotations exist
        """
        if not self.index:
            raise Exception('Build index before running pointer check')
        errors = []
        annots = self.dataset['annotations']
        iter_ = ub.ProgIter(annots, desc='check annots', enabled=verbose)
        for ann in iter_:
            aid = ann['id']
            cid = ann['category_id']
            gid = ann['image_id']

            if cid not in self.cats:
                errors.append('aid={} references bad cid={}'.format(aid, cid))
            else:
                if self.cats[cid]['id'] != cid:
                    errors.append('cid={} has a bad index'.format(cid))

            if gid not in self.imgs:
                errors.append('aid={} references bad gid={}'.format(aid, gid))
            else:
                if self.imgs[gid]['id'] != gid:
                    errors.append('gid={} has a bad index'.format(gid))
        if errors:
            raise Exception('\n'.join(errors))
        elif verbose:
            print('Pointers are consistent')
        return True

    def _build_index(self):
        self.index.build(self)

    def _clear_index(self):
        self.index.clear()

    @classmethod
    def union(CocoDataset, *others, **kw):
        """
        Merges multiple `CocoDataset` items into one. Names and associations
        are retained, but ids may be different.

        TODO: are supercategories broken?
        """
        # if hasattr(CocoDataset, '__class__'):
        #     # This is an instance not an object
        #     return CocoDataset.__class__.union(CocoDataset, *others, **kw)

        def _coco_union(relative_dsets):
            """ union of dictionary based data structure """
            merged = _dict([
                ('categories', []),
                ('licenses', []),
                ('info', []),
                ('images', []),
                ('annotations', []),
            ])

            merged_cat_name_to_id = {}

            def update_ifnotin(d1, d2):
                """ copies keys from d2 that doent exist in d1 into d1 """
                for k, v in d2.items():
                    if k not in d1:
                        d1[k] = v
                return d1

            for subdir, old_dset in relative_dsets:
                # Create temporary indexes to map from old to new
                cat_id_map = {}
                img_id_map = {}

                # Add the licenses / info into the merged dataset
                # Licenses / info are unused in our datas, so this might not be
                # correct
                merged['licenses'].extend(old_dset.get('licenses', []))
                merged['info'].extend(old_dset.get('info', []))

                # Add the categories into the merged dataset
                for old_cat in old_dset['categories']:
                    new_id = merged_cat_name_to_id.get(old_cat['name'], None)
                    # The same category might exist in different datasets.
                    if new_id is None:
                        # Only add if it does not yet exist
                        new_id = len(merged_cat_name_to_id) + 1
                        merged_cat_name_to_id[old_cat['name']] = new_id
                        new_cat = _dict([
                            ('id', new_id),
                            ('name', old_cat['name']),
                            # ('supercategory', old_cat['supercategory']),
                        ])
                        update_ifnotin(new_cat, old_cat)
                        merged['categories'].append(new_cat)
                    cat_id_map[old_cat['id']] = new_id

                # Add the images into the merged dataset
                for old_img in old_dset['images']:
                    new_img = _dict([
                        ('id', len(merged['images']) + 1),
                        ('file_name', join(subdir, old_img['file_name'])),
                    ])
                    # copy over other metadata
                    update_ifnotin(new_img, old_img)
                    img_id_map[old_img['id']] = new_img['id']
                    merged['images'].append(new_img)

                # Add the annotations into the merged dataset
                for old_annot in old_dset['annotations']:
                    old_cat_id = old_annot['category_id']
                    old_img_id = old_annot['image_id']
                    new_cat_id = cat_id_map.get(old_cat_id, None)
                    new_img_id = img_id_map.get(old_img_id, None)
                    if new_cat_id is None:
                        warnings.warn('annot {} in {} has bad category-id {}'.format(
                            old_annot, subdir, old_cat_id))
                        # raise Exception
                    if new_img_id is None:
                        warnings.warn('annot {} in {} has bad image-id {}'.format(
                            old_annot, subdir, old_img_id))
                        # sanity check:
                        # if any(img['id'] == old_img_id for img in old_dset['images']):
                        #     raise Exception('Image id {} does not exist in {}'.format(old_img_id, subdir))
                    new_annot = _dict([
                        ('id', len(merged['annotations']) + 1),
                        ('image_id', new_img_id),
                        ('category_id', new_cat_id),
                    ])
                    update_ifnotin(new_annot, old_annot)
                    merged['annotations'].append(new_annot)
            return merged

        relative_dsets = [(d.img_root, d.dataset) for d in others]
        merged = _coco_union(relative_dsets)
        return CocoDataset(merged, **kw)

    def subset(self, gids, copy=False):
        """
        Return a subset of the larger coco dataset by specifying which images
        to port. All annotations in those images will be taken.

        Args:
            gids (List[int]): image-ids to copy into a new dataset
            copy (bool, default=False): if True, makes a deep copy of
                all nested attributes, otherwise makes a shallow copy.

        Example:
            >>> self = CocoDataset.demo()
            >>> gids = [1, 3]
            >>> sub_dset = self.subset(gids)
            >>> assert len(self.gid_to_aids) == 3
            >>> assert len(sub_dset.gid_to_aids) == 2

        Example:
            >>> self = CocoDataset.demo()
            >>> sub1 = self.subset([1])
            >>> sub2 = self.subset([2])
            >>> sub3 = self.subset([3])
            >>> others = [sub1, sub2, sub3]
            >>> rejoined = CocoDataset.union(*others)
            >>> assert len(sub1.anns) == 9
            >>> assert len(sub2.anns) == 2
            >>> assert len(sub3.anns) == 0
            >>> assert rejoined.basic_stats() == self.basic_stats()
        """
        new_dataset = _dict([(k, []) for k in self.dataset])
        new_dataset['categories'] = self.dataset['categories']
        new_dataset['info'] = self.dataset.get('info', [])
        new_dataset['licenses'] = self.dataset.get('licenses', [])

        gids = sorted(set(gids))
        sub_aids = sorted([aid for gid in gids
                           for aid in self.gid_to_aids.get(gid, [])])
        new_dataset['annotations'] = list(ub.take(self.anns, sub_aids))
        new_dataset['images'] = list(ub.take(self.imgs, gids))

        if copy:
            from copy import deepcopy
            new_dataset = deepcopy(new_dataset)

        sub_dset = CocoDataset(new_dataset, img_root=self.img_root)
        return sub_dset


def delitems(items, remove_idxs, thresh=750):
    if len(remove_idxs) > thresh:
        # Its typically faster to just make a new list when there are
        # lots and lots of items to remove.
        keep_idxs = sorted(set(range(len(items))) - set(remove_idxs))
        newlist = [items[idx] for idx in keep_idxs]
        items[:] = newlist
    else:
        # However, when there are a few hundred items to remove, del is faster.
        for idx in sorted(remove_idxs, reverse=True):
            del items[idx]


def demo_coco_data():
    """
    Simple data for testing
    """
    import netharn as nh
    gpath1 = nh.util.grab_test_image_fpath('astro')
    gpath2 = nh.util.grab_test_image_fpath('carl')
    gpath3 = nh.util.grab_test_image_fpath('stars')
    # gpath1 = ub.grabdata('https://i.imgur.com/KXhKM72.png')
    # gpath2 = ub.grabdata('https://i.imgur.com/flTHWFD.png')
    # gpath3 = ub.grabdata('https://i.imgur.com/kCi7C1r.png')

    from os.path import commonprefix, relpath
    img_root = commonprefix([gpath1, gpath2, gpath3])

    # Make file names relative for consistent testing purpose
    gname1 = relpath(gpath1, img_root)
    gname2 = relpath(gpath2, img_root)
    gname3 = relpath(gpath3, img_root)

    dataset = {
        'img_root': img_root,

        'categories': [
            {'id': 1, 'name': 'astronaut', 'supercategory': 'human'},
            {'id': 2, 'name': 'rocket', 'supercategory': 'object'},
            {'id': 3, 'name': 'helmet', 'supercategory': 'object'},
            {'id': 4, 'name': 'mouth', 'supercategory': 'human'},
            {'id': 5, 'name': 'star', 'supercategory': 'object'},
            {'id': 6, 'name': 'astronomer', 'supercategory': 'human'},
            {'id': 7, 'name': 'astroturf', 'supercategory': 'object'},
        ],
        'images': [
            {'id': 1, 'file_name': gname1},
            {'id': 2, 'file_name': gname2},
            {'id': 3, 'file_name': gname3},
        ],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1,
             'bbox': [10, 10, 360, 490]},
            {'id': 2, 'image_id': 1, 'category_id': 2,
             'bbox': [350, 5, 130, 290]},
            {'id': 3, 'image_id': 1, 'category_id': 3,
             'line': [326, 369, 500, 500]},
            {'id': 4, 'image_id': 1, 'category_id': 4,
             'keypoints': [202, 139, 1, 215, 150, 1, 229, 150, 1, 244, 142, 1]},
            {'id': 5, 'image_id': 1, 'category_id': 5,
             'keypoints': [37, 65, 1]},
            {'id': 6, 'image_id': 1, 'category_id': 5,
             'keypoints': [37, 16, 1]},
            {'id': 7, 'image_id': 1, 'category_id': 5,
             'keypoints': [3, 9, 1]},
            {'id': 8, 'image_id': 1, 'category_id': 5,
             'keypoints': [2, 111, 1]},
            {'id': 9, 'image_id': 1, 'category_id': 5,
             'keypoints': [2, 60, 1]},
            {'id': 10, 'image_id': 2, 'category_id': 6,
             'bbox': [37, 6, 230, 240]},
            {'id': 11, 'image_id': 2, 'category_id': 4,
             'bbox': [124, 96, 45, 18]}
        ],
        'licenses': [],
        'info': [],
    }
    return dataset


if __name__ == '__main__':
    r"""
    CommandLine:
        xdoctest netharn.data.coco_api all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
