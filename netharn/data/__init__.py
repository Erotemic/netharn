"""
mkinit netharn.data')"
"""
# flake8: noqa

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.data import base
    from netharn.data import coco_api
    from netharn.data import collate
    from netharn.data import toydata
    from netharn.data import transforms
    from netharn.data import voc

    from netharn.data.base import (DataMixin,)
    from netharn.data.coco_api import (CocoDataset, CocoExtrasMixin, annot_type,
                                       demo_coco_data,)
    from netharn.data.collate import (CollateException, default_collate,
                                      list_collate, padded_collate,)
    from netharn.data.toydata import (ToyData1d, ToyData2d,)
    from netharn.data.voc import (VOCDataset,)

    __all__ = ['CocoDataset', 'CocoExtrasMixin', 'CollateException', 'DataMixin',
               'ToyData1d', 'ToyData2d', 'VOCDataset', 'annot_type', 'base',
               'coco_api', 'collate', 'default_collate', 'demo_coco_data',
               'list_collate', 'padded_collate', 'toydata', 'transforms', 'voc']
