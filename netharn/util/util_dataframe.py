"""
mkinit ~/code/kwarray/kwarray/__init__.py --relative --nomods -w
mkinit ~/code/netharn/netharn/util/util_dataframe.py --relative --nomods

"""
# this module is a hack for backwards compatibility
from kwarray.dataframe_light import DataFrameLight, DataFrameArray, LocLight
__all__ = ['DataFrameLight', 'DataFrameArray', 'LocLight']
