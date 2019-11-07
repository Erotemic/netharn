"""
mkinit netharn.util
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__extra_all__ = [
    'profiler',
]


# <AUTOGEN_INIT>
from netharn.util import imutil
from netharn.util import mpl_3d
from netharn.util import mpl_multiplot
from netharn.util import mplutil
from netharn.util import util_averages
from netharn.util import util_boxes
from netharn.util import util_cachestamp
from netharn.util import util_cv2
from netharn.util import util_dataframe
from netharn.util import util_demodata
from netharn.util import util_filesys
from netharn.util import util_fname
from netharn.util import util_groups
from netharn.util import util_idstr
from netharn.util import util_io
from netharn.util import util_iter
from netharn.util import util_json
from netharn.util import util_misc
from netharn.util import util_nms
from netharn.util import util_numpy
from netharn.util import util_random
from netharn.util import util_resources
from netharn.util import util_slider
from netharn.util import util_subextreme
from netharn.util import util_tensorboard
from netharn.util import util_torch
from netharn.util import util_zip

from netharn.util.imutil import (CV2_INTERPOLATION_TYPES, adjust_gamma,
                                 atleast_3channels, convert_colorspace,
                                 ensure_alpha_channel, ensure_float01,
                                 ensure_grayscale, get_num_channels,
                                 image_slices, imread, imscale, imwrite,
                                 load_image_paths, make_channels_comparable,
                                 num_channels, overlay_alpha_images,
                                 overlay_colorized, run_length_encoding,
                                 stack_images, stack_images_grid,
                                 stack_multiple_images, wide_strides_1d,)
from netharn.util.mpl_3d import (plot_surface3d,)
from netharn.util.mpl_multiplot import (is_list_of_lists, is_list_of_scalars,
                                        is_listlike, multi_plot,)
from netharn.util.mplutil import (Color, PlotNums, adjust_subplots, aggensure,
                                  autompl, axes_extent, colorbar,
                                  colorbar_image, copy_figure_to_clipboard,
                                  dict_intersection, distinct_colors,
                                  distinct_markers, draw_border, draw_boxes,
                                  draw_line_segments, ensure_fnum,
                                  extract_axes_extents, figure, imshow,
                                  interpolated_colormap, legend, make_heatmask,
                                  make_legend_img, next_fnum,
                                  pandas_plot_matrix, qtensure,
                                  render_figure_to_image, reverse_colormap,
                                  save_parts, savefig2, scores_to_cmap,
                                  scores_to_color, set_figtitle,
                                  set_mpl_backend, show_if_requested,)
from netharn.util.util_averages import (CumMovingAve, ExpMovingAve,
                                        InternalRunningStats, MovingAve,
                                        RunningStats, WindowedMovingAve,
                                        absdev, stats_dict,)
from netharn.util.util_boxes import (Boxes, TORCH_HAS_EMPTY_SHAPE, box_ious,)
from netharn.util.util_cachestamp import (CacheStamp,)
from netharn.util.util_cv2 import (draw_boxes_on_image, draw_text_on_image,
                                   putMultiLineText,)
from netharn.util.util_dataframe import (DataFrameArray, DataFrameLight,
                                         LocLight,)
from netharn.util.util_demodata import (grab_test_image,
                                        grab_test_image_fpath,)
from netharn.util.util_filesys import (get_file_info,)
from netharn.util.util_fname import (align_paths, check_aligned, dumpsafe,
                                     shortest_unique_prefixes,
                                     shortest_unique_suffixes,)
from netharn.util.util_groups import (apply_grouping, group_consecutive,
                                      group_consecutive_indices, group_indices,
                                      group_items,)
from netharn.util.util_idstr import (compact_idstr, make_idstr,
                                     make_short_idstr,)
from netharn.util.util_io import (read_arr, read_h5arr, write_arr,
                                  write_h5arr,)
from netharn.util.util_iter import (roundrobin,)
from netharn.util.util_json import (LossyJSONEncoder, NumpyEncoder, read_json,
                                    walk_json, write_json,)
from netharn.util.util_misc import (SupressPrint,)
from netharn.util.util_nms import (available_nms_impls, daq_spatial_nms,
                                   non_max_supression,)
from netharn.util.util_numpy import (atleast_nd, isect_flags,
                                     iter_reduce_ufunc,)
from netharn.util.util_random import (ensure_rng, random_combinations,
                                      random_product, seed_global, shuffle,)
from netharn.util.util_resources import (ensure_ulimit,)
from netharn.util.util_slider import (SlidingIndexDataset, SlidingSlices,
                                      SlidingWindow, Stitcher,)
from netharn.util.util_subextreme import (argsubmax, argsubmaxima,)
from netharn.util.util_tensorboard import (read_tensorboard_scalars,)
from netharn.util.util_torch import (BatchNormContext, DisableBatchNorm,
                                     IgnoreLayerContext, ModuleMixin,
                                     grad_context, number_of_parameters,
                                     one_hot_embedding, one_hot_lookup,
                                     trainable_layers,)
from netharn.util.util_zip import (split_archive, zopen,)

__all__ = ['BatchNormContext', 'Boxes', 'CV2_INTERPOLATION_TYPES',
           'CacheStamp', 'Color', 'CumMovingAve', 'DataFrameArray',
           'DataFrameLight', 'DisableBatchNorm', 'ExpMovingAve',
           'IgnoreLayerContext', 'InternalRunningStats', 'LocLight',
           'LossyJSONEncoder', 'ModuleMixin', 'MovingAve', 'NumpyEncoder',
           'PlotNums', 'RunningStats', 'SlidingIndexDataset', 'SlidingSlices',
           'SlidingWindow', 'Stitcher', 'SupressPrint',
           'TORCH_HAS_EMPTY_SHAPE', 'WindowedMovingAve', 'absdev',
           'adjust_gamma', 'adjust_subplots', 'aggensure', 'align_paths',
           'apply_grouping', 'argsubmax', 'argsubmaxima', 'atleast_3channels',
           'atleast_nd', 'autompl', 'available_nms_impls', 'axes_extent',
           'box_ious', 'check_aligned', 'colorbar', 'colorbar_image',
           'compact_idstr', 'convert_colorspace', 'copy_figure_to_clipboard',
           'daq_spatial_nms', 'dict_intersection', 'distinct_colors',
           'distinct_markers', 'draw_border', 'draw_boxes',
           'draw_boxes_on_image', 'draw_line_segments', 'draw_text_on_image',
           'dumpsafe', 'ensure_alpha_channel', 'ensure_float01', 'ensure_fnum',
           'ensure_grayscale', 'ensure_rng', 'ensure_ulimit',
           'extract_axes_extents', 'figure', 'get_file_info',
           'get_num_channels', 'grab_test_image', 'grab_test_image_fpath',
           'grad_context', 'group_consecutive', 'group_consecutive_indices',
           'group_indices', 'group_items', 'image_slices', 'imread', 'imscale',
           'imshow', 'imutil', 'imwrite', 'interpolated_colormap',
           'is_list_of_lists', 'is_list_of_scalars', 'is_listlike',
           'isect_flags', 'iter_reduce_ufunc', 'legend', 'load_image_paths',
           'make_channels_comparable', 'make_heatmask', 'make_idstr',
           'make_legend_img', 'make_short_idstr', 'mpl_3d', 'mpl_multiplot',
           'mplutil', 'multi_plot', 'next_fnum', 'non_max_supression',
           'num_channels', 'number_of_parameters', 'one_hot_embedding',
           'one_hot_lookup', 'overlay_alpha_images', 'overlay_colorized',
           'pandas_plot_matrix', 'plot_surface3d', 'profiler',
           'putMultiLineText', 'qtensure', 'random_combinations',
           'random_product', 'read_arr', 'read_h5arr', 'read_json',
           'read_tensorboard_scalars', 'render_figure_to_image',
           'reverse_colormap', 'roundrobin', 'run_length_encoding',
           'save_parts', 'savefig2', 'scores_to_cmap', 'scores_to_color',
           'seed_global', 'set_figtitle', 'set_mpl_backend',
           'shortest_unique_prefixes', 'shortest_unique_suffixes',
           'show_if_requested', 'shuffle', 'split_archive', 'stack_images',
           'stack_images_grid', 'stack_multiple_images', 'stats_dict',
           'trainable_layers', 'util_averages', 'util_boxes',
           'util_cachestamp', 'util_cv2', 'util_dataframe', 'util_demodata',
           'util_filesys', 'util_fname', 'util_groups', 'util_idstr',
           'util_io', 'util_iter', 'util_json', 'util_misc', 'util_nms',
           'util_numpy', 'util_random', 'util_resources', 'util_slider',
           'util_subextreme', 'util_tensorboard', 'util_torch', 'util_zip',
           'walk_json', 'wide_strides_1d', 'write_arr', 'write_h5arr',
           'write_json', 'zopen']
# </AUTOGEN_INIT>


class _DummyProf(object):
    pass


profiler = _DummyProf()
profiler.profile = lambda x: x
profiler.IS_PROFILING = False
