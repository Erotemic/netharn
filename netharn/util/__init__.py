"""
mkinit netharn.util
"""
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

__DYNAMIC__ = False
if __DYNAMIC__:
    import mkinit
    exec(mkinit.dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from netharn.util import imutil
    from netharn.util import mplutil
    from netharn.util import nms
    from netharn.util import profiler
    from netharn.util import torch_utils
    from netharn.util import util_averages
    from netharn.util import util_boxes
    from netharn.util import util_cv2
    from netharn.util import util_demodata
    from netharn.util import util_fname
    from netharn.util import util_idstr
    from netharn.util import util_io
    from netharn.util import util_iter
    from netharn.util import util_json
    from netharn.util import util_numpy
    from netharn.util import util_random
    from netharn.util import util_resources
    from netharn.util import util_subextreme
    from netharn.util.imutil import (CV2_INTERPOLATION_TYPES, adjust_gamma,
                                     atleast_3channels, convert_colorspace,
                                     ensure_alpha_channel, ensure_float01,
                                     ensure_grayscale, get_num_channels,
                                     grab_test_imgpath, image_slices, imread,
                                     imscale, imwrite, load_image_paths,
                                     make_channels_comparable,
                                     overlay_alpha_images, overlay_colorized,
                                     run_length_encoding, wide_strides_1d,)
    from netharn.util.mplutil import (Color, PlotNums, adjust_subplots,
                                      axes_extent, colorbar, colorbar_image,
                                      copy_figure_to_clipboard,
                                      deterministic_shuffle, dict_intersection,
                                      distinct_colors, distinct_markers,
                                      draw_border, draw_boxes, draw_line_segments,
                                      ensure_fnum, extract_axes_extents, figure,
                                      imshow, legend, make_heatmask, multi_plot,
                                      next_fnum, pandas_plot_matrix, qtensure,
                                      render_figure_to_image, reverse_colormap,
                                      save_parts, savefig2, scores_to_cmap,
                                      scores_to_color, set_figtitle,
                                      show_if_requested,)
    from netharn.util.nms import (non_max_supression,)
    from netharn.util.profiler import (IS_PROFILING, KernprofParser,
                                       dump_global_profile_report, dynamic_profile,
                                       find_parent_class, find_pattern_above_row,
                                       find_pyclass_above_row, profile,
                                       profile_onthefly,)
    from netharn.util.torch_utils import (grad_context, number_of_parameters,)
    from netharn.util.util_averages import (CumMovingAve, ExpMovingAve,
                                            InternalRunningStats, MovingAve,
                                            RunningStats, WindowedMovingAve,
                                            absdev, stats_dict,)
    from netharn.util.util_boxes import (Boxes, box_ious, box_ious_py,
                                         box_ious_torch,)
    from netharn.util.util_cv2 import (draw_boxes_on_image, draw_text_on_image,
                                       putMultiLineText,)
    from netharn.util.util_demodata import (grab_test_image,)
    from netharn.util.util_fname import (align_paths, check_aligned, dumpsafe,
                                         shortest_unique_prefixes,
                                         shortest_unique_suffixes,)
    from netharn.util.util_idstr import (compact_idstr, make_idstr,
                                         make_short_idstr,)
    from netharn.util.util_io import (read_arr, read_h5arr, write_arr,
                                      write_h5arr,)
    from netharn.util.util_iter import (roundrobin,)
    from netharn.util.util_json import (JSONEncoder, NumpyAwareJSONEncoder,
                                        NumpyEncoder, read_json, walk_json,
                                        write_json,)
    from netharn.util.util_numpy import (apply_grouping, atleast_nd, group_indices,
                                         group_items, isect_flags,
                                         iter_reduce_ufunc,)
    from netharn.util.util_random import (ensure_rng, random_combinations,
                                          random_product, shuffle,)
    from netharn.util.util_resources import (ensure_ulimit,)
    from netharn.util.util_subextreme import (argsubmax, argsubmaxima,)
