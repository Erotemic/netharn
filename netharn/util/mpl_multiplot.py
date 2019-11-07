# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import ubelt as ub
import six
import warnings
from six.moves import zip_longest
from . import mplutil as mpl_core


def multi_plot(xdata=None, ydata=None, xydata=None, **kwargs):
    r"""
    plots multiple lines, bars, etc...

    One function call that concisely describes the all of the most commonly
    used parameters needed when plotting a bar / line char. This is especially
    useful when multiple plots are needed in the same domain.

    Args:
        xdata (List[ndarray] | Dict[str, ndarray] | ndarray):
            x-coordinate data common to all y-coordinate values or xdata for
            each line/bar in ydata.  Mutually exclusive with xydata.

        ydata (List[ndarray] | Dict[str, ndarray] | ndarary):
            y-coordinate values for each line/bar to plot. Can also
            be just a single ndarray of scalar values. Mutually exclusive with
            xydata.

        xydata (Dict[str, Tuple[ndarray, ndarray]]):
            mapping from labels to a tuple of xdata and ydata for a each line.

        **kwargs:

            fnum (int):
                figure number to draw on

            pnum (Tuple[int, int, int]):
                plot number to draw on within the figure: e.g. (1, 1, 1)

            label (List[str]): if you specified ydata as a List[ndarray]
                this is the label for each line in that list. Note this is
                unnecessary if you specify input as a dictionary mapping labels
                to lines.

            label_list : same as `label`

            marker (str|List|Dict): type of matplotlib marker to use at every
                data point. Can be specified for all lines jointly or for each
                line independently.

            marker_list : same as `marker`

            transpose (bool, default=False): swaps x and y data.

            kind (str, default='plot'):
                The kind of plot. Can either be 'plot' or 'bar'.
                We parse these other kwargs if:
                    if kind='plot':
                        spread
                    if kind='bar':
                        stacked, width

            Misc:
                use_legend, legend_loc
            Labels:
                xlabel, ylabel, title, figtitle
                ticksize, titlesize, legendsize, labelsize
            Grid:
                gridlinewidth, gridlinestyle
            Ticks:
                num_xticks, num_yticks, tickwidth, ticklength, ticksize
                xticklabels, yticklabels, <-overwrites previous
            Data:
                xmin, xmax, ymin, ymax, spread_list
                # can append _list to any of these
                # these can be dictionaries if ydata was also a dict

                xscale in [linear, log, logit, symlog]
                yscale in [linear, log, logit, symlog]

                plot_kw_keys = ['label', 'color', 'marker', 'markersize',
                    'markeredgewidth', 'linewidth', 'linestyle']
                any plot_kw key can be a scalar (corresponding to all ydatas),
                a list if ydata was specified as a list, or a dict if ydata was
                specified as a dict.

                kind = ['bar', 'plot', ...]

    Returns:
        matplotlib.axes.Axes: ax : the axes that was drawn on

    References:
        matplotlib.org/examples/api/barchart_demo.html

    Example:
        >>> import netharn as nh
        >>> nh.util.autompl()
        >>> # The new way to use multi_plot is to pass ydata as a dict of lists
        >>> ydata = {
        >>>     'spamΣ': [1, 1, 2, 3, 5, 8, 13],
        >>>     'eggs': [3, 3, 3, 3, 3, np.nan, np.nan],
        >>>     'jamµ': [5, 3, np.nan, 1, 2, np.nan, np.nan],
        >>>     'pram': [4, 2, np.nan, 0, 0, np.nan, 1],
        >>> }
        >>> ax = nh.util.multi_plot(ydata=ydata, title='ΣΣΣµµµ',
        >>>                      xlabel='\nfdsΣΣΣµµµ', linestyle='--')
        >>> nh.util.show_if_requested()

    Example:
        >>> # Old way to use multi_plot is a list of lists
        >>> import netharn as nh
        >>> nh.util.autompl()
        >>> xdata = [1, 2, 3, 4, 5]
        >>> ydata_list = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, np.nan, 2, 1], [4, 3, np.nan, 1, 0]]
        >>> kwargs = {'label': ['spamΣ', 'eggs', 'jamµ', 'pram'],  'linestyle': '-'}
        >>> #ax = multi_plot(xdata, ydata_list, title='$\phi_1(\\vec{x})$', xlabel='\nfds', **kwargs)
        >>> ax = multi_plot(xdata, ydata_list, title='ΣΣΣµµµ', xlabel='\nfdsΣΣΣµµµ', **kwargs)
        >>> nh.util.show_if_requested()

    Example:
        >>> # Simple way to use multi_plot is to pass xdata and ydata exactly
        >>> # like you would use plt.plot
        >>> import netharn as nh
        >>> nh.util.autompl()
        >>> ax = multi_plot([1, 2, 3], [4, 5, 6], fnum=4, label='foo')
        >>> nh.util.show_if_requested()

    Example:
        >>> import netharn as nh
        >>> nh.util.autompl()
        >>> xydata = {'a': ([0, 1, 2], [0, 1, 2]), 'b': ([0, 2, 4], [2, 1, 0])}
        >>> ax = nh.util.multi_plot(xydata=xydata, fnum=4)
        >>> nh.util.show_if_requested()

    Ignore:
        >>> import netharn as nh
        >>> nh.util.autompl()
        >>> ydata = {
        >>>     str(i): np.random.rand(100) + i for i in range(30)
        >>> }
        >>> ax = nh.util.multi_plot(ydata=ydata, fnum=1, doclf=True)
        >>> nh.util.show_if_requested()
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    # Initial integration with mpl rcParams standards
    mplrc = mpl.rcParams
    # mplrc.update({
    #     # 'legend.fontsize': custom_figure.LEGEND_SIZE,
    #     # 'legend.framealpha':
    #     # 'axes.titlesize': custom_figure.TITLE_SIZE,
    #     # 'axes.labelsize': custom_figure.LABEL_SIZE,
    #     # 'legend.facecolor': 'w',
    #     # 'font.family': 'sans-serif',
    #     # 'xtick.labelsize': custom_figure.TICK_SIZE,
    #     # 'ytick.labelsize': custom_figure.TICK_SIZE,
    # })
    if 'rcParams' in kwargs:
        mplrc = mplrc.copy()
        mplrc.update(kwargs['rcParams'])

    if xydata is not None:
        if xdata is not None or ydata is not None:
            raise ValueError('Cannot specify xydata with xdata or ydata')
        if isinstance(xydata, dict):
            xdata = ub.odict((k, np.array(xy[0])) for k, xy in xydata.items())
            ydata = ub.odict((k, np.array(xy[1])) for k, xy in xydata.items())
        else:
            raise ValueError('Only supports xydata as Dict at the moment')

    if bool('label' in kwargs) and bool('label_list' in kwargs):
        raise ValueError('Specify either label or label_list')

    if isinstance(ydata, dict):
        # Case where ydata is a dictionary
        if isinstance(xdata, six.string_types):
            # Special-er case where xdata is specified in ydata
            xkey = xdata
            ykeys = set(ydata.keys()) - {xkey}
            xdata = ydata[xkey]
        else:
            ykeys = list(ydata.keys())
        # Normalize input into ydata_list
        ydata_list = list(ub.take(ydata, ykeys))

        default_label_list = kwargs.pop('label', ykeys)
        kwargs['label_list'] = kwargs.get('label_list', default_label_list)
    else:
        # ydata should be a List[ndarray] or an ndarray
        ydata_list = ydata
        ykeys = None

    # allow ydata_list to be passed without a container
    if is_list_of_scalars(ydata_list):
        ydata_list = [np.array(ydata_list)]

    if xdata is None:
        xdata = list(range(len(ydata_list[0])))

    num_lines = len(ydata_list)

    # Transform xdata into xdata_list
    if isinstance(xdata, dict):
        xdata_list = [np.array(xdata[k], copy=True) for k in ykeys]
    elif is_list_of_lists(xdata):
        xdata_list = [np.array(xd, copy=True) for xd in xdata]
    else:
        xdata_list = [np.array(xdata, copy=True)] * num_lines

    fnum = mpl_core.ensure_fnum(kwargs.get('fnum', None))
    pnum = kwargs.get('pnum', None)
    kind = kwargs.get('kind', 'plot')
    transpose = kwargs.get('transpose', False)

    def parsekw_list(key, kwargs, num_lines=num_lines, ykeys=ykeys):
        """ copies relevant plot commands into plot_list_kw """
        if key in kwargs:
            val_list = kwargs[key]
        elif key + '_list' in kwargs:
            # warnings.warn('*_list is depricated, just use kwarg {}'.format(key))
            val_list = kwargs[key + '_list']
        elif key + 's' in kwargs:
            # hack, multiple ways to do something
            warnings.warn('*s depricated, just use kwarg {}'.format(key))
            val_list = kwargs[key + 's']
        else:
            val_list = None

        if val_list is not None:
            if isinstance(val_list, dict):
                if ykeys is None:
                    raise ValueError(
                        'Kwarg {!r} was a dict, but ydata was not'.format(key))
                else:
                    val_list = [val_list[key] for key in ykeys]

            if not isinstance(val_list, list):
                val_list = [val_list] * num_lines

        return val_list

    if kind == 'plot':
        if 'marker' not in kwargs:
            # kwargs['marker'] = mplrc['lines.marker']
            kwargs['marker'] = 'distinct'
            # kwargs['marker'] = 'cycle'

        if isinstance(kwargs['marker'], six.string_types):
            if kwargs['marker'] == 'distinct':
                kwargs['marker'] = mpl_core.distinct_markers(num_lines)
            elif kwargs['marker'] == 'cycle':
                # Note the length of marker and linestyle cycles should be
                # relatively prime.
                # https://matplotlib.org/api/markers_api.html
                marker_cycle = ['.', '*', 'x']
                kwargs['marker'] = [marker_cycle[i % len(marker_cycle)] for i in range(num_lines)]
            # else:
            #     raise KeyError(kwargs['marker'])

        if 'linestyle' not in kwargs:
            # kwargs['linestyle'] = 'distinct'
            kwargs['linestyle'] = mplrc['lines.linestyle']
            # kwargs['linestyle'] = 'cycle'

        if isinstance(kwargs['linestyle'], six.string_types):
            if kwargs['linestyle'] == 'cycle':
                # https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
                linestyle_cycle = ['solid', 'dashed', 'dashdot', 'dotted']
                kwargs['linestyle'] = [linestyle_cycle[i % len(linestyle_cycle)] for i in range(num_lines)]

    if 'color' not in kwargs:
        # kwargs['color'] = 'jet'
        # kwargs['color'] = 'gist_rainbow'
        kwargs['color'] = 'distinct'

    if isinstance(kwargs['color'], six.string_types):
        if kwargs['color'] == 'distinct':
            kwargs['color'] = mpl_core.distinct_colors(num_lines, randomize=0)
        else:
            cm = plt.get_cmap(kwargs['color'])
            kwargs['color'] = [cm(i / num_lines) for i in range(num_lines)]

    # Parse out arguments to ax.plot
    plot_kw_keys = ['label', 'color', 'marker', 'markersize',
                    'markeredgewidth', 'linewidth', 'linestyle', 'alpha']
    # hackish / extra args that dont go to plot, but help
    extra_plot_kw_keys = ['spread_alpha', 'autolabel', 'edgecolor', 'fill']
    plot_kw_keys += extra_plot_kw_keys
    plot_ks_vals = [parsekw_list(key, kwargs) for key in plot_kw_keys]
    plot_list_kw = dict([
        (key, vals)
        for key, vals in zip(plot_kw_keys, plot_ks_vals) if vals is not None
    ])

    if kind == 'plot':
        if 'spread_alpha' not in plot_list_kw:
            plot_list_kw['spread_alpha'] = [.2] * num_lines

    if kind == 'bar':
        # Remove non-bar kwargs
        for key in ['markeredgewidth', 'linewidth', 'marker', 'markersize', 'linestyle']:
            plot_list_kw.pop(key, None)

        stacked = kwargs.get('stacked', False)
        width_key = 'height' if transpose else 'width'
        if 'width_list' in kwargs:
            plot_list_kw[width_key] = kwargs['width_list']
        else:
            width = kwargs.get('width', .9)
            # if width is None:
            #     # HACK: need variable width
            #     # width = np.mean(np.diff(xdata_list[0]))
            #     width = .9
            if not stacked:
                width /= num_lines
            #plot_list_kw['orientation'] = ['horizontal'] * num_lines
            plot_list_kw[width_key] = [width] * num_lines

    spread_list = kwargs.get('spread_list', None)
    if spread_list is None:
        pass

    # nest into a list of dicts for each line in the multiplot
    valid_keys = list(set(plot_list_kw.keys()) - set(extra_plot_kw_keys))
    valid_vals = list(ub.dict_take(plot_list_kw, valid_keys))
    plot_kw_list = [dict(zip(valid_keys, vals)) for vals in zip(*valid_vals)]

    extra_kw_keys = [key for key in extra_plot_kw_keys if key in plot_list_kw]
    extra_kw_vals = list(ub.dict_take(plot_list_kw, extra_kw_keys))
    extra_kw_list = [dict(zip(extra_kw_keys, vals)) for vals in zip(*extra_kw_vals)]

    # Get passed in axes or setup a new figure
    ax = kwargs.get('ax', None)
    if ax is None:
        doclf = kwargs.get('doclf', False)
        fig = mpl_core.figure(fnum=fnum, pnum=pnum, docla=False, doclf=doclf)
        ax = fig.gca()
    else:
        plt.sca(ax)
        fig = ax.figure

    # +---------------
    # Draw plot lines
    ydata_list = np.array(ydata_list)

    if transpose:
        if kind == 'bar':
            plot_func = ax.barh
        elif kind == 'plot':
            def plot_func(_x, _y, **kw):
                return ax.plot(_y, _x, **kw)
    else:
        plot_func = getattr(ax, kind)  # usually ax.plot

    if len(ydata_list) > 0:
        # raise ValueError('no ydata')
        _iter = enumerate(zip_longest(xdata_list, ydata_list, plot_kw_list, extra_kw_list))
        for count, (_xdata, _ydata, plot_kw, extra_kw) in _iter:
            ymask = np.isfinite(_ydata)
            ydata_ = _ydata.compress(ymask)
            xdata_ = _xdata.compress(ymask)
            if kind == 'bar':
                if stacked:
                    # Plot bars on top of each other
                    xdata_ = xdata_
                else:
                    # Plot bars side by side
                    baseoffset = (width * num_lines) / 2
                    lineoffset = (width * count)
                    offset = baseoffset - lineoffset  # Fixeme for more histogram bars
                    xdata_ = xdata_ - offset
                # width_key = 'height' if transpose else 'width'
                # plot_kw[width_key] = np.diff(xdata)
            objs = plot_func(xdata_, ydata_, **plot_kw)

            if kind == 'bar':
                if extra_kw is not None and 'edgecolor' in extra_kw:
                    for rect in objs:
                        rect.set_edgecolor(extra_kw['edgecolor'])
                if extra_kw is not None and extra_kw.get('autolabel', False):
                    # FIXME: probably a more cannonical way to include bar
                    # autolabeling with tranpose support, but this is a hack that
                    # works for now
                    for rect in objs:
                        if transpose:
                            numlbl = width = rect.get_width()
                            xpos = width + ((_xdata.max() - _xdata.min()) * .005)
                            ypos = rect.get_y() + rect.get_height() / 2.
                            ha, va = 'left', 'center'
                        else:
                            numlbl = height = rect.get_height()
                            xpos = rect.get_x() + rect.get_width() / 2.
                            ypos = 1.05 * height
                            ha, va = 'center', 'bottom'
                        barlbl = '%.3f' % (numlbl,)
                        ax.text(xpos, ypos, barlbl, ha=ha, va=va)

            # print('extra_kw = %r' % (extra_kw,))
            if kind == 'plot' and extra_kw.get('fill', False):
                ax.fill_between(_xdata, ydata_, alpha=plot_kw.get('alpha', 1.0),
                                color=plot_kw.get('color', None))  # , zorder=0)

            if spread_list is not None:
                # Plots a spread around plot lines usually indicating standard
                # deviation
                _xdata = np.array(_xdata)
                spread = spread_list[count]
                ydata_ave = np.array(ydata_)
                y_data_dev = np.array(spread)
                y_data_max = ydata_ave + y_data_dev
                y_data_min = ydata_ave - y_data_dev
                ax = plt.gca()
                spread_alpha = extra_kw['spread_alpha']
                ax.fill_between(_xdata, y_data_min, y_data_max, alpha=spread_alpha,
                                color=plot_kw.get('color', None))  # , zorder=0)
        ydata = _ydata  # HACK
        xdata = _xdata  # HACK
    # L________________

    #max_y = max(np.max(y_data), max_y)
    #min_y = np.min(y_data) if min_y is None else min(np.min(y_data), min_y)

    if transpose:
        #xdata_list = ydata_list
        ydata = xdata
        # Hack / Fix any transpose issues
        def transpose_key(key):
            if key.startswith('x'):
                return 'y' + key[1:]
            elif key.startswith('y'):
                return 'x' + key[1:]
            elif key.startswith('num_x'):
                # hackier, fixme to use regex or something
                return 'num_y' + key[5:]
            elif key.startswith('num_y'):
                # hackier, fixme to use regex or something
                return 'num_x' + key[5:]
            else:
                return key
        kwargs = {transpose_key(key): val for key, val in kwargs.items()}

    # Setup axes labeling
    title      = kwargs.get('title', None)
    xlabel     = kwargs.get('xlabel', '')
    ylabel     = kwargs.get('ylabel', '')
    def none_or_unicode(text):
        return None if text is None else ub.ensure_unicode(text)

    xlabel = none_or_unicode(xlabel)
    ylabel = none_or_unicode(ylabel)
    title = none_or_unicode(title)

    titlesize  = kwargs.get('titlesize',  mplrc['axes.titlesize'])
    labelsize  = kwargs.get('labelsize',  mplrc['axes.labelsize'])
    legendsize = kwargs.get('legendsize', mplrc['legend.fontsize'])
    xticksize = kwargs.get('ticksize', mplrc['xtick.labelsize'])
    yticksize = kwargs.get('ticksize', mplrc['ytick.labelsize'])
    family = kwargs.get('fontfamily', mplrc['font.family'])

    tickformat = kwargs.get('tickformat', None)
    ytickformat = kwargs.get('ytickformat', tickformat)
    xtickformat = kwargs.get('xtickformat', tickformat)

    # 'DejaVu Sans','Verdana', 'Arial'
    weight = kwargs.get('fontweight', None)
    if weight is None:
        weight = 'normal'

    labelkw = {
        'fontproperties': mpl.font_manager.FontProperties(
            weight=weight,
            family=family, size=labelsize)
    }
    ax.set_xlabel(xlabel, **labelkw)
    ax.set_ylabel(ylabel, **labelkw)

    tick_fontprop = mpl.font_manager.FontProperties(family=family,
                                                    weight=weight)

    if tick_fontprop is not None:
        for ticklabel in ax.get_xticklabels():
            ticklabel.set_fontproperties(tick_fontprop)
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_fontproperties(tick_fontprop)
    if xticksize is not None:
        for ticklabel in ax.get_xticklabels():
            ticklabel.set_fontsize(xticksize)
    if yticksize is not None:
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_fontsize(yticksize)

    if xtickformat is not None:
        # mpl.ticker.StrMethodFormatter  # new style
        # mpl.ticker.FormatStrFormatter  # old style
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(xtickformat))
    if ytickformat is not None:
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(ytickformat))

    xtick_kw = ytick_kw = {
        'width': kwargs.get('tickwidth', None),
        'length': kwargs.get('ticklength', None),
    }
    xtick_kw = {k: v for k, v in xtick_kw.items() if v is not None}
    ytick_kw = {k: v for k, v in ytick_kw.items() if v is not None}
    ax.xaxis.set_tick_params(**xtick_kw)
    ax.yaxis.set_tick_params(**ytick_kw)

    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

    # Setup axes limits
    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
        if xlim is not None:
            if 'xmin' not in kwargs and 'xmax' not in kwargs:
                kwargs['xmin'] = xlim[0]
                kwargs['xmax'] = xlim[1]
            else:
                raise ValueError('use xmax, xmin instead of xlim')
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
        if ylim is not None:
            if 'ymin' not in kwargs and 'ymax' not in kwargs:
                kwargs['ymin'] = ylim[0]
                kwargs['ymax'] = ylim[1]
            else:
                raise ValueError('use ymax, ymin instead of ylim')

    xmin = kwargs.get('xmin', ax.get_xlim()[0])
    xmax = kwargs.get('xmax', ax.get_xlim()[1])
    ymin = kwargs.get('ymin', ax.get_ylim()[0])
    ymax = kwargs.get('ymax', ax.get_ylim()[1])

    text_type = six.text_type

    if text_type(xmax) == 'data':
        xmax = max([xd.max() for xd in xdata_list])
    if text_type(xmin) == 'data':
        xmin = min([xd.min() for xd in xdata_list])

    # Setup axes ticks
    num_xticks = kwargs.get('num_xticks', None)
    num_yticks = kwargs.get('num_yticks', None)

    if num_xticks is not None:
        # TODO check if xdata is integral
        if xdata.dtype.kind == 'i':
            xticks = np.linspace(np.ceil(xmin), np.floor(xmax),
                                 num_xticks).astype(np.int32)
        else:
            xticks = np.linspace((xmin), (xmax), num_xticks)
        ax.set_xticks(xticks)
    if num_yticks is not None:
        if ydata.dtype.kind == 'i':
            yticks = np.linspace(np.ceil(ymin), np.floor(ymax),
                                 num_yticks).astype(np.int32)
        else:
            yticks = np.linspace((ymin), (ymax), num_yticks)
        ax.set_yticks(yticks)

    force_xticks = kwargs.get('force_xticks', None)
    if force_xticks is not None:
        xticks = np.array(sorted(ax.get_xticks().tolist() + force_xticks))
        ax.set_xticks(xticks)

    yticklabels = kwargs.get('yticklabels', None)
    if yticklabels is not None:
        # Hack ONLY WORKS WHEN TRANSPOSE = True
        # Overrides num_yticks
        ax.set_yticks(ydata)
        ax.set_yticklabels(yticklabels)

    xticklabels = kwargs.get('xticklabels', None)
    if xticklabels is not None:
        # Overrides num_xticks
        ax.set_xticks(xdata)
        ax.set_xticklabels(xticklabels)

    xticks = kwargs.get('xticks', None)
    if xticks is not None:
        print('xticks = {!r}'.format(xticks))
        ax.set_xticks(xticks)

    yticks = kwargs.get('yticks', None)
    if yticks is not None:
        ax.set_yticks(yticks)

    xtick_rotation = kwargs.get('xtick_rotation', None)
    if xtick_rotation is not None:
        [lbl.set_rotation(xtick_rotation)
         for lbl in ax.get_xticklabels()]
    ytick_rotation = kwargs.get('ytick_rotation', None)
    if ytick_rotation is not None:
        [lbl.set_rotation(ytick_rotation)
         for lbl in ax.get_yticklabels()]

    # Axis padding
    xpad = kwargs.get('xpad', None)
    ypad = kwargs.get('ypad', None)
    xpad_factor = kwargs.get('xpad_factor', None)
    ypad_factor = kwargs.get('ypad_factor', None)
    if xpad is None and xpad_factor is not None:
        xpad = (xmax - xmin) * xpad_factor
    if ypad is None and ypad_factor is not None:
        ypad = (ymax - ymin) * ypad_factor
    xpad = 0 if xpad is None else xpad
    ypad = 0 if ypad is None else ypad
    ypad_high = kwargs.get('ypad_high', ypad)
    ypad_low  = kwargs.get('ypad_low', ypad)
    xpad_high = kwargs.get('xpad_high', xpad)
    xpad_low  = kwargs.get('xpad_low', xpad)
    xmin, xmax = (xmin - xpad_low), (xmax + xpad_high)
    ymin, ymax = (ymin - ypad_low), (ymax + ypad_high)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    xscale = kwargs.get('xscale', None)
    yscale = kwargs.get('yscale', None)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xscale is not None:
        ax.set_xscale(xscale)

    gridlinestyle = kwargs.get('gridlinestyle', None)
    gridlinewidth = kwargs.get('gridlinewidth', None)
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    if gridlinestyle:
        for line in gridlines:
            line.set_linestyle(gridlinestyle)
    if gridlinewidth:
        for line in gridlines:
            line.set_linewidth(gridlinewidth)

    # Setup title
    if title is not None:
        titlekw = {
            'fontproperties': mpl.font_manager.FontProperties(
                family=family,
                weight=weight,
                size=titlesize)
        }
        ax.set_title(title, **titlekw)

    use_legend   = kwargs.get('use_legend', 'label' in valid_keys)
    legend_loc   = kwargs.get('legend_loc', mplrc['legend.loc'])
    legend_alpha = kwargs.get('legend_alpha', mplrc['legend.framealpha'])
    if use_legend:
        legendkw = {
            'alpha': legend_alpha,
            'fontproperties': mpl.font_manager.FontProperties(
                family=family,
                weight=weight,
                size=legendsize)
        }
        mpl_core.legend(loc=legend_loc, ax=ax, **legendkw)

    figtitle = kwargs.get('figtitle', None)

    if figtitle is not None:
        # mplrc['figure.titlesize'] TODO?
        mpl_core.set_figtitle(figtitle, fontfamily=family, fontweight=weight,
                              size=kwargs.get('figtitlesize'))

    # TODO: return better info
    return ax


def is_listlike(data):
    flag = isinstance(data, (list, np.ndarray, tuple, pd.Series))
    flag &= hasattr(data, '__getitem__') and hasattr(data, '__len__')
    return flag


def is_list_of_scalars(data):
    if is_listlike(data):
        if len(data) > 0 and not is_listlike(data[0]):
            return True
    return False


def is_list_of_lists(data):
    if is_listlike(data):
        if len(data) > 0 and is_listlike(data[0]):
            return True
    return False
