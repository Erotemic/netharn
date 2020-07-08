"""
DEPRECATED

USE kwcoco.metrics instead!
"""
import numpy as np
import ubelt as ub
import warnings


def draw_roc(roc_info, prefix='', fnum=1, **kw):
    """
    NOTE: There needs to be enough negative examples for using ROC
    to make any sense!

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> from netharn.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=100, nboxes=(0, 30), n_fp=(0, 1), nclasses=3,
        >>>     box_noise=0.00, cls_noise=.0, score_noise=1.0)
        >>> dmet.true_detections(0).data
        >>> cfsn_vecs = dmet.confusion_vectors(compat='mutex', prioritize='iou', bias=0)
        >>> print(cfsn_vecs.data._pandas().sort_values('score'))
        >>> classes = cfsn_vecs.classes
        >>> roc_info = ub.peek(cfsn_vecs.binarize_ovr().roc()['perclass'].values())
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_roc(roc_info)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    fp_count = roc_info['fp_count']
    fp_rate = roc_info['fpr']
    tp_rate = roc_info['tpr']
    auc = roc_info['auc']
    realpos_total = roc_info['realpos_total']

    title = prefix + 'AUC*: {:.4f}'.format(auc)
    xscale = 'linear'
    falsepos_total = fp_count[-1]

    ax = kwplot.multi_plot(
        list(fp_rate), list(tp_rate), marker='',
        # xlabel='FA count (false positive count)',
        xlabel='fpr (count={})'.format(falsepos_total),
        ylabel='tpr (count={})'.format(int(realpos_total)),
        title=title, xscale=xscale,
        ylim=(0, 1), ypad=1e-2,
        xlim=(0, 1), xpad=1e-2,
        fnum=fnum, **kw)

    return ax


def draw_perclass_roc(cx_to_rocinfo, classes=None, prefix='', fnum=1,
                      fp_axis='count', **kw):
    """

    fp_axis can be count or rate

                cx_to_rocinfo = roc_perclass
    """
    import kwplot
    # Sort by descending AP
    cxs = list(cx_to_rocinfo.keys())
    priority = np.array([item['auc'] for item in cx_to_rocinfo.values()])
    priority[np.isnan(priority)] = -np.inf
    cxs = list(ub.take(cxs, np.argsort(priority)))[::-1]
    xydata = ub.odict()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        mAUC = np.nanmean([item['auc'] for item in cx_to_rocinfo.values()])

    if fp_axis == 'count':
        xlabel = 'FP-count'
    elif fp_axis == 'rate':
        xlabel = 'FPR'
    else:
        raise KeyError(fp_axis)

    for cx in cxs:
        peritem = cx_to_rocinfo[cx]

        catname = classes[cx] if isinstance(cx, int) else cx

        auc = peritem['auc']
        tpr = peritem['tpr']

        nsupport = int(peritem['nsupport'])
        if 'realpos_total' in peritem:
            z = peritem['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = 'auc={:0.2f}: {} ({:d}/{:d})'.format(auc, catname, int(peritem['realpos_total']), round(nsupport, 2))
            else:
                label = 'auc={:0.2f}: {} ({:.2f}/{:d})'.format(auc, catname, round(peritem['realpos_total'], 2), round(nsupport, 2))
        else:
            label = 'auc={:0.2f}: {} ({:d})'.format(auc, catname, round(nsupport, 2))

        if fp_axis == 'count':
            fp_count = peritem['fp_count']
            xydata[label] = (fp_count, tpr)
        elif fp_axis == 'rate':
            fpr = peritem['fpr']
            xydata[label] = (fpr, tpr)

    ax = kwplot.multi_plot(
        xydata=xydata, fnum=fnum,
        ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel=xlabel, ylabel='TPR',
        title=prefix + 'perclass mAUC={:.4f}'.format(mAUC),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle', **kw
    )
    return ax


def draw_perclass_prcurve(cx_to_peritem, classes=None, prefix='', fnum=1, **kw):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> from netharn.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> classes = cfsn_vecs.classes
        >>> cx_to_peritem = cfsn_vecs.binarize_ovr().precision_recall()['perclass']
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_perclass_prcurve(cx_to_peritem, classes)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    # Sort by descending AP
    cxs = list(cx_to_peritem.keys())
    priority = np.array([item['ap'] for item in cx_to_peritem.values()])
    priority[np.isnan(priority)] = -np.inf
    cxs = list(ub.take(cxs, np.argsort(priority)))[::-1]
    aps = []
    xydata = ub.odict()
    for cx in cxs:
        peritem = cx_to_peritem[cx]
        catname = classes[cx] if isinstance(cx, int) else cx
        ap = peritem['ap']
        if 'pr' in peritem:
            pr = peritem['pr']
        elif 'ppv' in peritem:
            pr = (peritem['ppv'], peritem['tpr'])
        elif 'prec' in peritem:
            pr = (peritem['prec'], peritem['rec'])
        else:
            raise KeyError('pr, prec, or ppv not in peritem')

        if np.isfinite(ap):
            aps.append(ap)
            (precision, recall) = pr
        else:
            aps.append(np.nan)
            precision, recall = [0], [0]

        if precision is None and recall is None:
            # I thought AP=nan in this case, but I missed something
            precision, recall = [0], [0]

        nsupport = int(peritem['nsupport'])
        if 'realpos_total' in peritem:
            z = peritem['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = 'ap={:0.2f}: {} ({:d}/{:d})'.format(ap, catname, int(peritem['realpos_total']), round(nsupport, 2))
            else:
                label = 'ap={:0.2f}: {} ({:.2f}/{:d})'.format(ap, catname, round(peritem['realpos_total'], 2), round(nsupport, 2))
        else:
            label = 'ap={:0.2f}: {} ({:d})'.format(ap, catname, round(nsupport, 2))
        xydata[label] = (recall, precision)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        mAP = np.nanmean(aps)

    ax = kwplot.multi_plot(
        xydata=xydata, fnum=fnum,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='recall', ylabel='precision',
        title=prefix + 'perclass mAP={:.4f}'.format(mAP),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle', **kw
    )
    return ax


def draw_perclass_thresholds(cx_to_peritem, key='mcc', classes=None, prefix='', fnum=1, **kw):
    """
    Notes:
        Each category is inspected independently of one another, there is no
        notion of confusion.

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> from netharn.metrics.drawing import *  # NOQA
        >>> from netharn.metrics import ConfusionVectors
        >>> cfsn_vecs = ConfusionVectors.demo()
        >>> classes = cfsn_vecs.classes
        >>> ovr_cfsn = cfsn_vecs.binarize_ovr(keyby='name')
        >>> cx_to_peritem = ovr_cfsn.threshold_curves()['perclass']
        >>> import kwplot
        >>> kwplot.autompl()
        >>> key = 'mcc'
        >>> draw_perclass_thresholds(cx_to_peritem, key, classes)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    # Sort by descending "best value"
    cxs = list(cx_to_peritem.keys())

    try:
        priority = np.array([item['_max_' + key][0] for item in cx_to_peritem.values()])
        priority[np.isnan(priority)] = -np.inf
        cxs = list(ub.take(cxs, np.argsort(priority)))[::-1]
    except KeyError:
        pass

    xydata = ub.odict()
    for cx in cxs:
        peritem = cx_to_peritem[cx]
        catname = classes[cx] if isinstance(cx, int) else cx

        thresholds = peritem['thresholds']
        measure = peritem[key]
        try:
            best_label = peritem['max_{}'.format(key)]
        except KeyError:
            max_idx = measure.argmax()
            best_thresh = thresholds[max_idx]
            best_measure = measure[max_idx]
            best_label = '{}={:0.2f}@{:0.2f}'.format(key, best_measure, best_thresh)

        nsupport = int(peritem['nsupport'])
        if 'realpos_total' in peritem:
            z = peritem['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = '{}: {} ({:d}/{:d})'.format(best_label, catname, int(peritem['realpos_total']), round(nsupport, 2))
            else:
                label = '{}: {} ({:.2f}/{:d})'.format(best_label, catname, round(peritem['realpos_total'], 2), round(nsupport, 2))
        else:
            label = '{}: {} ({:d})'.format(best_label, catname, round(nsupport, 2))
        xydata[label] = (thresholds, measure)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)

    ax = kwplot.multi_plot(
        xydata=xydata, fnum=fnum,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='threshold', ylabel=key,
        title=prefix + 'perclass {}'.format(key),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle', **kw
    )
    return ax


def draw_prcurve(peritem, prefix='', fnum=1, **kw):
    """
    TODO: rename to draw prcurve. Just draws a single pr curve.

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> from netharn.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
        >>> cfsn_vecs = dmet.confusion_vectors()

        >>> classes = cfsn_vecs.classes
        >>> peritem = cfsn_vecs.binarize_peritem().precision_recall()
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_prcurve(peritem)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    aps = []
    ap = peritem['ap']
    if 'pr' in peritem:
        pr = peritem['pr']
    elif 'ppv' in peritem:
        pr = (peritem['ppv'], peritem['tpr'])
    elif 'prec' in peritem:
        pr = (peritem['prec'], peritem['rec'])
    else:
        raise KeyError('pr, prec, or ppv not in peritem')
    if np.isfinite(ap):
        aps.append(ap)
        (precision, recall) = pr
    else:
        precision, recall = [0], [0]
    if precision is None and recall is None:
        # I thought AP=nan in this case, but I missed something
        precision, recall = [0], [0]

    nsupport = int(peritem['nsupport'])
    if 'realpos_total' in peritem:
        z = peritem['realpos_total']
        if abs(z - int(z)) < 1e-8:
            label = 'ap={:0.2f}: ({:d}/{:d})'.format(ap, int(peritem['realpos_total']), round(nsupport, 2))
        else:
            label = 'ap={:0.2f}: ({:.2f}/{:d})'.format(ap, round(peritem['realpos_total'], 2), round(nsupport, 2))
    else:
        label = 'ap={:0.2f}: ({:d})'.format(ap, nsupport)

    ax = kwplot.multi_plot(
        xdata=recall, ydata=precision, fnum=fnum, label=label,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='recall', ylabel='precision',
        title=prefix + 'peritem AP={:.4f}'.format(ap),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle', **kw
    )
    return ax


def draw_threshold_curves(info, keys=None, prefix='', fnum=1, **kw):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/netharn'))
        >>> from netharn.metrics.drawing import *  # NOQA
        >>> from netharn.metrics import DetectionMetrics
        >>> dmet = DetectionMetrics.demo(
        >>>     nimgs=10, nboxes=(0, 10), n_fp=(0, 1), nclasses=3)
        >>> cfsn_vecs = dmet.confusion_vectors()
        >>> info = cfsn_vecs.binarize_peritem().threshold_curves()
        >>> keys = None
        >>> import kwplot
        >>> kwplot.autompl()
        >>> draw_threshold_curves(info, keys)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    import kwimage
    thresh = info['thresholds']

    if keys is None:
        keys = {'g1', 'f1', 'acc', 'mcc'}

    idx_to_colors = kwimage.Color.distinct(len(keys), space='rgba')
    idx_to_best_pt = {}

    xydata = {}
    colors = {}
    for idx, key in enumerate(keys):
        color = idx_to_colors[idx]
        measure = info[key]
        max_idx = measure.argmax()
        best_thresh = thresh[max_idx]
        best_measure = measure[max_idx]
        best_label = '{}={:0.2f}@{:0.2f}'.format(key, best_measure, best_thresh)

        nsupport = int(info['nsupport'])
        if 'realpos_total' in info:
            z = info['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = '{}: ({:d}/{:d})'.format(best_label, int(info['realpos_total']), round(nsupport, 2))
            else:
                label = '{}: ({:.2f}/{:d})'.format(best_label, round(info['realpos_total'], 2), round(nsupport, 2))
        else:
            label = '{}: ({:d})'.format(best_label, nsupport)
        xydata[label] = (thresh, measure)
        colors[label] = color
        idx_to_best_pt[idx] = (best_thresh, best_measure)

    ax = kwplot.multi_plot(
        xydata=xydata, fnum=fnum,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='threshold', ylabel=key,
        title=prefix + 'threshold curves',
        legend_loc='lower right',
        color=colors,
        linestyle='cycle', marker='cycle', **kw
    )
    for idx, best_pt in idx_to_best_pt.items():
        best_thresh, best_measure = best_pt
        color = idx_to_colors[idx]
        ax.plot(best_thresh, best_measure, '*', color=color)
    return ax

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/metrics/drawing.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
