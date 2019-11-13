import numpy as np
import ubelt as ub
import warnings


def draw_roc(roc_info, prefix=''):
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
        >>> roc_info = cfsn_vecs.binarize_ovr().roc()['perclass'][1]
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
    figtitle = None
    falsepos_total = fp_count[-1]

    ax = kwplot.multi_plot(
        list(fp_rate), list(tp_rate), marker='',
        # xlabel='FA count (false positive count)',
        xlabel='fpr (count={})'.format(falsepos_total),
        ylabel='tpr (count={})'.format(int(realpos_total)),
        title=title, xscale=xscale,
        ylim=(0, 1), ypad=1e-2,
        xlim=(0, 1), xpad=1e-2,
        figtitle=figtitle, fnum=1, doclf=True)

    return ax


def draw_perclass_roc(cx_to_rocinfo, classes, prefix=''):
    """
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

    for cx in cxs:
        peritem = cx_to_rocinfo[cx]
        catname = classes[cx]
        auc = peritem['auc']

        tpr = peritem['tpr']
        fp_count = peritem['fp_count']

        nsupport = int(peritem['nsupport'])
        if 'realpos_total' in peritem:
            z = peritem['realpos_total']
            if abs(z - int(z)) < 1e-8:
                label = 'auc={:0.2f}: {} ({:d}/{:d})'.format(auc, catname, int(peritem['realpos_total']), nsupport)
            else:
                label = 'auc={:0.2f}: {} ({}/{:d})'.format(auc, catname, peritem['realpos_total'], nsupport)
        else:
            label = 'auc={:0.2f}: {} ({:d})'.format(auc, catname, nsupport)
        xydata[label] = (fp_count, tpr)

    ax = kwplot.multi_plot(
        xydata=xydata, doclf=True, fnum=1,
        ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='FP-count', ylabel='TPR',
        title=prefix + 'perclass mAUC={:.4f}'.format(mAUC),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle',
    )
    return ax


def draw_perclass_prcurve(cx_to_peritem, classes, prefix=''):
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
        catname = classes[cx]
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
                label = 'ap={:0.2f}: {} ({:d}/{:d})'.format(ap, catname, int(peritem['realpos_total']), nsupport)
            else:
                label = 'ap={:0.2f}: {} ({}/{:d})'.format(ap, catname, peritem['realpos_total'], nsupport)
        else:
            label = 'ap={:0.2f}: {} ({:d})'.format(ap, catname, nsupport)
        xydata[label] = (recall, precision)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        mAP = np.nanmean(aps)

    ax = kwplot.multi_plot(
        xydata=xydata, doclf=True, fnum=1,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='recall', ylabel='precision',
        title=prefix + 'perclass mAP={:.4f}'.format(mAP),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle',
    )
    return ax


def draw_peritem_prcurve(peritem, prefix=''):
    """
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
        >>> draw_peritem_prcurve(peritem)
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
            label = 'ap={:0.2f}: ({:d}/{:d})'.format(ap, int(peritem['realpos_total']), nsupport)
        else:
            label = 'ap={:0.2f}: ({}/{:d})'.format(ap, peritem['realpos_total'], nsupport)
    else:
        label = 'ap={:0.2f}: ({:d})'.format(ap, nsupport)

    ax = kwplot.multi_plot(
        xdata=recall, ydata=precision, doclf=True, fnum=1, label=label,
        xlim=(0, 1), ylim=(0, 1), xpad=0.01, ypad=0.01,
        xlabel='recall', ylabel='precision',
        title=prefix + 'peritem AP={:.4f}'.format(ap),
        legend_loc='lower right',
        color='distinct', linestyle='cycle', marker='cycle',
    )
    return ax
