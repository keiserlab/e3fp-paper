"""Methods for plotting validation curves.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .defaults import DefaultFonts
from ..crossvalidation.util import get_auc


fonts = DefaultFonts()


def get_from_list(l, i):
    """Look up index in list, returning ``None`` if missing or wrong type."""
    try:
        v = l[i]
    except (IndexError, TypeError):
        v = None
    return v


def plot_roc_curves(roc_lists, ax, y_min=0., names=None, colors=None,
                    linestyles=None, ref_line=True, show_legend=True,
                    only_best=True, title="", alpha=1., show_inset=False,
                    legend_fontsize=fonts.legend_fontsize):
    """Plot ROC curves

    Parameters
    ----------
    roc_lists : list of ndarray or list of list of ndarray
        If list of ndarray, each array corresponds to a single ROC curve,
        where the first and second indices of the array correspond to the FPR
        and TPR, respectively. If list of list of ndarray, then each list of
        ndarray is considered to be a set of ROC curves that should be
        grouped.
    ax : axis
        matplotlib axis
    y_min : float, optional
        Minimum value of y, used to visually discriminate between well-
        performing curves.
    names : list of str or None, optional
        Names of ROC sets in `roc_lists`
    colors : list or None, optional
        Colors to be used for ROC sets in `roc_lists`
    linestyles : list or None, optional
        Line styles to be used for ROC sets in `roc_lists`
    ref_line : bool, optional
        Plot expected ROC curve of random classifier.
    show_legend : bool, optional
        Show legend.
    only_best : bool, optional
        Only plot best (highest AUC) ROC curve in a set.
    title : str, optional
        Title of plot
    alpha : float, optional
        Alpha of curves.
    show_inset : bool, optional
        Show inset of plot in lower right corner.
    legend_fontsize : int, optional
        Font size of legend
    """
    legend_lines = []
    legend_names = []

    if isinstance(roc_lists[0], np.ndarray):
        roc_list = [[x] for x in roc_lists]

    if show_inset:
        if show_legend:
            ax_inset = inset_axes(ax, "25%", "25%", loc=4, borderpad=.7)
        else:
            ax_inset = inset_axes(ax, "33%", "33%", loc=4, borderpad=.7)

    for i, roc_list in enumerate(roc_lists):
        color = get_from_list(colors, i)
        name = get_from_list(names, i)
        linestyle = get_from_list(linestyles, i)
        aucs = [get_auc(x[0], x[1]) for x in roc_list]
        auc_rocs = sorted(zip(aucs, roc_list), reverse=True)
        try:
            auc = auc_rocs[0][0]
        except IndexError:
            continue
        if only_best:
            line, = ax.plot(auc_rocs[0][1][0], auc_rocs[0][1][1], linewidth=2,
                            linestyle=linestyle, zorder=i + 2, color=color,
                            alpha=alpha)
            if show_inset:
                ax_inset.plot(auc_rocs[0][1][0], auc_rocs[0][1][1],
                              linewidth=1, linestyle=linestyle, zorder=i + 2,
                              color=color, alpha=alpha)
            name += " ({:.4f})".format(auc)
        else:
            for j, (auc, roc) in enumerate(auc_rocs):
                line, = ax.plot(roc[0], roc[1], linewidth=1,
                                linestyle=linestyle, zorder=i + 2, color=color,
                                alpha=alpha)
                if show_inset:
                    ax_inset.plot(roc[0], roc[1], linewidth=1,
                                  linestyle=linestyle, zorder=i + 2,
                                  color=color, alpha=alpha)

            name += " ({:.4f} +/- {:.4f})".format(np.mean(zip(*auc_rocs)[0]),
                                                  np.std(zip(*auc_rocs)[0]))
        legend_lines.append(line)
        legend_names.append(name)

    legend_lines = legend_lines[::-1]
    legend_names = legend_names[::-1]
    if ref_line:
        line, = ax.plot([0, 1], [0, 1], linewidth=1, color="lightgrey",
                        linestyle="--", label="Random", zorder=1)
        if show_inset:
            ax_inset.plot([0, 1], [0, 1], linewidth=1, color="lightgrey",
                          linestyle="--", label="Random", zorder=1)
        legend_lines.append(line)
        legend_names.append("Random (0.5)")

    ax.set_xlim(0., 1.01)
    ax.set_ylim(y_min, 1.001)
    ax.set_aspect(1. / ax.get_data_ratio())
    if show_inset:
        ax_inset.set_xlim(-.02, 1.02)
        ax_inset.set_ylim(-.02, 1.02)
        ax_inset.set_aspect(1.)
        ax_inset.set_xticks([0., 1.])
        ax_inset.set_yticks([0., 1.])
    ax.set_xlabel("False Positive Rate", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)
    if show_legend:
        if show_inset:
            legend_loc = (.125, 0)
        else:
            legend_loc = 'lower center'
        legend = ax.legend(legend_lines, legend_names, loc=legend_loc,
                           fontsize=legend_fontsize, frameon=True)
        legend.get_frame().set_linewidth(0)


def plot_enrichment_curves(curve_lists, ax, *args, **kwargs):
    plot_roc_curves(curve_lists, ax, *args, **kwargs)
    ax.set_ylabel("Fraction of Actives Found",
                  fontsize=fonts.ax_label_fontsize)
    ax.set_xlabel("Fraction of Database Screened",
                  fontsize=fonts.ax_label_fontsize)


def plot_prc_curves(prc_lists, ax, names=None, colors=None, linestyles=None,
                    ref_val=None, show_legend=True, only_best=True, title="",
                    alpha=1., legend_fontsize=fonts.legend_fontsize):
    """Plot precision-recall (PRC) curves

    Parameters
    ----------
    prc_lists : list of ndarray or list of list of ndarray
        If list of ndarray, each array corresponds to a single PRC curve,
        where the first and second indices of the array correspond to the FPR
        and TPR, respectively. If list of list of ndarray, then each list of
        ndarray is considered to be a set of PRC curves that should be
        grouped.
    ax : axis
        matplotlib axis
    names : list of str or None, optional
        Names of PRC sets in `prc_lists`
    colors : list or None, optional
        Colors to be used for PRC sets in `prc_lists`
    linestyles : list or None, optional
        Line styles to be used for ROC sets in `prc_lists`
    ref_val : float or None, optional
        If float is provided, value corresponds to the PRC expected from a
        random classifier, which is a horizontal line corresponding to percent
        of pairs which are positives.
    show_legend : bool, optional
        Show legend.
    only_best : bool, optional
        Only plot best (highest AUC) PRC curve in a set.
    title : str, optional
        Title of plot
    alpha : float, optional
        Alpha of curves.
    legend_fontsize : int, optional
        Font size of legend

    Deleted Parameters
    ------------------
    y_min : float, optional
        Minimum value of y, used to visually discriminate between well-
        performing curves.
    """
    legend_lines = []
    legend_names = []

    if isinstance(prc_lists[0], np.ndarray):
        prc_list = [[x] for x in prc_lists]

    for i, prc_list in enumerate(prc_lists):
        color = get_from_list(colors, i)
        name = get_from_list(names, i)
        linestyle = get_from_list(linestyles, i)
        aucs = [get_auc(x[0], x[1]) for x in prc_list]
        auc_prcs = sorted(zip(aucs, prc_list), reverse=True)
        try:
            auc = auc_prcs[0][0]
        except IndexError:
            continue
        if only_best:
            line, = ax.plot(auc_prcs[0][1][0], auc_prcs[0][1][1], linewidth=2,
                            linestyle=linestyle, zorder=i + 2, color=color,
                            alpha=alpha)
            name += " ({:.4f})".format(auc)
        else:
            for j, (auc, prc) in enumerate(auc_prcs):
                line, = ax.plot(prc[0], prc[1], linewidth=1,
                                linestyle=linestyle, zorder=i + 2,
                                color=color, alpha=alpha)
            name += " ({:.4f} +/- {:.4f})".format(np.mean(zip(*auc_prcs)[0]),
                                                  np.std(zip(*auc_prcs)[0]))
        legend_lines.append(line)
        legend_names.append(name)

    legend_lines = legend_lines[::-1]
    legend_names = legend_names[::-1]
    if ref_val is not None:
        line, = ax.plot([0, 1], [ref_val, ref_val], linewidth=1,
                        color="lightgrey", linestyle="--", label="Random",
                        zorder=1)
        legend_lines.append(line)
        legend_names.append("Random ({:.4f})".format(ref_val))

    ax.set_xlim(0., 1.01)
    ax.set_ylim(0, 1.001)
    ax.set_aspect(1. / ax.get_data_ratio())
    ax.set_xlabel("Recall", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel("Precision", fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)
    if show_legend:
        legend = ax.legend(legend_lines, legend_names, loc='lower left',
                           fontsize=legend_fontsize, frameon=True)
        legend.get_frame().set_linewidth(0)


def plot_auc_stats(repeat_aucs_list, ax, names=None, colors=None,
                   show_legend=True, show_inset=False, xlabel=""):
    """Plot errorbars of AUCs.

    Parameters
    ----------
    repeat_aucs_list : list of list of float
        List of list of fold AUCs from bootstrapped ROC curves.
    ax : axis
        matplotlib axis
    names : list of str or None, optional
        Names of PRC sets in `prc_lists`
    colors : list or None, optional
        Colors to be used for PRC sets in `prc_lists`
    show_legend : bool, optional
        Show legend
    show_inset : bool, optional
        Show insert over full range
    xlabel : str, optional
        X-axis label, indicating type of AUC
    """
    legend_lines = []
    if show_inset:
        xmax = .52
        ax_inset = inset_axes(ax, .3, 1.2, loc=4, borderpad=.2)
    else:
        xmax = .85

    ticks = np.linspace(.04, xmax, len(repeat_aucs_list))

    legend_names = []
    for i, aucs in enumerate(repeat_aucs_list):
        color = get_from_list(colors, i)
        name = get_from_list(names, i)
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        _, caps, _ = ax.errorbar(ticks[i], mean_auc, yerr=std_auc,
                                 zorder=i + 2, color=color, fmt='o',
                                 capsize=2, ms=6, linewidth=1)
        legend_names.append(name)
        legend_lines.append(caps)

        for cap in caps:
            cap.set_markeredgewidth(1)

        if show_inset:
            _, caps, _ = ax_inset.errorbar(ticks[i], mean_auc, yerr=std_auc,
                                           ms=4, zorder=i + 2, color=color,
                                           fmt='o', capsize=1.5, linewidth=1)
            for cap in caps:
                cap.set_markeredgewidth(1)

    ax.set_xlim(-.05, 1.25)
    ax.set_xlabel(xlabel)
    if names is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(legend_names, rotation=45)
    else:
        ax.set_xticks([])

    if show_inset:
        ax_inset.set_xlim(0., xmax + .1)
        ax_inset.set_ylim(-0.1, 1.1)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([0, 1])

    if show_legend and names is not None:
        ax.legend(legend_lines, names, loc=3, fontsize=fonts.legend_fontsize)


def plot_auc_scatter(aucs_dictx, aucs_dicty, ax, colors=None, ref_line=True,
                     xlabel="X AUCs", ylabel="Y AUCs", title=""):
    """Make scatter plot of AUCs.

    Parameters
    ----------
    aucs_dictx : dict
        dict matching key to AUC Only keys common to `aucs_dicty` are used.
    aucs_dicty : dict
        dict matching key to AUC. Only keys common to `aucs_dictx` are used.
    ax : axis
        Matplotlib axis
    colors : list or None, optional
        Colors to be used for ROC sets in `roc_lists`
    ref_line : bool, optional
        Plot expected ROC curve of random classifier.
    xlabel : str, optional
        x-axis label
    ylabel : str, optional
        y-axis label
    title : str, optional
        Title of plot
    """
    data = np.array([(v, aucs_dicty[k]) for k, v in aucs_dictx.iteritems()
                     if k in aucs_dicty], dtype=np.double)

    if ref_line:
        ax.plot([0, 1], [0, 1], linewidth=1, color="lightgrey",
                linestyle="--", label="Random", zorder=1)

    above = np.where(data[:, 1] > data[:, 0])
    below = np.where(data[:, 1] <= data[:, 0])
    x, y = data[above].T
    ax.scatter(x, y, s=20, marker="o", facecolors=colors[0],
               edgecolors='none', alpha=0.3, zorder=3)
    x, y = data[below].T
    ax.scatter(x, y, s=20, marker="o", facecolors=colors[1],
               edgecolors='none', alpha=0.3, zorder=2)

    min_val = np.amin(data)
    ax_min = int(min_val * 20) / 20.0
    ax.set_xlim(ax_min, 1.005)
    ax.set_ylim(ax_min, 1.005)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)
