"""Methods for plotting validation curves.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import numpy as np
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
                    ref_line=True, show_legend=True, only_best=True, title="",
                    alpha=1.):
    """Plot ROC curves

    Parameters
    ----------
    roc_lists : list of ndarray or list of list of ndarray
        If list of ndarray, each array corresponds to a single ROC curve, where
        the first and second indices of the array correspond to the FPR and TPR,
        respectively. If list of list of ndarray, then each list of ndarray is
        considered to be a set of ROC curves that should be grouped. 
    ax : axis
        matplotlib axis
    y_min : float, optional
        Minimum value of y, used to visually discriminate between well-
        performing curves.
    names : list of str or None, optional
        Names of ROC sets in `roc_lists`
    colors : list or None, optional
        Colors to be used for ROC sets in `roc_lists`
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
    """
    legend_lines = []
    legend_names = []

    if isinstance(roc_lists[0], np.ndarray):
        roc_list = [[x] for x in roc_list]

    for i, roc_list in enumerate(roc_lists):
        color = get_from_list(colors, i)
        name = get_from_list(names, i)
        aucs = [get_auc(x[0], x[1]) for x in roc_list]
        auc_rocs = sorted(zip(aucs, roc_list), reverse=True)
        auc = auc_rocs[0][0]
        if only_best:
            line, = ax.plot(auc_rocs[0][1][0], auc_rocs[0][1][1], linewidth=1,
                            zorder=i + 2, color=color, alpha=alpha)
            name += " ({:.4f})".format(auc)
        else:
            for j, (auc, roc) in enumerate(auc_rocs):
                line, = ax.plot(roc[0], roc[1], linewidth=1,
                                zorder=i + 2, color=color, alpha=alpha)
            name += " ({:.4f} +/- {:.4f})".format(np.mean(zip(*auc_rocs)[0]),
                                                  np.std(zip(*auc_rocs)[0]))
        legend_lines.append(line)
        legend_names.append(name)

    legend_lines = legend_lines[::-1]
    legend_names = legend_names[::-1]
    if ref_line:
        line, = ax.plot([0, 1], [0, 1], linewidth=1, color="lightgrey",
                        linestyle="--", label="Random", zorder=1)
        legend_lines.append(line)
        legend_names.append("Random (0.5)")

    ax.set_xlim(0., 1.01)
    ax.set_ylim(y_min, 1.001)
    ax.set_aspect(1. / ax.get_data_ratio())
    ax.set_xlabel("False Positive Rate", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)
    if show_legend:
        ax.legend(legend_lines, legend_names, loc=3,
                  fontsize=fonts.legend_fontsize)


def plot_prc_curves(rec_prec_tuples_list, ax, names=None, show_legend=False,
                    title="", ref_val=None, colors=None, alpha=1.):
    """Plot precision-recall (PRC) curve(s)."""
    if names is None or len(names) != len(rec_prec_tuples_list):
        names = ["" for i in xrange(len(rec_prec_tuples_list))]

    lines = []
    for i, rec_prec_tuple in enumerate(rec_prec_tuples_list):
        try:
            color = colors[i]
        except IndexError:
            color = colors
        line, = ax.plot(*rec_prec_tuple, linewidth=2, label=names[i],
                        zorder=i + 2, color=color, alpha=alpha)
        lines.append(line)

    names = names[::-1]
    lines = lines[::-1]
    if ref_val is not None:
        line, = ax.plot([0, 1], [ref_val, ref_val], linewidth=1,
                        color="lightgrey", linestyle="--", label="Random",
                        zorder=1)
        lines.append(line)
        names = names + ["Random"]

    ax.set_xlim(0., 1.01)
    ax.set_ylim(0., 1.01)
    ax.set_aspect(1. / ax.get_data_ratio())
    ax.set_xlabel("Recall", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel("Precision", fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)
    if show_legend:
        ax.legend(lines, names, loc=3, fontsize=fonts.legend_fontsize)
