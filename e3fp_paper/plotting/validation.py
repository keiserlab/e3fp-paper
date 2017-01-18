"""Methods for plotting validation curves.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from e3fp_paper.plotting.defaults import DefaultFonts

fonts = DefaultFonts()

def plot_roc_curves(fp_tp_tuples_list, ax, names=None, show_legend=False,
                    title="", ref_line=True, colors=[], alpha=1.):
    """Plot random operator characteristic (ROC) curve(s)."""
    if names is None or len(names) != len(fp_tp_tuples_list):
        names = ["" for i in xrange(len(fp_tp_tuples_list))]

    lines = []
    for i, fp_tp_tuple in enumerate(fp_tp_tuples_list):
        try:
            color = colors[i]
        except IndexError:
            color = colors
        line, = ax.plot(*fp_tp_tuple, linewidth=2, label=names[i],
                        zorder=i + 2, color=color, alpha=alpha)
        lines.append(line)

    names = names[::-1]
    lines = lines[::-1]
    if ref_line:
        line, = ax.plot([0, 1], [0, 1], linewidth=1, color="lightgrey",
                        linestyle="--", label="Random", zorder=1)
        lines.append(line)
        names = names + ["Random"]

    ax.set_xlim(0., 1.01)
    ax.set_ylim(Y_CUTOFF - .01, 1.01)
    ax.set_aspect(1. / ax.get_data_ratio())
    ax.set_xlabel("False Positive Rate", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)
    if show_legend:
        ax.legend(lines, names, loc=3, fontsize=fonts.legend_fontsize)

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
