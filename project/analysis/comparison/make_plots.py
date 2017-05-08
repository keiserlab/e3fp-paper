"""Make main text E3FP/ECFP comparison/crossvalidation figure.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import logging

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from python_utilities.scripting import setup_logging
from e3fp_paper.plotting.defaults import DefaultColors, DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels
from e3fp_paper.plotting.validation import plot_roc_curves, plot_prc_curves, \
                                           plot_auc_stats, plot_auc_scatter
from e3fp_paper.crossvalidation.util import rocs_from_cv_dir, \
                                            prcs_from_cv_dir, \
                                            prc_roc_aucs_from_cv_dirs, \
                                            target_aucs_from_cv_dirs

fonts = DefaultFonts()
def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
CV_BASEDIR = os.path.join(PROJECT_DIR, 'crossvalidation', 'sea')
E3FP_CV_DIR = os.path.join(CV_BASEDIR, "e3fp")
ECFP_CV_DIR = os.path.join(CV_BASEDIR, "ecfp4")

E3FP_NOSTEREO_REPEAT_DIRS = glob.glob(os.path.join(CV_BASEDIR,
                                                   "e3fp-nostereo*"))
E3FP_REPEAT_DIRS = [x for x in glob.glob(os.path.join(CV_BASEDIR, "e3fp*"))
                    if x not in E3FP_NOSTEREO_REPEAT_DIRS]
E2FP_STEREO_REPEAT_DIRS = glob.glob(os.path.join(CV_BASEDIR, "e2fp-stereo*"))
E2FP_REPEAT_DIRS = [x for x in glob.glob(os.path.join(CV_BASEDIR, "e2fp*"))
                    if x not in E2FP_STEREO_REPEAT_DIRS]
ECFP_CHIRAL_REPEAT_DIRS = glob.glob(os.path.join(CV_BASEDIR, "ecfp4-chiral*"))
ECFP_REPEAT_DIRS = [x for x in glob.glob(os.path.join(CV_BASEDIR, "ecfp4*"))
                    if x not in ECFP_CHIRAL_REPEAT_DIRS]

TC_COUNTS_FILE = os.path.join(PROJECT_DIR, "fingerprint_comparison",
                              "ecfp_e3fp_tcs_counts.csv.bz2")
EXAMPLES_FILE = os.path.join(PROJECT_DIR, "data", "examples.txt")
OUT_IMAGE_BASENAME = "fig_1"
ROC_YMIN = .969
FRAC_POS = .0051
ECFP_TC_THRESH = .3
E3FP_TC_THRESH = .2
setup_logging(verbose=True)


def plot_tc_scatter(outlier_df, ax):
    ul = outlier_df.loc["Upper left"]
    ur = outlier_df.loc["Upper"]
    lr = outlier_df.loc["Lower Right"]
    fr = outlier_df.loc["Far Right"]
    for df, ms in zip([ul, ur, lr, fr], ['X', 'P', 's', 'D']):
        ax.scatter(df["ECFP4 TC"], df["E3FP TC"], s=20, linewidths=1,
                   marker=ms, facecolors='r', edgecolors='none', alpha=.8,
                   zorder=100)


def calculate_line(x, m=1., b=0.):
    return m * x + b


def get_weighted_leastsq(xdata, ydata, weights=None):
    if weights is None:
        sigma = None
        weights = np.ones_like(xdata)
    else:
        sigma = 1. / np.sqrt(weights)
    popt, pcov = curve_fit(calculate_line, xdata, ydata, [1., 0.], sigma=sigma)
    residuals = ydata - calculate_line(xdata, *popt)
    resid_sumsq = np.dot(weights, residuals**2)
    mean_y = np.average(ydata, weights=weights)
    total_sumsq = np.dot(weights, (ydata - mean_y)**2)
    r2 = 1. - (resid_sumsq / total_sumsq)
    return popt[0], popt[1], r2


def plot_tc_heatmap(counts_df, ax, outliers_df=None, cols=[], title="",
                    ref_line=True, cutoffs=[], cmap="bone_r", limit=.5):
    x, y, count = counts_df[cols[0]], counts_df[cols[1]], counts_df["Count"]

    if ref_line:
        ax.plot([0, 1], [0, 1], linewidth=1, color="gray", alpha=.5,
                linestyle="--", label="Equal", zorder=2)
        ax.axhline(E3FP_TC_THRESH, linewidth=1, color="forestgreen",
                   alpha=.75, linestyle='--', zorder=2)
        ax.axvline(ECFP_TC_THRESH, linewidth=1, color="royalblue", alpha=.75,
                   linestyle='--', zorder=2)
        slope, intercept, r2 = get_weighted_leastsq(x, y, weights=count)
        ax.plot([0, 1], [intercept, slope + intercept], linewidth=1,
                color="r", alpha=.75, linestyle="--", label="Trend", zorder=2)
        print("Fit with slope {:.4f}, intercept {:.4f}, and R^2 {:.4f}".format(
            slope, intercept, r2))

    max_val = max(max(x), max(y), limit)
    extent = (0, max_val, 0, max_val)

    ax.hexbin(x, y, C=count, cmap=cmap, norm=matplotlib.colors.LogNorm(),
              gridsize=50, extent=extent, edgecolors='none', zorder=1)

    if outliers_df is not None:
        x, y = outliers_df[cols[0]], outliers_df[cols[1]]
        if len(x) > 0:
            ax.scatter(x, y, s=15, marker="x", facecolors="r",
                       edgecolors='none', alpha=.3, zorder=3)

    ax.set_xlim(0., max_val + .02)
    ax.set_ylim(0., max_val + .02)
    ax.set_aspect('equal')
    ax.set_xlabel(cols[0], fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel(cols[1], fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)


def plot_tc_hists(counts_df, ax, cols=[], title="", colors=[], thresholds=[]):
    ax.set_yscale("log")
    count = counts_df["Count"]
    ref_line_colors = ["royalblue", "forestgreen"]
    for i, col in enumerate(cols):
        name = col.split(" ")[0]
        tcs = counts_df[col]
        alpha = 1.
        if i > 0:
            alpha = .9
        else:
            alpha = 1.
        try:
            color = colors[i]
        except IndexError:
            color = None
        sns.distplot(tcs, label=name, kde=False, ax=ax, norm_hist=True,
                     color=color, hist_kws={"linewidth": 0,
                                            "histtype": "stepfilled",
                                            "alpha": alpha,
                                            "zorder": 2 * i + 1,
                                            "weights": count})
        ax.axvline(thresholds[i], linewidth=1, color=ref_line_colors[i],
                   alpha=.75, linestyle='--', zorder=2 * i + 2)

    ax.set_xlabel("TC", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel("Log Frequency", fontsize=fonts.ax_label_fontsize)
    ax.set_xlim(0, counts_df[cols].values.max())
    ax.set_title(title, fontsize=fonts.title_fontsize)
    ax.legend(loc=1, fontsize=fonts.legend_fontsize)


def get_best_fold_by_prc(cv_dir):
    log_file = os.path.join(cv_dir, "log.txt")
    logging.debug("Opening log file: {}".format(log_file))
    fold_results = []
    with open(log_file, "r") as f:
        for line in f:
            if "|Fold " in line:
                fold = int(line.split('Fold ')[1].split()[0])
                auprc = float(line.split('AUPRC of ')[1].split()[0][:-1])
                fold_results.append((auprc, fold))
    return sorted(fold_results)[-1][1]


def get_best_fold_by_roc(cv_dir):
    log_file = os.path.join(cv_dir, "log.txt")
    logging.debug("Opening log file: {}".format(log_file))
    fold_results = []
    with open(log_file, "r") as f:
        for line in f:
            if "|Fold " in line:
                fold = int(line.split('Fold ')[1].split()[0])
                auroc = float(line.split('AUROC of ')[1].split()[0])
                fold_results.append((auroc, fold))
    return sorted(fold_results)[-1][1]


if __name__ == "__main__":
    panel_axes = []
    panel_xoffsets = []
    fig = plt.figure(figsize=(6.4, 8))
    gs = gridspec.GridSpec(3, 4)
    tc_counts_df = pd.DataFrame.from_csv(TC_COUNTS_FILE, sep="\t",
                                         index_col=None)
    tc_examples_df = pd.DataFrame.from_csv(EXAMPLES_FILE, sep="\t")
    ax = plt.subplot(gs[0, :2])
    plot_tc_heatmap(tc_counts_df, ax=ax, outliers_df=None,
                    cols=["ECFP4 TC", "E3FP Max TC"], cutoffs=[.3, .18])
    plot_tc_scatter(tc_examples_df, ax)
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.23)

    ax = plt.subplot(gs[0, 2:])
    plot_tc_hists(tc_counts_df, cols=["ECFP4 TC", "E3FP Max TC"], ax=ax,
                  colors=[def_colors.ecfp_color, def_colors.e3fp_color],
                  thresholds=[ECFP_TC_THRESH, E3FP_TC_THRESH])
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.20)

    cv_repeat_dirs = [ECFP_REPEAT_DIRS, ECFP_CHIRAL_REPEAT_DIRS,
                      E2FP_REPEAT_DIRS, E2FP_STEREO_REPEAT_DIRS,
                      E3FP_NOSTEREO_REPEAT_DIRS,
                      E3FP_REPEAT_DIRS]
    cv_repeat_names = ["ECFP4", "ECFP4-Chiral", "E2FP", "E2FP-Stereo",
                       "E3FP-NoStereo", "E3FP"]
    all_colors = [def_colors.ecfp_color, def_colors.ecfp_chiral_color,
                  def_colors.e2fp_color, def_colors.e2fp_stereo_color,
                  def_colors.e3fp_nostereo_color, def_colors.e3fp_color]
    ax = plt.subplot(gs[1, :2])
    prc_lists = [[z for y in x for z in prcs_from_cv_dir(y)]
                 for x in cv_repeat_dirs]
    plot_prc_curves(prc_lists, ref_val=FRAC_POS, ax=ax, names=cv_repeat_names,
                    colors=all_colors, alpha=.75, only_best=True)
    del prc_lists
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.20)

    ax = plt.subplot(gs[1, 2:])
    roc_lists = [[z for y in x for z in rocs_from_cv_dir(y)]
                 for x in cv_repeat_dirs]
    plot_roc_curves(roc_lists, ax=ax, names=cv_repeat_names,
                    colors=all_colors, y_min=ROC_YMIN, alpha=.75,
                    only_best=True)
    del roc_lists
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.28)

    repeat_aucs = [prc_roc_aucs_from_cv_dirs(x) for x in cv_repeat_dirs]
    ax1 = plt.subplot(gs[2, 0])
    plot_auc_stats(
        [x[1] for x in repeat_aucs], ax=ax1, names=None,
        colors=all_colors,
        xlabel="AUPRC", show_legend=False, show_inset=True)
    ax2 = plt.subplot(gs[2, 1])
    plot_auc_stats(
        [x[0] for x in repeat_aucs], ax=ax2, names=None,
        colors=all_colors,
        xlabel="AUROC", show_legend=False, show_inset=True)
    sns.despine(ax=ax1, offset=10)
    sns.despine(ax=ax2, offset=10)
    panel_axes.append(ax1)
    panel_xoffsets.append(.35)

    ax = plt.subplot(gs[2, 2:])
    _, ecfp_auprc_dicts = target_aucs_from_cv_dirs(ECFP_REPEAT_DIRS)
    ecfp_auprc_dicts = {k: np.mean(v) for k, v in ecfp_auprc_dicts.items()}
    _, e3fp_auprc_dicts = target_aucs_from_cv_dirs(E3FP_REPEAT_DIRS)
    e3fp_auprc_dicts = {k: np.mean(v) for k, v in e3fp_auprc_dicts.items()}
    plot_auc_scatter(ecfp_auprc_dicts, e3fp_auprc_dicts, ax,
                     xlabel="ECFP4 Mean AUPRC",
                     ylabel="E3FP Mean AUPRC",
                     colors=[def_colors.e3fp_color, def_colors.ecfp_color])
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.23)

    add_panel_labels(axes=panel_axes, xoffset=panel_xoffsets)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(OUT_IMAGE_BASENAME + ".png", dpi=300)
    fig.savefig(OUT_IMAGE_BASENAME + ".tif", dpi=300)
