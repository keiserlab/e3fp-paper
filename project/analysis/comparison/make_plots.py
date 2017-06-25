"""Make main text E3FP/ECFP comparison/crossvalidation figure.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from python_utilities.scripting import setup_logging
from e3fp_paper.plotting.defaults import DefaultColors, DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels
from e3fp_paper.plotting.validation import plot_roc_curves, plot_prc_curves, \
                                           plot_auc_stats, plot_auc_scatter
from e3fp_paper.plotting.comparison import plot_tc_scatter, plot_tc_hists, \
                                           plot_tc_heatmap
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
OUT_IMAGE_BASENAME = "fig_2"
ROC_YMIN = .969
FRAC_POS = .0051
ECFP_TC_THRESH = .3
E3FP_TC_THRESH = .2
setup_logging(verbose=True)


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
                    cols=["ECFP4 TC", "E3FP Max TC"],
                    thresholds=[ECFP_TC_THRESH, E3FP_TC_THRESH])
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
