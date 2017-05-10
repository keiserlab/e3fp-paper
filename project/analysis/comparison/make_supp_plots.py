"""Make supplementary crossvalidation PRC/ROC figures.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import logging

from matplotlib import pyplot as plt
import seaborn as sns

from python_utilities.scripting import setup_logging
from e3fp_paper.plotting.defaults import DefaultColors
from e3fp_paper.plotting.util import add_panel_labels
from e3fp_paper.plotting.validation import plot_roc_curves, plot_prc_curves
from e3fp_paper.crossvalidation.util import rocs_from_cv_dir, \
                                            prcs_from_cv_dir

sns.set_style("white")

def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
CV_BASEDIR = os.path.join(PROJECT_DIR, 'crossvalidation', 'sea')
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

SEA_PRC_BASENAME = "fig_s7"
SEA_ROC_BASENAME = "fig_s8"

CV_BASEDIR = os.path.join(PROJECT_DIR, 'crossvalidation', 'classifiers')
E3FP_NB_DIR = os.path.join(CV_BASEDIR, "e3fp_nbc")
ECFP_NB_DIR = os.path.join(CV_BASEDIR, "ecfp4_nbc")
E3FP_RF_DIR = os.path.join(CV_BASEDIR, "e3fp_rf")
ECFP_RF_DIR = os.path.join(CV_BASEDIR, "ecfp4_rf")
E3FP_SVM_DIR = os.path.join(CV_BASEDIR, "e3fp_linsvm")
ECFP_SVM_DIR = os.path.join(CV_BASEDIR, "ecfp4_linsvm")
E3FP_NN_DIR = os.path.join(CV_BASEDIR, "e3fp_nn")
ECFP_NN_DIR = os.path.join(CV_BASEDIR, "ecfp4_nn")

CV_MEAN_BASEDIR = os.path.join(PROJECT_DIR, 'crossvalidation',
                               'classifiers_mean')
ECFP_MEAN_NB_DIR = os.path.join(CV_MEAN_BASEDIR, "ecfp4_nbc")
ECFP_MEAN_RF_DIR = os.path.join(CV_MEAN_BASEDIR, "ecfp4_rf")
ECFP_MEAN_SVM_DIR = os.path.join(CV_MEAN_BASEDIR, "ecfp4_linsvm")
ECFP_MEAN_NN_DIR = os.path.join(CV_MEAN_BASEDIR, "ecfp4_nn")
E3FP_MEAN_NB_DIR = os.path.join(CV_MEAN_BASEDIR, "e3fp_nbc")
E3FP_MEAN_RF_DIR = os.path.join(CV_MEAN_BASEDIR, "e3fp_rf")
E3FP_MEAN_SVM_DIR = os.path.join(CV_MEAN_BASEDIR, "e3fp_linsvm")
E3FP_MEAN_NN_DIR = os.path.join(CV_MEAN_BASEDIR, "e3fp_nn")
SKLEARN_PRC_BASENAME = "fig_s5"
SKLEARN_ROC_BASENAME = "fig_s6"

SEA_ROC_YMIN = .969
SKLEARN_ROC_YMIN = .874
FRAC_POS = .0051
setup_logging(verbose=True)


def get_best_fold_by_prc(cv_dir):
    """Get number of best fold by fold AUPRC."""
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
    """Get number of best fold by fold AUROC."""
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
    cv_repeat_dirs = [ECFP_REPEAT_DIRS, ECFP_CHIRAL_REPEAT_DIRS,
                      E2FP_REPEAT_DIRS, E2FP_STEREO_REPEAT_DIRS,
                      E3FP_NOSTEREO_REPEAT_DIRS,
                      E3FP_REPEAT_DIRS]
    cv_repeat_names = ["ECFP4", "ECFP4-Chiral", "E2FP", "E2FP-Stereo",
                       "E3FP-NoStereo", "E3FP"]
    panel_names = ["A", "B", "C", "D", "E", "F"]
    all_colors = [def_colors.ecfp_color, def_colors.ecfp_chiral_color,
                  def_colors.e2fp_color, def_colors.e2fp_stereo_color,
                  def_colors.e3fp_nostereo_color, def_colors.e3fp_color]

    # SEA PRC figure
    panel_axes = []
    fig = plt.figure(figsize=(5.3, 8))
    for i, (cv_dirs, name, panel_name, color) in enumerate(
            zip(cv_repeat_dirs, cv_repeat_names, panel_names, all_colors)):
        ax = fig.add_subplot(3, 2, i + 1)
        prc_lists = [[prcs_from_cv_dir(x)[0]] for x in cv_dirs]
        names = [name] * len(prc_lists)
        colors = [color] * len(prc_lists)
        plot_prc_curves(prc_lists, ref_val=FRAC_POS, ax=ax, names=names,
                        colors=colors, alpha=.6, title=name, only_best=False,
                        show_legend=False)
        sns.despine(ax=ax, offset=10)
        panel_axes.append(ax)
    add_panel_labels(fig, axes=panel_axes, xoffset=.25)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(SEA_PRC_BASENAME + ".png", dpi=300)
    fig.savefig(SEA_PRC_BASENAME + ".tif", dpi=300)

    # SEA ROC figure
    panel_axes = []
    fig = plt.figure(figsize=(5.3, 8))
    for i, (cv_dirs, name, panel_name, color) in enumerate(
            zip(cv_repeat_dirs, cv_repeat_names, panel_names, all_colors)):
        ax = fig.add_subplot(3, 2, i + 1)
        roc_lists = [rocs_from_cv_dir(x) for x in cv_dirs]
        names = [name] * len(roc_lists)
        colors = [color] * len(roc_lists)
        plot_roc_curves(roc_lists, ref_line=True, ax=ax, names=names,
                        colors=colors, alpha=.6, title=name, only_best=False,
                        y_min=SEA_ROC_YMIN, show_legend=False, show_inset=True)
        sns.despine(ax=ax, offset=10)
        panel_axes.append(ax)
    add_panel_labels(fig, axes=panel_axes, xoffset=.3)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(SEA_ROC_BASENAME + ".png", dpi=300)
    fig.savefig(SEA_ROC_BASENAME + ".tif", dpi=300)

    sklearn_dir_pairs = [(ECFP_NB_DIR, E3FP_NB_DIR,
                          ECFP_MEAN_NB_DIR, E3FP_MEAN_NB_DIR),
                         (ECFP_RF_DIR, E3FP_RF_DIR,
                          ECFP_MEAN_RF_DIR, E3FP_MEAN_RF_DIR),
                         (ECFP_SVM_DIR, E3FP_SVM_DIR,
                          ECFP_MEAN_SVM_DIR, E3FP_MEAN_SVM_DIR),
                         (ECFP_NN_DIR, E3FP_NN_DIR,
                          ECFP_MEAN_NN_DIR, E3FP_MEAN_NN_DIR)]
    sklearn_names = ["NBC", "RF", "LinSVM", "NN"]
    panel_names = ["A", "B", "C", "D"]
    cv_names = ["ECFP4", "E3FP"]
    cv_colors = [def_colors.ecfp_color, def_colors.e3fp_color]

    # SKLearn PRC figure
    fig = plt.figure(figsize=(7, 7))
    for i, (cv_dir_pair, name, panel_name) in enumerate(
            zip(sklearn_dir_pairs, sklearn_names, panel_names)):
        ax = fig.add_subplot(2, 2, i + 1)
        best_folds = [get_best_fold_by_prc(x) for x in cv_dir_pair]
        prc_lists = [prcs_from_cv_dir(x, fold=f) for x, f
                     in zip(cv_dir_pair, best_folds)]
        names = cv_names + ["Mean " + x for x in cv_names]
        linestyles = ['-' for x in cv_names] + ["--" for x in cv_names]
        colors = cv_colors + cv_colors
        plot_prc_curves(prc_lists, ref_val=FRAC_POS, ax=ax, names=names,
                        colors=colors, linestyles=linestyles, alpha=.6,
                        only_best=True, title=name)
        del prc_lists
        sns.despine(ax=ax, offset=10)
    add_panel_labels(fig)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(SKLEARN_PRC_BASENAME + ".png", dpi=300)
    fig.savefig(SKLEARN_PRC_BASENAME + ".tif", dpi=300)

    # SKLearn ROC figure
    panel_axes = []
    fig = plt.figure(figsize=(7, 7))
    for i, (cv_dir_pair, name, panel_name) in enumerate(
            zip(sklearn_dir_pairs, sklearn_names, panel_names)):
        ax = fig.add_subplot(2, 2, i + 1)
        best_folds = [get_best_fold_by_roc(x) for x in cv_dir_pair]
        roc_lists = [rocs_from_cv_dir(x, fold=f) for x, f
                     in zip(cv_dir_pair, best_folds)]
        names = cv_names + ["Mean " + x for x in cv_names]
        linestyles = ['-' for x in cv_names] + ["--" for x in cv_names]
        colors = cv_colors + cv_colors
        plot_roc_curves(roc_lists, ax=ax, names=names, colors=colors,
                        linestyles=linestyles, y_min=SKLEARN_ROC_YMIN,
                        alpha=.6, only_best=True, title=name, show_inset=True)
        del roc_lists
        sns.despine(ax=ax, offset=10)
        panel_axes.append(ax)
    add_panel_labels(fig, axes=panel_axes, xoffset=.2)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(SKLEARN_ROC_BASENAME + ".png", dpi=300)
    fig.savefig(SKLEARN_ROC_BASENAME + ".tif", dpi=300)
