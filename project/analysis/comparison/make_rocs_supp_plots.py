"""Make supplementary text ROCS comparison figures.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from python_utilities.scripting import setup_logging
from e3fp_paper.crossvalidation.util import enrichments_from_cv_dir
from e3fp_paper.plotting.defaults import DefaultColors, DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels
from e3fp_paper.plotting.comparison import plot_tc_hists, plot_tc_heatmap
from e3fp_paper.plotting.validation import plot_enrichment_curves

fonts = DefaultFonts()
def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
BASEDIR = os.path.join(PROJECT_DIR, 'fingerprint_comparison')
SHAPE_TC_COUNTS_FILE = os.path.join(BASEDIR, "rocs-shape_e3fp_tcs.csv.gz")
COMBO_TC_COUNTS_FILE = os.path.join(BASEDIR, "rocs-combo_e3fp_tcs.csv.gz")
CV_BASEDIR = os.path.join(BASEDIR, "maxtc_enrichment")
ECFP4_CV_DIR = os.path.join(CV_BASEDIR, "ecfp4_maxtc_filtered")
E3FP_CV_DIR = os.path.join(CV_BASEDIR, "e3fp_maxtc_filtered")
ROCS_SHAPE_CV_DIR = os.path.join(CV_BASEDIR, "rocs-shape_maxtc_filtered")
ROCS_COMBO_CV_DIR = os.path.join(CV_BASEDIR, "rocs-combo_maxtc_filtered")

RANDOM_CONF_DIR = os.path.join(BASEDIR, "random_conformers")
RANDOM_PAIR_DIR = os.path.join(BASEDIR, "random_conformer_pairs")

OUT_GLOBAL_BASENAME = "rocs_tcs_supp_v3"
OUT_CONFS_BASENAME = "rocs_conf_rmsds_supp_v3"
EC_YMIN = .6
setup_logging(verbose=True)


def get_best_fold_by_enrichment(cv_dir):
    """Get number of best fold by fold AUEC."""
    log_file = os.path.join(cv_dir, "cv_log.txt")
    logging.debug("Opening log file: {}".format(log_file))
    fold_results = []
    with open(log_file, "r") as f:
        for line in f:
            if "|Fold " in line and 'enrich' in line:
                fold = int(line.split('Fold ')[1].split()[0])
                auec = float(line.split('AUC of ')[1].split()[0][:-1])
                fold_results.append((auec, fold))
    return sorted(fold_results)[-1][1]


def remove_bad_pairs(df):
    cols = df.columns[:2]
    return df.loc[(df[cols[0]] >=0) & (df[cols[1]] >= 0)]


def plot_comp_vs_rmsd(rmsd_memmap, tcs_memmaps, axes, rmsd_name='RMSD',
                      tcs_names=None, fit_line=False):
    if tcs_names is None:
        tcs_names = ["TC" for x in tcs_memmaps]

    rmsds = np.memmap(rmsd_memmap, mode='r', dtype=np.double)
    df = pd.DataFrame()
    df[rmsd_name] = rmsds

    for i, (mmap, name) in enumerate(zip(tcs_memmaps, tcs_names)):
        ax = axes[i]
        tcs = np.memmap(mmap, mode='r', dtype=np.double)
        df[name] = tcs
        plot_tc_heatmap(df, ax=ax, outliers_df=None, cols=[rmsd_name, name],
                        ref_line=False, fit_line=fit_line, logscale=True,
                        set_auto_limits=True)
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels([str(i) for i in range(1, 8)])


if __name__ == "__main__":
    # Overview ROCS vs fingerprints figure
    fig = plt.figure(figsize=(5, 6.5))
    panel_axes = []
    panel_xoffsets = []

    # Enrichment curves
    ax = fig.add_subplot(321)
    cv_dirs = [ECFP4_CV_DIR, E3FP_CV_DIR, ROCS_SHAPE_CV_DIR, ROCS_COMBO_CV_DIR]
    cv_names = ["ECFP4", "E3FP", "ROCS-shape", "ROCS-combo"]
    cv_colors = [def_colors.ecfp_color, def_colors.e3fp_color,
                 def_colors.rocs_shape_color, def_colors.rocs_combo_color]
    ec_curves = []
    for i, dirname in enumerate(cv_dirs):
        best_ec_fold = get_best_fold_by_enrichment(dirname)
        ec = [enrichments_from_cv_dir(dirname, fold=best_ec_fold)[0]]
        ec_curves.append(ec)
    plot_enrichment_curves(ec_curves, ax=ax, names=cv_names, colors=cv_colors,
                           alpha=.8, y_min=EC_YMIN)
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.27)

    colors = [def_colors.rocs_shape_color, def_colors.e3fp_color]
    tc_counts_df = pd.DataFrame.from_csv(SHAPE_TC_COUNTS_FILE, sep="\t",
                                         index_col=None)
    tc_counts_df = remove_bad_pairs(tc_counts_df)

    # Shape TCs
    names = ["ROCS-shape", "E3FP"]
    ax = fig.add_subplot(323)
    plot_tc_heatmap(tc_counts_df, ax=ax, outliers_df=None,
                    cols=names, logscale=True, set_auto_limits=True)
    ax.set_aspect(1)
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.27)

    ax = fig.add_subplot(324)
    plot_tc_hists(tc_counts_df, cols=names, ax=ax,
                  colors=colors, logscale=True)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 30)  # prevent unsightly overlap with legend
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.23)

    # Combo TCs
    colors = [def_colors.rocs_combo_color, def_colors.e3fp_color]
    tc_counts_df = pd.DataFrame.from_csv(COMBO_TC_COUNTS_FILE, sep="\t",
                                         index_col=None)
    tc_counts_df = remove_bad_pairs(tc_counts_df)
    names = ["ROCS-combo", "E3FP"]
    ax = fig.add_subplot(325)
    plot_tc_heatmap(tc_counts_df, ax=ax, outliers_df=None,
                    cols=names, logscale=True, set_auto_limits=True)
    ax.set_aspect(2)
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.27)

    ax = fig.add_subplot(326)
    plot_tc_hists(tc_counts_df, cols=names, ax=ax,
                  colors=colors, logscale=True)
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.23)

    add_panel_labels(axes=panel_axes, xoffset=panel_xoffsets)
    fig.tight_layout(rect=[0.01, 0, 1, .97])
    fig.savefig(OUT_GLOBAL_BASENAME + ".png", dpi=300)
    fig.savefig(OUT_GLOBAL_BASENAME + ".tif", dpi=300)

    # Plot Comparisons vs RMSDS
    fig = plt.figure(figsize=(6.4, 3.1))
    rmsd_file = os.path.join(RANDOM_CONF_DIR, "rmsds.bin")
    comp_files = [
        os.path.join(RANDOM_CONF_DIR, x) for x in [
            'ecfp_tcs.bin', 'e3fp_tcs.bin', 'rocs_shape.bin',
            'rocs_combo.bin']]
    names = ['ECFP4 TC', 'E3FP Max TC', 'ROCS-shape', 'ROCS-combo']
    axes = [fig.add_subplot(2, 4, i + 1) for i in range(4)]
    plot_comp_vs_rmsd(rmsd_file, comp_files, axes, rmsd_name='RMSD',
                      tcs_names=names, fit_line=False)

    rmsd_file = os.path.join(RANDOM_PAIR_DIR, "rmsds.bin")
    comp_files = [
        os.path.join(RANDOM_PAIR_DIR, x) for x in [
            'tfds.bin', 'e3fp_tcs.bin', 'fastrocs_shape_tcs.bin',
            'fastrocs_combo_tcs.bin']]
    names = ['TFD', 'E3FP Max TC', 'ROCS-shape', 'ROCS-combo']
    axes = [fig.add_subplot(2, 4, i + 1) for i in range(4, 8)]
    plot_comp_vs_rmsd(rmsd_file, comp_files, axes, rmsd_name='RMSD',
                      tcs_names=names, fit_line=True)

    sns.despine(fig=fig, offset=1)
    add_panel_labels(fig=fig, xoffset=.38)
    fig.tight_layout(rect=[0.01, 0, 1, .97])
    fig.savefig(OUT_CONFS_BASENAME + ".png", dpi=300)
    fig.savefig(OUT_CONFS_BASENAME + ".tif", dpi=300)
