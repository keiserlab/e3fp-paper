"""Make supplementary PDB enrichment figures.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging

from matplotlib import pyplot as plt
import seaborn as sns

from python_utilities.scripting import setup_logging
from e3fp_paper.plotting.defaults import DefaultColors
from e3fp_paper.plotting.util import add_panel_labels
from e3fp_paper.plotting.validation import plot_roc_curves, plot_prc_curves, \
    plot_enrichment_curves
from e3fp_paper.crossvalidation.util import rocs_from_cv_dir, \
    prcs_from_cv_dir, enrichments_from_cv_dir, results_from_fold_dir

sns.set_style("white")

def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
PDB_BASEDIR = os.path.join(PROJECT_DIR, 'pdb_enrichment')
ECFP_CHEMBL_PDB_DIR = os.path.join(PDB_BASEDIR, "ecfp4_chembl_pdb")
E3FP_CHEMBL_PDB_DIR = os.path.join(PDB_BASEDIR, "e3fp_chembl_pdb")
E3FP_RDKIT_CHEMBL_PDB_DIR = os.path.join(PDB_BASEDIR, "e3fp_rdkit_chembl_pdb")

ECFP_PDB_CV_DIR = os.path.join(PDB_BASEDIR, "ecfp4_cv_sea")
E3FP_PDB_CV_DIR = os.path.join(PDB_BASEDIR, "e3fp_cv_sea")
E3FP_RDKIT_PDB_CV_DIR = os.path.join(PDB_BASEDIR, "e3fp_rdkit_cv_sea")

PDB_FIG_BASENAME = "pdb_fig_sx"

ROC_YMIN = 0
FRAC_POS = .0432
FRAC_POS_CV = .0410
setup_logging(verbose=True)


def get_best_fold_by_prc(cv_dir):
    """Get number of best fold by fold AUPRC."""
    log_file = os.path.join(cv_dir, "cv_log.txt")
    logging.debug("Opening log file: {}".format(log_file))
    fold_results = []
    with open(log_file, "r") as f:
        for line in f:
            if "|Fold " in line and 'PRC' in line:
                fold = int(line.split('Fold ')[1].split()[0])
                auprc = float(line.split('AUPRC of ')[1].split()[0][:-1])
                fold_results.append((auprc, fold))
    return sorted(fold_results)[-1][1]


def get_best_fold_by_roc(cv_dir):
    """Get number of best fold by fold AUROC."""
    log_file = os.path.join(cv_dir, "cv_log.txt")
    logging.debug("Opening log file: {}".format(log_file))
    fold_results = []
    with open(log_file, "r") as f:
        for line in f:
            if "|Fold " in line and 'ROC' in line:
                fold = int(line.split('Fold ')[1].split()[0])
                auroc = float(line.split('AUROC of ')[1].split()[0])
                fold_results.append((auroc, fold))
    return sorted(fold_results)[-1][1]


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


if __name__ == "__main__":
    names = ["ECFP4", "E3FP-PDB", "E3FP-RDKit"]
    colors = [def_colors.ecfp_color, def_colors.e3fp_nostereo_color,
              def_colors.e3fp_color]
    fig = plt.figure(figsize=(6.4, 4.3))

    # CHEMBL vs PDB validation plots
    dirnames = [ECFP_CHEMBL_PDB_DIR, E3FP_CHEMBL_PDB_DIR,
                E3FP_RDKIT_CHEMBL_PDB_DIR]
    prc_list = []
    roc_list = []
    enrich_list = []
    for i, dirname in enumerate(dirnames):
        print(dirname)
        prc_list.append(
            [results_from_fold_dir(dirname, basename="combined_prc")[0]])
        roc_list.append(
            [results_from_fold_dir(dirname, basename="combined_roc")[0]])
        enrich_list.append(
            [results_from_fold_dir(dirname,
                                   basename="combined_enrichment")[0]])

    axes = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
    plot_prc_curves(prc_list, ref_val=FRAC_POS, ax=axes[0], names=names,
                    colors=colors, alpha=.8)
    plot_roc_curves(roc_list, ax=axes[1], names=names,
                    colors=colors, alpha=.8, y_min=ROC_YMIN)
    plot_enrichment_curves(enrich_list, ax=axes[2], names=names, colors=colors,
                           alpha=.8, y_min=ROC_YMIN)

    # PDB-only CV plots
    dirnames = [ECFP_PDB_CV_DIR, E3FP_PDB_CV_DIR, E3FP_RDKIT_PDB_CV_DIR]
    prc_list = []
    roc_list = []
    enrich_list = []
    for i, dirname in enumerate(dirnames):
        best_prc_fold = get_best_fold_by_prc(dirname)
        best_roc_fold = get_best_fold_by_roc(dirname)
        best_enrich_fold = get_best_fold_by_enrichment(dirname)

        prc_list.append([prcs_from_cv_dir(dirname, fold=best_prc_fold)[0]])
        roc_list.append([rocs_from_cv_dir(dirname, fold=best_roc_fold)[0]])
        enrich_list.append(
            [enrichments_from_cv_dir(dirname, fold=best_enrich_fold)[0]])

    axes = [fig.add_subplot(2, 3, 3 + i + 1) for i in range(3)]
    plot_prc_curves(prc_list, ref_val=FRAC_POS_CV, ax=axes[0], names=names,
                    colors=colors, alpha=.8)
    plot_roc_curves(roc_list, ax=axes[1], names=names,
                    colors=colors, alpha=.8, y_min=ROC_YMIN)
    plot_enrichment_curves(enrich_list, ax=axes[2], names=names, colors=colors,
                           alpha=.8, y_min=ROC_YMIN)

    sns.despine(fig=fig, offset=5)
    add_panel_labels(fig=fig, xoffset=.26)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(PDB_FIG_BASENAME + ".png", dpi=300)
    fig.savefig(PDB_FIG_BASENAME + ".tif", dpi=300)
