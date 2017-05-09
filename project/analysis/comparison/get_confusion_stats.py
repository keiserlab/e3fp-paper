"""Calculate performance stats for specific thresholds from confusion matrix.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import print_function, division
import logging
import os
import glob
import cPickle as pkl
import sys

import numpy as np
from scipy.sparse import issparse
from python_utilities.scripting import setup_logging
from python_utilities.io_tools import smart_open
from sklearn.metrics import confusion_matrix, precision_recall_curve


def get_max_f1_thresh(y_true, y_score):
    """Get maximum F1-score and corresponding threshold."""
    precision, recall, thresh = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall)
    max_f1_ind = np.argmax(f1)
    max_f1 = f1[max_f1_ind]
    max_f1_thresh = thresh[max_f1_ind]
    return max_f1, max_f1_thresh


def get_metrics_at_thresh(y_true, y_score, thresh):
    """Get sensitivity, specificity, precision and F1-score at threshold."""
    y_pred = y_score >= thresh
    confusion = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    sensitivity = tp / (tp + fn)  # recall
    specificity = tn / (fp + tn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    return sensitivity, specificity, precision, f1


def compute_fold_metrics(target_mol_array, mask_file, results_file,
                         thresh=None):
    """Compute metrics from fold at maximum F1-score or threshold."""
    logging.info("Loading mask file.")
    with smart_open(mask_file, "rb") as f:
        train_test_mask = pkl.load(f)
    test_mask = train_test_mask == 1
    del train_test_mask

    logging.info("Loading results from file.")
    with np.load(results_file) as data:
        results = data["results"]

    logging.info("Computing metrics.")
    y_true = target_mol_array[test_mask].ravel()
    y_score = results[test_mask].ravel()
    nan_inds = np.where(~np.isnan(y_score))
    y_true, y_score = y_true[nan_inds], y_score[nan_inds]
    del results, test_mask, target_mol_array

    if thresh is None:
        f1, thresh = get_max_f1_thresh(y_true, y_score)
    pvalue = 10**(-thresh)

    sensitivity, specificity, precision, f1 = get_metrics_at_thresh(y_true,
                                                                    y_score,
                                                                    thresh)
    logging.debug(("P-value: {:.4g}  Sensitivity: {:.4f}  "
                   "Specificity: {:.4f}  Precision: {:.4f}  "
                   "F1: {:.4f}").format(pvalue, sensitivity, specificity,
                                        precision, f1))

    return (pvalue, sensitivity, specificity, precision, f1)


def compute_average_metrics(cv_dir, thresh=None):
    """Compute fold metrics averaged across fold."""
    input_file = os.path.join(cv_dir, "inputs.pkl.bz2")
    fold_dirs = glob.glob(os.path.join(cv_dir, "*/"))

    logging.debug("Loading input files.")
    with smart_open(input_file, "rb") as f:
        (fp_array, mol_to_fp_inds, target_mol_array,
         target_list, mol_list) = pkl.load(f)
    del fp_array, mol_to_fp_inds, target_list, mol_list

    if issparse(target_mol_array):
        target_mol_array = target_mol_array.toarray().astype(np.bool)

    fold_metrics = []
    for fold_dir in sorted(fold_dirs):
        mask_file = glob.glob(os.path.join(fold_dir, "*mask*"))[0]
        results_file = glob.glob(os.path.join(fold_dir, "*result*"))[0]
        fold_metric = compute_fold_metrics(target_mol_array, mask_file,
                                           results_file, thresh=thresh)
        fold_metrics.append(fold_metric)

    fold_metrics = np.asarray(fold_metrics)
    mean_metrics = fold_metrics.mean(axis=0)
    std_metrics = fold_metrics.std(axis=0)
    logging.debug(("P-value: {:.4g} +/- {:.4g}  "
                   "Sensitivity: {:.4f} +/- {:.4f}  "
                   "Specificity: {:.4f} +/- {:.4f}  "
                   "Precision: {:.4f}  +/- {:.4f}  "
                   "F1: {:.4f} +/- {:.4f}").format(
                        mean_metrics[0], std_metrics[0],
                        mean_metrics[1], std_metrics[1],
                        mean_metrics[2], std_metrics[2],
                        mean_metrics[3], std_metrics[3],
                        mean_metrics[4], std_metrics[4]))
    return mean_metrics


if __name__ == "__main__":
    try:
        e3fp_cv_dir, ecfp_cv_dir = sys.argv[1:3]
    except IndexError:
        sys.exit("Usage: python get_confusion_stats.py <e3fp_cv_dir> "
                 "<ecfp_cv_dir>")

    setup_logging(verbose=True)

    logging.info("Getting average metrics for E3FP")
    metrics = compute_average_metrics(e3fp_cv_dir)
    max_thresh = -np.log10(metrics[0])
    logging.info("Getting average metrics for E3FP at p-value: {:.4g}".format(
        metrics[0]))
    compute_average_metrics(e3fp_cv_dir, thresh=max_thresh)
    logging.info("Getting average metrics for ECFP4 at p-value: {:.4g}".format(
        metrics[0]))
    compute_average_metrics(ecfp_cv_dir, thresh=max_thresh)

    logging.info("Getting average metrics for ECFP4")
    metrics = compute_average_metrics(ecfp_cv_dir)
    max_thresh = -np.log10(metrics[0])
    logging.info("Getting average metrics for ECFP4 at p-value: {:.4g}".format(
        metrics[0]))
    compute_average_metrics(ecfp_cv_dir, thresh=max_thresh)
