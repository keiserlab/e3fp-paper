"""Filtering and file generation methods for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import os
import glob
import re
import sys
import csv
import warnings
import logging
import itertools
import cPickle as pkl
import copy

import numpy as np
from scipy.sparse import issparse, lil_matrix, csr_matrix
# from sklearn import cross_validation as cv
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve, precision_recall_curve
try:
    from sklearn.exception import UndefinedMetricWarning
except ImportError:  # backwards compatibility with versions <0.17.2
    from sklearn.metrics.base import UndefinedMetricWarning
from seacore.util.library import SetValue
from python_utilities.io_tools import smart_open
import e3fp.fingerprint.fprint as fp
from e3fp.fingerprint.db import FingerprintDatabase as DB
from ..pipeline import native_tuple_to_fprint
from ..sea_utils.util import molecules_to_lists_dicts, \
                             targets_to_dict, \
                             native_tuple_to_indices

csv.field_size_limit(sys.maxsize)

OUT_CSV_EXT_DEF = ".csv.gz"
KT = 0.592  # kcal/mol


class InputProcessor(object):

    """Process fingerprint arrays before splitting."""

    def __init__(self, mode, energies_file=None):
        if mode not in ("union", "mean", "mean-boltzmann", "first"):
            raise ValueError("processing mode must be union, mean, "
                             "mean-boltzmann, or first.")
        self.mode = mode
        self.energies_file = energies_file
        if mode == "mean-boltzmann" and energies_file is None:
            raise ValueError("'energies_file' must be specified if mode "
                             "mean-boltzmann is provided.")

    def process_fingerprints(self, fprint_dict):
        new_fprint_dict = {}
        if self.mode == "union":
            for mol_name, fprints in fprint_dict.iteritems():
                new_fprint_dict[mol_name] = [
                    fp.Fingerprint.from_fingerprint(fp.add(fprints))]
                new_fprint_dict[mol_name][0].name = mol_name
        elif self.mode == "mean":
            for mol_name, fprints in fprint_dict.iteritems():
                new_fprint_dict[mol_name] = [fp.mean(fprints)]
                new_fprint_dict[mol_name][0].name = mol_name
        elif self.mode == "mean-boltzmann":
            energies_dict = {}
            with smart_open(self.energies_file, "r") as f:
                for line in f:
                    name, energy = line.rstrip().split('\t')
                    energies_dict[name] = float(energy)
            for mol_name, fprints in fprint_dict.iteritems():
                energies = np.array([energies_dict[fprint.name]
                                     for fprint in fprints])
                # factor out max term to reduce overflow
                e_min = energies.min()
                adjusted_energies = energies - e_min
                probs = np.exp(-adjusted_energies / KT)
                prob_sum = np.sum(probs)
                if prob_sum == 0.:
                    logging.warning(
                        ("Boltzmann probabilities for {} sum to 0. Using "
                         "unweighted mean.").format(mol_name))
                    new_fprint_dict[mol_name] = [fp.mean(fprints)]
                else:
                    if prob_sum == 1. and probs.shape[0] > 1:
                        logging.info(
                            ("Boltzmann probabilities for {} dominated by 1 "
                             "term.").format(mol_name))
                    new_fprint_dict[mol_name] = [fp.mean(fprints,
                                                         weights=probs)]
                new_fprint_dict[mol_name][0].name = mol_name
        elif self.mode == "first":
            new_fprint_dict = {}
            for mol_name, fprints in fprint_dict.iteritems():
                new_fprint_dict[mol_name] = []
                for proto_name, proto_fprints in itertools.groupby(
                        fprints, key=lambda x: x.name.split('_')[0]):
                    first_fprint = copy.deepcopy(list(proto_fprints)[0])
                    first_fprint.name = proto_name
                    new_fprint_dict[mol_name].append(first_fprint)
        return new_fprint_dict


def targets_to_array(targets, mol_list, dtype=np.int8, dense=False):
    if isinstance(targets, dict):
        targets_dict = targets
    else:
        targets_dict = targets_to_dict(targets)
    target_list = sorted(targets_dict.keys())
    target_num, mol_num = len(target_list), len(mol_list)

    mol_inds = {x: i for i, x in enumerate(mol_list)}
    target_mol_array = lil_matrix((target_num, mol_num), dtype=dtype)
    for i, target_key in enumerate(target_list):
        inds = [mol_inds[mol_name] for mol_name
                in targets_dict[target_key].cids]
        target_mol_array[i, inds] = True

    return target_mol_array.tocsr(), target_list


def molecules_to_array(molecules, mol_list, dense=False, processor=None):
    """Convert molecules to array or sparse matrix.

    Parameters
    ----------
    molecules : dict or string
        Molecules file or mol_list_dict.
    mol_list : list
        List of molecules, used to determine order of array.
    dense : bool, optional
        Return dense array.
    processor : InputProcessor, optional
        Object that processes fingerprints before building the database.

    Returns
    -------
    fp_array : ndarray or csr_matrix
        Row-based sparse matrix or ndarray containing fingerprints.
    mol_indices : dict
        Map from index in `mol_list` to list of row indices of fingerprints.
    """
    if isinstance(molecules, dict):
        mol_list_dict = molecules
    else:
        _, mol_list_dict, _ = molecules_to_lists_dicts(molecules)

    assert(set(mol_list_dict.keys()) == set(mol_list))

    fprint_dict = {k: [native_tuple_to_fprint(v) for v in vs]
                   for k, vs in mol_list_dict.iteritems()}
    del mol_list_dict

    try:
        fprint_dict = processor.process_fingerprints(fprint_dict)
    except AttributeError:
        pass

    mol_indices_dict = {}
    fprints_list = []
    max_ind = 0
    for k, mol_name in enumerate(mol_list):
        fprints = fprint_dict[mol_name]
        fp_num = len(fprints)
        row_inds = range(max_ind, max_ind + fp_num)
        mol_indices_dict[k] = row_inds
        max_ind += fp_num
        fprints_list += fprints

    db = DB(fp_type=fprints_list[0].__class__, level=fprints_list[0].level)

    db.add_fingerprints(fprints_list)

    all_fps = db.array
    if dense:
        all_fps = all_fps.toarray().astype(all_fps.dtype)

    return all_fps, mol_indices_dict


def save_cv_inputs(fn, fp_array, mol_to_fp_inds, target_mol_array,
                   target_list, mol_list):
    """Save inputs necessary for all CV folds."""
    with smart_open(fn, "wb") as f:
        pkl.dump((fp_array, mol_to_fp_inds, target_mol_array,
                  target_list, mol_list), f,
                 pkl.HIGHEST_PROTOCOL)


def load_cv_inputs(fn):
    """Load inputs necessary for all CV folds."""
    with smart_open(fn, "rb") as f:
        (fp_array, mol_to_fp_inds, target_mol_array,
         target_list, mol_list) = pkl.load(f)
    return (fp_array, mol_to_fp_inds, target_mol_array, target_list, mol_list)


def train_test_dicts_from_mask(mol_list_dict, mol_list, target_dict,
                               target_list, train_test_mask):
    test_inds = set(np.where(np.any(train_test_mask == 1, axis=0))[0])
    train_inds = set(np.where(np.any(train_test_mask == -1, axis=0))[0])
    test_mol_list_dict = {x: mol_list_dict[x] for i, x in enumerate(mol_list)
                          if i in test_inds}
    train_mol_list_dict = {x: mol_list_dict[x] for i, x in enumerate(mol_list)
                           if i in train_inds}

    test_target_dict = {}
    train_target_dict = {}
    for i, target_key in enumerate(target_list):
        test_mol_inds = np.where(train_test_mask[i, :] == 1)[0]
        test_mols = {mol_list[j] for j in test_mol_inds}
        sv = target_dict[target_key]
        pos_mols = set(sv.cids)
        test_target_dict[target_key] = SetValue(
            sv.name, sorted(pos_mols.intersection(test_mols)), sv.description)
        train_target_dict[target_key] = SetValue(
            sv.name, sorted(pos_mols.difference(test_mols)), sv.description)

    return (train_mol_list_dict, train_target_dict,
            test_mol_list_dict, test_target_dict)


def filter_targets_by_molnum(targets_dict, n):
    """Return targets that have at least `n` binders."""
    return dict([(k, v) for k, v in targets_dict.iteritems()
                 if len(v.cids) >= n])


def merge_dicts(*ds):
    """Given dicts with (key, value), merge to dict with (key, [values])."""
    merged = {}
    for d in ds:
        for k, v in d.iteritems():
            merged.setdefault(k, []).append(v)
    return merged


def average_dict_values(*ds):
    """Given dicts with (key, value), return dict with (key, mean_value)."""
    merged = merge_dicts(*ds)
    return {k: np.mean(v) for k, v in merged.iteritems()}


def auc_dict_from_fp_tp_dict(d):
    """Calculate AUC from dict with tuple of FP rate and TP rate"""
    return {k: get_auc(v[0], v[1]) for k, v in d.iteritems()}


def logauc_dict_from_fp_tp_dict(d):
    """Calculate logAUC from dict with tuple of FP rate and TP rate"""
    return {k: get_logauc(v[0], v[1]) for k, v in d.iteritems()}


def get_delta_auc_dict(d1, d2):
    """Given 2 AUC dicts, subtract the AUC of the second from the first."""
    return {k: (v - d2[k]) for k, v in d1.iteritems() if k in d2}


def get_roc_prc_auc(true_false, metrics):
    """Calculate ROC and precision-recall curves (PRC) and their AUCs.

    Parameters
    ----------
    true_false : ndarray of int
        Array of 1s and 0s for what should be true hits and false hits,
        respectively.
    metrics : ndarray of double
        Array of metrics corresponding to positive hits.

    Returns
    -------
    roc : 3xN array of double
        The first two rows of the array, when plotted against each other, form
        an ROC curve, and the third is the thresholds corresponding to each
        point
    auroc : float
        Area under the ROC curve
    prc : tuple of 2xN array of double and 1-D array of double
        The first value in the tuple is the array with points on the PR curve.
        The second value are thresholds which correspond to some of these
        points (see scikit-learn documentation).
    auroc : float
        Area under the precision-recall curve
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            fpr, tpr, roc_thresh = roc_curve(true_false, metrics,
                                             drop_intermediate=True)
        except UndefinedMetricWarning as e:
            raise
        auroc = get_auc(fpr, tpr)

        warnings.simplefilter("error")
        try:
            precision, recall, prc_thresh = precision_recall_curve(
                true_false, metrics)
        except RuntimeWarning as e:
            raise
        auprc = get_auc(recall, precision)

    return ((fpr, tpr, roc_thresh), auroc,
            (recall, precision, prc_thresh), auprc)


def enrichment_curve(y_true, y_score):
    """Calculate enrichment (cumulative-recall) curve.

    Enrichment curve is true positive rate (sensitivity/recall) vs ranking
    or percent of scores above a given threshold. The ratio of recall to
    ranked percentage is the enrichment factor.

    This method is based on scikit-learn's ROC curve implementation.

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_score : ndarray
        Scores.

    Returns
    -------
    dbp : ndarray
        Database percentage considered, where element i is the proportion of
        predicted positives with score >= thresholds[i].
    tpr : ndarray
        Increasing true positive rates where element i is the true positive
        rate of predictions with score >= thresholds[i].
    thresholds : ndarray
        Decreasing score values.
    """
    sorted_inds = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[sorted_inds]
    y_true = y_true[sorted_inds]

    unique_thresh_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[unique_thresh_inds, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_inds]

    tpr = tps / tps[-1]
    db_perc = (threshold_inds + 1) / y_true.size
    thresholds = y_score[threshold_inds]

    if len(db_perc) > 2:
        # Remove redundant points, for storage efficiency.
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(db_perc, 2),
                                                    np.diff(tpr, 2)),
                                      True])[0]
        db_perc = db_perc[optimal_idxs]
        tpr = tpr[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or db_perc[0] != 0:
        # Add "0" point
        tpr = np.r_[0, tpr]
        db_perc = np.r_[0, db_perc]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    return db_perc, tpr, thresholds


def get_auc(fp, tp, adjusted=False):
    """Calculate AUC from the FP and TP arrays of an ROC curve."""
    auc_val = sk_auc(fp, tp)
    if adjusted:
        auc_val -= 0.5
    return auc_val


def get_logroc(fp, tp, min_fp=0.001):
    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if (lam_index != 0):
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    return np.log10(x), y


def get_logauc(fp, tp, min_fp=0.001, adjusted=False):
    """Calculate logAUC, the AUC of the semilog ROC curve.

    `logAUC_lambda` is defined as the AUC of the ROC curve where the x-axis
    is in log space. In effect, this zooms the ROC curve onto the earlier
    portion of the curve where various classifiers will usually be
    differentiated. The adjusted logAUC is the logAUC minus the logAUC of
    a random classifier, resulting in positive values for better-than-random
    and negative otherwise.

    Reference:
        - Mysinger et al. J. Chem. Inf. Model. 2010, 50, 1561-1573.
    """
    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if (lam_index != 0):
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    dy = (y[1:] - y[:-1])
    with np.errstate(divide='ignore'):
        intercept = y[1:] - x[1:] * (dy / (x[1:] - x[:-1]))
        intercept[np.isinf(intercept)] = 0.
    norm = np.log10(1. / float(min_fp))
    areas = ((dy / np.log(10.)) + intercept * np.log10(x[1:] / x[:-1])) / norm
    logauc = np.sum(areas)
    if adjusted:
        logauc -= 0.144620062  # random curve logAUC
    return logauc


def get_youden(fp, tp):
    """Get Youden's index (height of ROC above random) for each point."""
    return tp - fp


def get_youden_index(fp, tp, return_coordinate=False):
    """Calculate Youden's J statistic from ROC curve.

    Youden's J statistic is defined as the maximum high of an ROC curve above
    the diagonal. Symbolically,
        J = max{TPR(FPR) - FPR}
    """
    youden = get_youden(fp, tp)
    index = np.argmax(youden)
    if return_coordinate:
        return youden[index], (fp[index], tp[index])
    else:
        return youden[index]


def results_from_fold_dir(fold_dir, basename="combined"):
    fns = glob.glob(os.path.join(fold_dir, "{}.*".format(basename)))
    results_list = []
    for fn in fns:
        logging.debug("Opening {}...".format(fn))
        with smart_open(fn, "rb") as f:
            results_list.append(pkl.load(f))
    return results_list


def results_from_cv_dir(cv_dir, fold=None, basename="combined"):
    if fold is None or not isinstance(fold, int):
        fold = "*/"
    fold_dirs = glob.glob(os.path.join(cv_dir, str(fold)))
    results_list = []
    for fold_dir in fold_dirs:
        results_list.extend(results_from_fold_dir(fold_dir, basename=basename))
    return results_list


def rocs_from_cv_dir(cv_dir, fold=None, basename="combined_roc"):
    return results_from_cv_dir(cv_dir, fold=fold, basename=basename)


def prcs_from_cv_dir(cv_dir, fold=None, basename="combined_prc"):
    return results_from_cv_dir(cv_dir, fold=fold, basename=basename)


def enrichments_from_cv_dir(cv_dir, fold=None,
                            basename="combined_enrichment"):
    return results_from_cv_dir(cv_dir, fold=fold, basename=basename)


def prc_roc_aucs_from_cv_dirs(cv_dirs):
    aucs_list = []
    for cv_dir in cv_dirs:
        log_file = glob.glob(os.path.join(cv_dir, "log.txt"))[0]
        with smart_open(log_file, "r") as f:
            for line in f:
                try:
                    m = re.search(
                        'Fold.*AUROC of (0\.\d+).*AUPRC of (0\.\d+)', line)
                    aucs = float(m.group(1)), float(m.group(2))
                    aucs_list.append(aucs)
                except AttributeError:
                    continue
    aurocs, auprcs = zip(*aucs_list)
    return aurocs, auprcs


def target_aucs_from_cv_dirs(cv_dirs):
    target_aurocs_dict = {}
    target_auprcs_dict = {}
    if isinstance(cv_dirs, str):
        cv_dirs = [cv_dirs]
    for cv_dir in cv_dirs:
        log_file = glob.glob(os.path.join(cv_dir, "log.txt"))[0]
        with smart_open(log_file, "r") as f:
            for line in f:
                try:
                    m = re.search(
                        'Target ([\w\d]+) .*AUROC of (0\.\d+).*AUPRC of (0\.\d+)', line)
                    tid = m.group(1)
                    aucs = float(m.group(2)), float(m.group(3))
                    target_aurocs_dict.setdefault(tid, [])
                    target_auprcs_dict.setdefault(tid, [])
                    target_aurocs_dict[tid].append(aucs[0])
                    target_auprcs_dict[tid].append(aucs[1])
                except AttributeError:
                    continue
    return target_aurocs_dict, target_auprcs_dict


def _make_cv_subdir(basedir, i):
    """Return cross-validation subdirectory."""
    return os.path.join(basedir, str(i))


def _make_cv_filename(out_dir, basename, group_type, i,
                      out_ext=OUT_CSV_EXT_DEF):
    """Return cross-validation filename for CSV file."""
    return os.path.join(_make_cv_subdir(out_dir, i),
                        "{!s}_{!s}_{:d}{!s}".format(basename, group_type, i,
                                                    out_ext))


def _make_csv_basename(filename):
    """Return basename of CSV file."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    if basename.endswith(".csv"):
        basename = os.path.splitext(basename)[0]
    return basename
