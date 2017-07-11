"""Run cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging
import cPickle as pkl
import copy

import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, precision_recall_curve
from python_utilities.io_tools import smart_open, touch_dir
from python_utilities.parallel import Parallelizer
from ..sea_utils.util import targets_to_dict, molecules_to_lists_dicts, \
                             mol_lists_targets_to_targets, \
                             lists_dicts_to_molecules, dict_to_targets, \
                             targets_to_mol_lists_targets, \
                             filter_targets_by_molecules
from .util import molecules_to_array, filter_targets_by_molnum, \
                  targets_to_array, train_test_dicts_from_mask, \
                  save_cv_inputs, load_cv_inputs, get_roc_prc_auc, \
                  enrichment_curve
from .methods import SEASearchCVMethod

MASK_DTYPE = np.byte
TARGET_MOL_DENSE_DTYPE = np.byte
TARGET_MOL_SPARSE_DTYPE = np.bool


class MoleculeSplitter(object):

    """Split target x mol array based on molecules (columns)."""

    def __init__(self, k, reduce_negatives=False, random_state=None):
        self.k = k
        self.reduce_negatives = reduce_negatives
        self.random_state = random_state

    def get_train_test_masks(self, target_mol_arr):
        ntargets, nmols = target_mol_arr.shape
        if self.reduce_negatives and issparse(target_mol_arr):
            target_mol_arr = target_mol_arr.toarray().astype(
                TARGET_MOL_SPARSE_DTYPE)

        kfold = KFold(n_splits=self.k, shuffle=True,
                      random_state=self.random_state)
        masks = []
        for i, (_, test) in enumerate(kfold.split(np.arange(nmols),
                                                  np.arange(nmols))):
            mask = np.ones((ntargets, nmols), dtype=MASK_DTYPE) * -1
            mask[:, test] = 1
            if self.reduce_negatives:
                mask = self.reduce_negatives_from_mask(mask, target_mol_arr)
            masks.append(mask)
        return masks

    def reduce_negatives_from_mask(self, mask, target_mol_arr):
        test_mask = mask == 1
        train_mask = mask == -1

        num_before = np.sum((mask != 0) & (target_mol_arr == 0))
        test_neg_inds = np.where(
            ~np.any(test_mask & target_mol_arr, axis=0))[0]
        test_remove_arr = np.zeros_like(target_mol_arr, dtype=np.bool)
        test_remove_arr[:, test_neg_inds] = 1
        test_remove_arr[train_mask] = 0
        mask[test_remove_arr] = 0
        del test_neg_inds, test_remove_arr

        train_mask = mask == -1
        train_neg_inds = np.where(
            ~np.any(train_mask & target_mol_arr, axis=0))[0]
        train_remove_arr = np.zeros_like(target_mol_arr, dtype=np.bool)
        train_remove_arr[:, train_neg_inds] = 1
        train_remove_arr[test_mask] = 0
        mask[train_remove_arr] = 0
        del test_mask, train_mask, train_neg_inds, train_remove_arr

        num_after = np.sum((mask != 0) & (target_mol_arr == 0))
        if num_before == num_after:
            logging.info("No negatives were reduced.")
        else:
            logging.info("Number of negatives reduced from {} to {}.".format(
                num_before, num_after))
        return mask


class ByTargetMoleculeSplitter(MoleculeSplitter):

    """Split target by mol array based on targets (rows).

       Test/training sets are stratified to balance number of known binders
       per target in each fold.
    """
    def get_train_test_masks(self, target_mol_arr):
        ntargets, nmols = target_mol_arr.shape
        if self.reduce_negatives and issparse(target_mol_arr):
            target_mol_arr = target_mol_arr.toarray().astype(
                TARGET_MOL_SPARSE_DTYPE)

        kfold = StratifiedKFold(n_splits=self.k, shuffle=True,
                                random_state=self.random_state)

        masks = [np.ones((ntargets, nmols), dtype=MASK_DTYPE) * -1
                 for i in range(self.k)]
        for i in xrange(ntargets):
            y = target_mol_arr[i, :]
            if issparse(y):
                y = y.toarray().ravel()
            for j, (_, test) in enumerate(kfold.split(y, y)):
                masks[j][i, test] = 1

        if self.reduce_negatives:
            masks = [self.reduce_negatives_from_mask(mask, target_mol_arr)
                     for mask in masks]
        return masks


class KFoldCrossValidator(object):

    """Class to perform k-fold cross-validation."""

    def __init__(self, k=5, splitter=MoleculeSplitter,
                 cv_method=SEASearchCVMethod(), input_processor=None,
                 parallelizer=None, out_dir=os.getcwd(), overwrite=False,
                 return_auc_type="roc", reduce_negatives=False,
                 fold_kwargs={}):
        if isinstance(splitter, type):
            self.splitter = splitter(k)
        else:
            assert splitter.k == k
            self.splitter = splitter
        self.k = k
        if (cv_method is SEASearchCVMethod and
                input_processor is not None):
            raise ValueError(
                "Input processing is not (currently) compatible with SEA.")
        self.cv_method = cv_method
        self.input_processor = input_processor
        self.overwrite = overwrite
        if parallelizer is None:
            self.parallelizer = Parallelizer(parallel_mode="serial")
        else:
            self.parallelizer = parallelizer
        self.out_dir = out_dir
        touch_dir(out_dir)
        self.input_file = os.path.join(self.out_dir, "inputs.pkl.bz2")
        self.return_auc_type = return_auc_type.lower()
        self.reduce_negatives = reduce_negatives
        self.fold_kwargs = fold_kwargs

    def run(self, molecules_file, targets_file, min_mols=50, affinity=None,
            overwrite=False):
        fold_validators = {
            fold_num: FoldValidator(fold_num, self._get_fold_dir(fold_num),
                                    cv_method=copy.deepcopy(self.cv_method),
                                    input_file=self.input_file,
                                    overwrite=self.overwrite,
                                    **self.fold_kwargs)
            for fold_num in range(self.k)}
        if not os.path.isfile(self.input_file) or not all(
                [x.fold_files_exist() for x in fold_validators.values()]):
            logging.info("Loading and filtering input files.")
            ((smiles_dict, mol_list_dict, fp_type),
             target_dict) = self.load_input_files(molecules_file, targets_file,
                                                  min_mols=min_mols,
                                                  affinity=affinity)

            mol_list = sorted(mol_list_dict.keys())
            if isinstance(self.cv_method, SEASearchCVMethod):
                # efficiency hack
                fp_array, mol_to_fp_inds = (None, None)
            else:
                logging.info("Converting inputs to arrays.")
                fp_array, mol_to_fp_inds = molecules_to_array(
                    mol_list_dict, mol_list,
                    processor=self.input_processor)
            target_mol_array, target_list = targets_to_array(
                target_dict, mol_list, dtype=TARGET_MOL_DENSE_DTYPE)
            total_imbalance = get_imbalance(target_mol_array)

            if self.overwrite or not os.path.isfile(self.input_file):
                logging.info("Saving arrays and labels to files.")
                save_cv_inputs(self.input_file, fp_array, mol_to_fp_inds,
                               target_mol_array, target_list, mol_list)
            del fp_array, mol_to_fp_inds

            logging.info("Splitting data into {} folds using {}.".format(
                self.k, type(self.splitter).__name__))
            if self.splitter.reduce_negatives:
                logging.info("After splitting, negatives will be reduced.")
            train_test_masks = self.splitter.get_train_test_masks(
                target_mol_array)
            for fold_num, train_test_mask in enumerate(train_test_masks):
                logging.info(
                    "Saving inputs to files (fold {})".format(fold_num))
                fold_val = fold_validators[fold_num]
                fold_val.save_fold_files(train_test_mask, mol_list,
                                         target_list, smiles_dict,
                                         mol_list_dict, fp_type, target_dict)
            del (smiles_dict, mol_list_dict, fp_type, target_mol_array,
                 mol_list, train_test_masks, target_dict, target_list)
        else:
            logging.info("Resuming from input and fold files.")
            (fp_array, mol_to_fp_inds, target_mol_array,
             target_list, mol_list) = load_cv_inputs(self.input_file)
            total_imbalance = get_imbalance(target_mol_array)
            del (fp_array, mol_to_fp_inds, target_mol_array, target_list,
                 mol_list)

        # run cross-validation and gather scores
        logging.info("Running fold validation.")
        para_args = sorted(fold_validators.items())
        aucs = zip(*self.parallelizer.run(_run_fold, para_args))[0]
        if fold_validators.values()[0].compute_combined:
            aurocs, auprcs = zip(*aucs)
            mean_auroc = np.mean(aurocs)
            std_auroc = np.std(aurocs)
            logging.info("CV Mean AUROC: {:.4f} +/- {:.4f}".format(mean_auroc,
                                                                   std_auroc))
            mean_auprc = np.mean(auprcs)
            std_auprc = np.std(auprcs)
            logging.info(("CV Mean AUPRC: {:.4f} +/- {:.4f} ({:.4f} of data "
                          "is positive)").format(mean_auprc, std_auprc,
                                                 total_imbalance))
        else:
            (mean_auroc, mean_auprc) = (None, None)

        target_aucs = []
        for fold_val in fold_validators.values():
            with smart_open(fold_val.target_aucs_file, "rb") as f:
                target_aucs.extend(pkl.load(f).values())
        target_aucs = np.array(target_aucs)
        mean_target_auroc, mean_target_auprc = np.mean(target_aucs, axis=0)
        std_target_auroc, std_target_auprc = np.std(target_aucs, axis=0)
        logging.info("CV Mean Target AUROC: {:.4f} +/- {:.4f}".format(
            mean_target_auroc, std_target_auroc))
        logging.info("CV Mean Target AUPRC: {:.4f} +/- {:.4f}".format(
            mean_target_auprc, std_target_auprc))

        if "target" in self.return_auc_type or mean_auroc is None:
            logging.info("Returning target AUC.")
            aucs = (mean_target_auroc, mean_target_auprc)
        else:
            logging.info("Returning combined average AUC.")
            aucs = (mean_auroc, mean_auprc)

        if "pr" in self.return_auc_type:
            logging.info("Returned AUC is AUPRC.")
            return aucs[1]
        elif "sum" in self.return_auc_type:
            return sum(aucs)
        else:
            logging.info("Returned AUC is AUROC.")
            return aucs[0]

    def load_input_files(self, molecules_file, targets_file, min_mols=50,
                         affinity=None, overwrite=False):
        smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
            molecules_file)

        mol_names_target_dict = mol_lists_targets_to_targets(
            targets_to_dict(targets_file, affinity=affinity))
        target_dict = filter_targets_by_molecules(mol_names_target_dict,
                                                  mol_lists_dict)
        del mol_names_target_dict
        target_dict = filter_targets_by_molnum(target_dict, n=min_mols)

        return (smiles_dict, mol_lists_dict, fp_type), target_dict

    def _get_fold_dir(self, fold_num):
        return os.path.join(self.out_dir, str(fold_num))


class FoldValidator(object):

    """Class to perform validation on a single fold."""

    def __init__(self, fold_num, out_dir, cv_method=SEASearchCVMethod(),
                 input_file=os.path.join(os.getcwd(), "input.pkl.bz2"),
                 compute_combined=True, overwrite=False):
        self.fold_num = fold_num
        self.out_dir = out_dir
        touch_dir(self.out_dir)
        self.input_file = input_file
        self.mask_file = os.path.join(out_dir, "train_test_mask.pkl.bz2")
        self.results_file = os.path.join(out_dir, "results.npz")
        self.target_aucs_file = os.path.join(out_dir, "target_aucs.pkl.bz2")
        self.combined_roc_file = os.path.join(out_dir, "combined_roc.pkl.bz2")
        self.combined_prc_file = os.path.join(out_dir, "combined_prc.pkl.bz2")
        self.combined_enrichment_file = os.path.join(
            out_dir, "combined_enrichment.pkl.bz2")
        if isinstance(cv_method, type):
            cv_method = cv_method()
        cv_method.out_dir = out_dir
        cv_method.overwrite = overwrite
        self.cv_method = cv_method
        self.compute_combined = compute_combined
        self.overwrite = overwrite

    def run(self):
        logging.debug("Loading input files for fold. (fold {})".format(
            self.fold_num))
        (fp_array, mol_to_fp_inds, target_mol_array,
         target_list, mol_list) = load_cv_inputs(self.input_file)

        with smart_open(self.mask_file, "rb") as f:
            train_test_mask = pkl.load(f)
        test_mask = train_test_mask == 1
        train_mask = train_test_mask == -1
        del train_test_mask

        if issparse(target_mol_array):
            target_mol_array = target_mol_array.toarray().astype(
                TARGET_MOL_SPARSE_DTYPE)

        if self.overwrite or not os.path.isfile(self.results_file):
            train_mask = np.invert(test_mask)
            if not self.cv_method.is_trained(target_list):
                logging.info("Training models. (fold {})".format(
                    self.fold_num))
                self.cv_method.train(fp_array, mol_to_fp_inds,
                                     target_mol_array, target_list, mol_list,
                                     mask=train_mask)
            del train_mask

            logging.info("Testing models. (fold {})".format(self.fold_num))
            results = self.cv_method.test(fp_array, mol_to_fp_inds,
                                          target_mol_array, target_list,
                                          mol_list, mask=test_mask)
            np.savez_compressed(self.results_file, results=results)
        else:
            logging.info("Loading results from file. (fold {})".format(
                self.fold_num))
            with np.load(self.results_file) as data:
                results = data["results"]
        del mol_list, fp_array, mol_to_fp_inds

        logging.info(("Computing target ROC and PR curves and AUCs. "
                      "(fold {})").format(self.fold_num))
        target_aucs = {}
        imbalances = []
        for i, target_key in enumerate(target_list):
            test_inds = np.where(test_mask[i, :])[0]
            y_true = target_mol_array[i, test_inds]
            y_score = results[i, test_inds]
            try:
                roc, auroc, prc, auprc = get_roc_prc_auc(y_true, y_score)
            except:
                logging.exception(
                    "Could not builds curves for target {} (fold {})".format(
                        target_key.tid, self.fold_num))
                continue
            target_aucs[target_key] = (auroc, auprc)
            imbalance = get_imbalance(y_true)
            imbalances.append(imbalance)
            logging.info(("Target {} produced an AUROC of {:.4f} and an AUPRC"
                          " of {:.4f}. ({:.4f} of data is positive) "
                          "(fold {})").format(target_key.tid, auroc, auprc,
                                              imbalance, self.fold_num))
        del roc, prc, y_true, y_score, target_list
        with smart_open(self.target_aucs_file, "wb") as f:
            pkl.dump(target_aucs, f, pkl.HIGHEST_PROTOCOL)
        target_aucs = np.array(target_aucs.values())
        mean_target_auroc, mean_target_auprc = np.mean(target_aucs, axis=0)
        std_target_auroc, std_target_auprc = np.std(target_aucs, axis=0)
        mean_imbalance, std_imbalance = (np.mean(imbalances),
                                         np.std(imbalances))
        logging.info(("Targets produced an average AUROC of {:.4f} +/- {:.4f}"
                      " and AUPRC of {:.4f} +/- {:.4f}. ({:.4f} +/- "
                      "{:.4f} of data is positive) (fold {})").format(
                          mean_target_auroc, std_target_auroc,
                          mean_target_auprc, std_target_auprc,
                          mean_imbalance, std_imbalance, self.fold_num))
        del target_aucs, imbalances

        if self.compute_combined:
            logging.info(("Computing combined ROC and PR curves and AUCs. "
                          "(fold {})").format(self.fold_num))
            y_true = target_mol_array[test_mask].ravel()
            y_score = results[test_mask].ravel()
            nan_inds = np.where(~np.isnan(y_score))
            y_true, y_score = y_true[nan_inds], y_score[nan_inds]
            del results, test_mask, target_mol_array

            logging.debug("Computing combined ROC curve (fold {})".format(
                self.fold_num))
            roc = roc_curve(y_true, y_score, drop_intermediate=True)
            auroc = auc(roc[0], roc[1])
            with smart_open(self.combined_roc_file, "wb") as f:
                pkl.dump(roc, f, pkl.HIGHEST_PROTOCOL)
            del roc

            logging.debug("Computing combined PR curve (fold {})".format(
                self.fold_num))
            precision, recall, prc_thresh = precision_recall_curve(
                y_true, y_score)
            prc = (recall, precision, prc_thresh)
            auprc = auc(prc[0], prc[1])
            with smart_open(self.combined_prc_file, "wb") as f:
                pkl.dump(prc, f, pkl.HIGHEST_PROTOCOL)
            del prc

            logging.debug(
                "Computing combined enrichment curve (fold {})".format(
                    self.fold_num))
            enrichment = enrichment_curve(y_true, y_score)
            auec = auc(enrichment[0], enrichment[1])
            with smart_open(self.combined_enrichment_file, "wb") as f:
                pkl.dump(enrichment, f, pkl.HIGHEST_PROTOCOL)
            del enrichment

            imbalance = get_imbalance(y_true)
            logging.info(("Fold {} produced an AUROC of {:.4f} and an AUPRC"
                          " of {:.4f}. ({:.4f} of data is positive)").format(
                              self.fold_num, auroc, auprc, imbalance))
            logging.info(
                "Fold {} produced an enrichment AUC of {:.4f}.".format(
                    self.fold_num, auec))

            return (auroc, auprc)
        else:
            return None

    def save_fold_files(self, train_test_mask, mol_list, target_list,
                        smiles_dict, mol_list_dict, fp_type, target_dict):
        with smart_open(self.mask_file, "wb") as f:
            pkl.dump(train_test_mask, f, pkl.HIGHEST_PROTOCOL)

        if not isinstance(self.cv_method, SEASearchCVMethod):
            return

        test_molecules_file = os.path.join(self.out_dir,
                                           "test_molecules.csv.bz2")
        test_targets_file = os.path.join(self.out_dir, "test_targets.csv.bz2")
        train_molecules_file = os.path.join(self.out_dir,
                                            "train_molecules.csv.bz2")
        train_targets_file = os.path.join(self.out_dir,
                                          "train_targets.csv.bz2")

        if self.overwrite or not all(map(
                os.path.isfile, (test_molecules_file, test_targets_file,
                                 train_molecules_file, train_targets_file))):
            (train_mol_list_dict,
             train_target_dict,
             test_mol_list_dict,
             test_target_dict) = train_test_dicts_from_mask(
                mol_list_dict, mol_list, target_dict,
                target_list, train_test_mask)
            lists_dicts_to_molecules(test_molecules_file, smiles_dict,
                                     test_mol_list_dict, fp_type)
            lists_dicts_to_molecules(train_molecules_file, smiles_dict,
                                     train_mol_list_dict, fp_type)
            train_target_dict = targets_to_mol_lists_targets(
                train_target_dict, train_mol_list_dict)
            test_target_dict = targets_to_mol_lists_targets(
                test_target_dict, test_mol_list_dict)
            dict_to_targets(train_targets_file, train_target_dict)
            dict_to_targets(test_targets_file, test_target_dict)

    def fold_files_exist(self):
        if not os.path.isfile(self.mask_file):
            return False

        if isinstance(self.cv_method, SEASearchCVMethod):
            test_molecules_file = os.path.join(self.out_dir,
                                               "test_molecules.csv.bz2")
            test_targets_file = os.path.join(self.out_dir,
                                             "test_targets.csv.bz2")
            train_molecules_file = os.path.join(self.out_dir,
                                                "train_molecules.csv.bz2")
            train_targets_file = os.path.join(self.out_dir,
                                              "train_targets.csv.bz2")
            if not all(map(os.path.isfile, [test_molecules_file,
                                            test_targets_file,
                                            train_molecules_file,
                                            train_targets_file])):
                return False
        return True


def get_imbalance(pos_array):
    if issparse(pos_array):
        num_true = pos_array.nnz
        num_tot = pos_array.shape[0] * pos_array.shape[1]
    else:
        num_true = float(np.sum(pos_array))
        num_tot = pos_array.size
    imbalance = num_true / float(num_tot)
    return imbalance


def _run_fold(fold_num, fold_val):
    return fold_val.run()
