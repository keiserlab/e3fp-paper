"""Validate query molecules against target targets.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import argparse
import logging
import cPickle as pkl
import os

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from python_utilities.scripting import setup_logging
from python_utilities.io_tools import smart_open, touch_dir
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts, \
    targets_to_dict, mol_lists_targets_to_targets, \
    targets_to_mol_lists_targets, dict_to_targets, lists_dicts_to_molecules
from e3fp_paper.crossvalidation.util import molecules_to_array, \
    targets_to_array
from e3fp_paper.crossvalidation.methods import SEASearchCVMethod
from e3fp_paper.crossvalidation.util import enrichment_curve
from e3fp_paper.crossvalidation.run import get_imbalance


def filter_mols_by_targets(mol_list_dict, targets_dict):
    cids = set.union(*[set(v.cids) for v in targets_dict.values()])
    return {k: v for k, v in mol_list_dict.items() if k in cids}


def process_input_files(molecules_file, targets_file, sea_format=False):
    smiles_dict, mol_list_dict, fp_type = molecules_to_lists_dicts(
        molecules_file)
    targets_dict = targets_to_dict(targets_file)
    targets_dict = mol_lists_targets_to_targets(targets_dict)
    mol_list_dict = filter_mols_by_targets(mol_list_dict, targets_dict)
    mol_list = sorted(mol_list_dict)
    target_mol_array, target_list = targets_to_array(
        targets_dict, mol_list, dtype=np.byte)
    target_mol_array = target_mol_array.toarray().astype(np.bool)
    if sea_format:
        targets_dict = targets_to_mol_lists_targets(targets_dict,
                                                    mol_list_dict)
        return (target_mol_array, targets_dict, smiles_dict, mol_list_dict,
                fp_type, target_list, mol_list)
    else:
        fp_array, mol_to_fp_inds = molecules_to_array(mol_list_dict, mol_list)
        fp_array = fp_array.toarray().astype(np.bool)
        return (fp_array, mol_to_fp_inds, target_mol_array, target_list,
                mol_list)


def main(query_molecules_file, query_targets_file, target_molecules_file,
         target_targets_file, method=SEASearchCVMethod, fit_file=None,
         log=None, out_dir="./"):
    setup_logging(log)

    method = method()
    method.out_dir = out_dir
    touch_dir(out_dir)
    if fit_file is None:
        fit_file = os.path.join(out_dir, "library.fit")

    logging.info("Loading target files.")
    if isinstance(method, SEASearchCVMethod):
        method.fit_file = fit_file
        (_, target_targets_dict, target_smiles_dict, target_mol_list_dict,
         target_fp_type, target_target_list,
         target_mol_list) = process_input_files(
             target_molecules_file, target_targets_file, sea_format=True)

        logging.info("Saving target SEA files.")
        dict_to_targets(method.train_targets_file, target_targets_dict)
        lists_dicts_to_molecules(method.train_molecules_file,
                                 target_smiles_dict, target_mol_list_dict,
                                 target_fp_type)

        target_fp_array = None
        target_mol_to_fp_inds = None
        target_target_mol_array = None
        mask = None
    else:
        (target_fp_array, target_mol_to_fp_inds, target_target_mol_array,
         target_target_list, target_mol_list) = process_input_files(
            target_molecules_file, target_targets_file, sea_format=False)
        mask = np.ones_like(target_target_mol_array, dtype=np.bool_)

    method.train(target_fp_array, target_mol_to_fp_inds,
                 target_target_mol_array, target_target_list, target_mol_list,
                 mask=mask)

    logging.info("Loading query files.")
    if isinstance(method, SEASearchCVMethod):
        (query_target_mol_array, query_targets_dict, query_smiles_dict,
         query_mol_list_dict, query_fp_type, query_target_list,
         query_mol_list) = process_input_files(
             query_molecules_file, query_targets_file, sea_format=True)

        logging.info("Saving query SEA files.")
        lists_dicts_to_molecules(method.test_molecules_file,
                                 query_smiles_dict, query_mol_list_dict,
                                 query_fp_type)

        query_fp_array = None
        query_mol_to_fp_inds = None
    else:
        (query_fp_array, query_mol_to_fp_inds, query_target_mol_array,
         query_target_list, query_mol_list) = process_input_files(
            query_molecules_file, query_targets_file, sea_format=False)

    mask = np.ones_like(query_target_mol_array, dtype=np.bool_)
    results = method.test(query_fp_array, query_mol_to_fp_inds,
                          query_target_mol_array, query_target_list,
                          query_mol_list, mask=mask)

    y_true = query_target_mol_array.ravel()
    y_score = results.ravel()
    nan_inds = np.where(~np.isnan(y_score))
    y_true, y_score = y_true[nan_inds], y_score[nan_inds]

    logging.info("Computing results curves.")
    roc_file, prc_file, enrich_file = [
        os.path.join(out_dir, "combined_{}.pkl.bz2".format(x))
        for x in ["roc", "prc", "enrichment"]]

    logging.info("Computing ROC curves.")
    roc = roc_curve(y_true, y_score, drop_intermediate=True)
    auroc = auc(roc[0], roc[1])
    with smart_open(roc_file, "wb") as f:
        pkl.dump(roc, f, pkl.HIGHEST_PROTOCOL)
    logging.info("AUROC: {:.4f}".format(auroc))

    logging.info("Computing PRC curves.")
    prc_rec = precision_recall_curve(y_true, y_score)
    prc = (prc_rec[1], prc_rec[0], prc_rec[2])
    auprc = auc(prc[0], prc[1])
    imbalance = get_imbalance(y_true)
    with smart_open(prc_file, "wb") as f:
        pkl.dump(prc, f, pkl.HIGHEST_PROTOCOL)
    logging.info("AUPRC: {:.4f} ({:.4f} of data is positive)".format(
        auprc, imbalance))

    logging.info("Computing enrichment curves.")
    enrichment = enrichment_curve(y_true, y_score)
    with smart_open(enrich_file, "wb") as f:
        pkl.dump(enrichment, f, pkl.HIGHEST_PROTOCOL)
    auec = auc(enrichment[0], enrichment[1])
    logging.info("AUE: {:.4f}".format(auec))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Validate query molecules against target targets""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('query_molecules_file', type=str,
                        help="""Path to SEA-format molecules file for
                             query.""")
    parser.add_argument('query_targets_file', type=str,
                        help="""Path to SEA-format targets file for query.""")
    parser.add_argument('target_molecules_file', type=str,
                        help="""Path to SEA-format molecules file for
                             searching.""")
    parser.add_argument('target_targets_file', type=str,
                        help="""Path to SEA-format targets file for
                             searching.""")
    parser.add_argument('--fit_file', type=str, default=None,
                        help="""SEA fit-file to use for target library,
                             needed when '--method' is 'sea'.""")
    parser.add_argument('-l', '--log', type=str, default=None,
                        help="""Log file.""")
    parser.add_argument('-o', '--out_dir', type=str, default="./",
                        help="""Output directory.""")
    params = parser.parse_args()
    main(params.query_molecules_file, params.query_targets_file,
         params.target_molecules_file, params.target_targets_file,
         fit_file=params.fit_file, log=params.log, out_dir=params.out_dir)
