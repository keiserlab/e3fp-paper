"""Get max Tanimoto coefficients between molecule fingerprints from two sets.

Authors: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import os
import logging
import argparse

import numpy as np
from python_utilities.io_tools import smart_open
from python_utilities.scripting import setup_logging
from python_utilities.parallel import Parallelizer, ALL_PARALLEL_MODES
from e3fp.fingerprint.metrics.array_metrics import tanimoto
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts
from e3fp_paper.crossvalidation.util import molecules_to_array

LOG_FREQ = 1000000  # min number of mol pairs between status messages


def run_batch(start_index, end_index,
              fp_array1=None, mol_names1=[], mol_indices_dict1={},
              fp_array2=None, mol_names2=[], mol_indices_dict2={},
              memmap_file=None):
    batch_size = (end_index - start_index + 1) * len(mol_names2)
    logging.info(
        "Will compare molecules from mol indices {} to {} ({} pairs)".format(
            start_index, end_index, batch_size))

    memmap = np.memmap(memmap_file, mode="r+", dtype=np.double,
                       shape=(len(mol_names1), len(mol_names2)))
    search_mol_start_inds = [
        mol_indices_dict2[i][0] for i in range(len(mol_names2))]

    pairs_since_last_log = 0
    total_pairs = 0
    for index, mol_name in enumerate(mol_names1):
        if index < start_index:
            continue

        fp_indices = mol_indices_dict1[index]

        # Batch compute max Tanimotos
        tcs = tanimoto(fp_array1[fp_indices], fp_array2)
        max_tcs = np.amax(tcs, axis=0)
        max_tcs = np.maximum.reduceat(max_tcs, search_mol_start_inds)
        memmap[index, :] = max_tcs
        pairs_since_last_log += max_tcs.shape[0]

        if pairs_since_last_log >= LOG_FREQ or index == end_index:
            total_pairs += pairs_since_last_log
            pairs_since_last_log = 0
            logging.info("{:.4%} completed.".format(total_pairs / batch_size))

        # Add mol to search mols
        if end_index and index >= end_index:
            logging.info("Ending at index {0} as requested ({1})".format(
                index, end_index))
            return 0
    return 0


def read_convert_mols(molecules_file):
    _, mol_list_dict, _ = molecules_to_lists_dicts(molecules_file)
    mol_names = sorted(mol_list_dict)
    fp_array, mol_indices_dict = molecules_to_array(mol_list_dict, mol_names)
    return fp_array, mol_names, mol_indices_dict


def get_start_end_inds(length, n):
    spaced_inds = np.linspace(0, length - 2, n, dtype=np.int)
    start_end_indices = zip(spaced_inds[:-1], spaced_inds[1:] + 1)
    return start_end_indices


def save_mol_names(fn, mol_names):
    with smart_open(fn, "w") as f:
        for mol_name in mol_names:
            f.write(mol_name + "\n")


def main(molecules_file1, molecules_file2, memmap_file, mol_names_file1,
         mol_names_file2, log=None, overwrite=False, parallel_mode=None,
         num_proc=None):
    setup_logging(log)
    logging.info("Reading first molecules file.")
    fp_array1, mol_names1, mol_indices_dict1 = read_convert_mols(
        molecules_file1)
    logging.info("Reading second molecules file.")
    fp_array2, mol_names2, mol_indices_dict2 = read_convert_mols(
        molecules_file2)

    if overwrite or not os.path.isfile(memmap_file):
        logging.info("Overwriting memmap file.")
        memmap = np.memmap(memmap_file, mode="w+", dtype=np.double,
                           shape=(len(mol_names1), len(mol_names2)))
        del memmap
        save_mol_names(mol_names_file1, mol_names1)
        save_mol_names(mol_names_file2, mol_names2)

    logging.info("Computing all pairwise Tanimotos.")

    para = Parallelizer(parallel_mode=parallel_mode, num_proc=num_proc)
    start_end_indices = get_start_end_inds(len(mol_names1), para.num_proc - 1)
    kwargs = {"fp_array1": fp_array1,
              "mol_names1": mol_names1,
              "mol_indices_dict1": mol_indices_dict1,
              "fp_array2": fp_array2,
              "mol_names2": mol_names2,
              "mol_indices_dict2": mol_indices_dict2,
              "memmap_file": memmap_file}
    para.run(run_batch, start_end_indices, kwargs=kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Compute pairwise max Tanimoto coefficients between molecules from
        one set and molecules from another.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('molecules_file1', type=str,
                        help="""Path to SEA-format molecules file 1.""")
    parser.add_argument('molecules_file2', type=str,
                        help="""Path to SEA-format molecules file 2.""")
    parser.add_argument('--memmap_file', type=str, default="max_tcs.dat",
                        help="""Path to NumPy memmap file with max TCs.""")
    parser.add_argument('--mol_names_file1', type=str,
                        default="mol_names1.csv.gz",
                        help="""Path to save first set of mol names.""")
    parser.add_argument('--mol_names_file2', type=str,
                        default="mol_names2.csv.gz",
                        help="""Path to save second set of mol names.""")
    parser.add_argument('-O', '--overwrite', action="store_true",
                        help="""Overwrite existing file(s).""")
    parser.add_argument('-l', '--log', type=str, default=None,
                        help="Log filename.")
    parser.add_argument('-p', '--num_proc', type=int, default=None,
                        help="""Set number of processors to use. Defaults to
                             all available.""")
    parser.add_argument('--parallel_mode', type=str, default=None,
                        choices=list(ALL_PARALLEL_MODES),
                        help="""Set parallelization mode to use.""")

    params = parser.parse_args()
    main(params.molecules_file1, params.molecules_file2, params.memmap_file,
         params.mol_names_file1, params.mol_names_file2, log=params.log,
         overwrite=params.overwrite, parallel_mode=params.parallel_mode,
         num_proc=params.num_proc)
