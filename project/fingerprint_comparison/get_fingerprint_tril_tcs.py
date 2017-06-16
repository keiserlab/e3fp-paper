"""Save pairwise max Tanimoto coefficients between molecule fingerprints.

Output files are a binary matrix corresponding to lower triangle of pairwise
max conformer-conformer Tanimotos for any molecule pair and a list of molecule
names in the same order.

Authors: Seth Axen, Nick Mew
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import os
import sys
import struct
import itertools
import logging
import argparse

import numpy as np
from python_utilities.io_tools import smart_open
from python_utilities.scripting import setup_logging
from python_utilities.parallel import Parallelizer, ALL_PARALLEL_MODES
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts
from e3fp_paper.crossvalidation.util import molecules_to_array
from e3fp_paper.crossvalidation.methods import tanimoto_kernel
from get_triangle_indices import get_triangle_indices, get_batch_size

SAVE_FREQ = 1000000  # min number of mol pairs between saves


def cache_tcs_to_binary(fn, tcs_tril):
    """Append new tcs to binary file and clear lower triangle."""
    with smart_open(fn, 'ab') as f:
        tcs_list = list(itertools.chain(*tcs_tril))
        f.write(struct.pack('d' * len(tcs_list), *tcs_list))
    tcs_tril[:] = [[]]


def cache_mol_names(fn, mol_names):
    """Append mol names to file."""
    with smart_open(fn, 'ab') as f:
        f.write('\n'.join(mol_names) + '\n')


def safe_unlink(fn):
    """Unlink file if exists."""
    try:
        os.remove(fn)
    except OSError:
        pass


def all_exist(*fns):
    return all([os.path.isfile(x) for x in fns])


def any_exist(*fns):
    return any([os.path.isfile(x) for x in fns])


def get_num_mols(mol_names_file):
    i = 0
    with smart_open(mol_names_file, 'rb') as f:
        for l in f:
            if len(l.rstrip()) > 0:
                i += 1
    return i


def run_batch(start_index, end_index, fp_array=None, mol_names=[],
              mol_indices_dict={}, overwrite=False):
    """Save pairwise TCs for specified region of lower triangle matrix."""
    base_output_name_strings = ['start-{0}'.format(start_index),
                                'end-{0}'.format(end_index)]
    max_tcs_file = '_'.join(['max_tcs'] + base_output_name_strings) + '.bin.gz'
    mol_names_file = ('_'.join(['mol_names'] + base_output_name_strings) +
                      '.csv.gz')

    batch_size = get_batch_size(start_index, end_index)
    logging.info("Will save max tcs to {} and mol names to {}".format(
        max_tcs_file, mol_names_file))

    total_pairs_searched = 0
    last_save_ind = -1

    # Remove files or resume
    if overwrite:
        logging.info("Removing old files.")
        safe_unlink(max_tcs_file)
        safe_unlink(mol_names_file)
    elif all_exist(max_tcs_file, mol_names_file):
        logging.info("Resuming from existing files.")
        existing_index = get_num_mols(mol_names_file)
        last_save_ind = existing_index - 1
        total_pairs_searched = get_batch_size(start_index, existing_index - 1)
        start_index = existing_index
        logging.info(
            "Found {0} mol names. Resuming from index {0}.".format(
                existing_index))
    elif any_exist(max_tcs_file, mol_names_file):
        sys.exit("Not all files exist, so cannot resume from old run.")

    max_tcs_tril = [[]]
    search_mol_names = mol_names[:start_index]
    search_fp_indices = [y for i in range(len(search_mol_names))
                         for y in mol_indices_dict[i]]
    search_mol_start_inds = [
        mol_indices_dict[i][0] for i in range(len(search_mol_names))]
    if search_mol_names:
        search_array = fp_array[search_fp_indices, :]
    else:
        search_array = None
    pairs_since_last_save = 0
    for index, mol_name in enumerate(mol_names):
        if index < start_index:
            continue

        fp_indices = mol_indices_dict[index]

        # Batch compute max Tanimotos
        if search_mol_names:
            tcs = tanimoto_kernel(fp_array[fp_indices], search_array)
            max_tcs = np.amax(tcs, axis=0)
            max_tcs = np.maximum.reduceat(
                max_tcs, search_mol_start_inds).tolist()
            max_tcs_tril.append(max_tcs)
            pairs_since_last_save += len(search_mol_names)

        # Add mol to search mols
        search_mol_names.append(mol_name)
        search_fp_indices.extend(fp_indices)
        search_mol_start_inds.append(fp_indices[0])
        search_array = fp_array[search_fp_indices, :]

        # Cache results to file
        if (search_mol_names and index >= start_index and
                (pairs_since_last_save >= SAVE_FREQ or
                 (end_index and index >= end_index))):
            total_pairs_searched += pairs_since_last_save
            perc_complete = total_pairs_searched / batch_size
            logging.info(("{} molecules recorded. Appending tcs to {} and "
                          "mol names to {} ({:.4%} complete)").format(
                              len(search_mol_names), max_tcs_file,
                              mol_names_file, perc_complete))
            cache_tcs_to_binary(max_tcs_file, max_tcs_tril)
            cache_mol_names(mol_names_file,
                            search_mol_names[last_save_ind + 1:])
            pairs_since_last_save = 0
            last_save_ind = index

        if end_index and index >= end_index:
            logging.info("Ending at index {0} as requested ({1})".format(
                index, end_index))
            return 0
    return 0


def main(molecules_file, log=None, overwrite=False, parallel_mode=None,
         num_proc=None):
    setup_logging(log)
    _, mol_list_dict, _ = molecules_to_lists_dicts(molecules_file)
    mol_names = sorted(mol_list_dict)
    fp_array, mol_indices_dict = molecules_to_array(mol_list_dict, mol_names)

    para = Parallelizer(parallel_mode=parallel_mode, num_proc=num_proc)
    start_end_indices = get_triangle_indices(len(mol_names), para.num_proc - 1)
    kwargs = {"fp_array": fp_array, "mol_names": mol_names,
              "mol_indices_dict": mol_indices_dict, "overwrite": overwrite}
    para.run(run_batch, start_end_indices, kwargs=kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Compute pairwise max Tanimoto coefficients between molecules.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('molecules_file', type=str,
                        help="""Path to SEA-format molecules file.""")
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
    main(params.molecules_file, log=params.log, overwrite=params.overwrite,
         parallel_mode=params.parallel_mode, num_proc=params.num_proc)
