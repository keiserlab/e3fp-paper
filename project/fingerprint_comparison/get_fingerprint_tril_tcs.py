"""Save pairwise max Tanimoto coefficients between molecule fingerprints.

Output files are a binary matrix corresponding to lower triangle of pairwise
max conformer-conformer Tanimotos for any molecule pair and a list of molecule
names in the same order.

Authors: Seth Axen, Nick Mew
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import sys
import struct
import itertools
import logging

import numpy as np
from python_utilities.parallel import Parallelizer
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts
from e3fp_paper.crossvalidation.util import molecules_to_array
from e3fp_paper.crossvalidation.methods import tanimoto_kernel
from get_triangle_indices import get_triangle_indices, get_batch_size

SAVE_FREQ = 1000000  # min number of mol pairs between saves


def cache_tcs_to_binary(fn, tcs_tril):
    with open(fn, 'ab') as f:
        tcs_list = list(itertools.chain(*tcs_tril))
        f.write(struct.pack('d' * len(tcs_list), *tcs_list))


def cache_mol_names(fn, mol_names):
    with open(fn, 'ab') as f:
        f.write('\n'.join(mol_names) + '\n')


def run_batch(start_index, end_index, fp_array=None, mol_names=[],
              mol_indices_dict={}):
    base_output_name_strings = ['start-{0}'.format(start_index),
                                'end-{0}'.format(end_index)]
    max_tcs_file = '_'.join(['max_tcs'] + base_output_name_strings) + '.bin'
    mol_names_file = ('_'.join(['mol_names'] + base_output_name_strings) +
                      '.csv')

    batch_size = get_batch_size(start_index, end_index)
    logging.info("Will save max tcs to {} and mol names to {}".format(
        max_tcs_file, mol_names_file))

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
    total_pairs_searched = 0
    pairs_since_last_save = 0
    last_save_ind = 0
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
            max_tcs_tril = [[]]
            pairs_since_last_save = 0
            last_save_ind = index

        if end_index and index >= end_index:
            logging.info("Ending at index {0} as requested ({1})".format(
                index, end_index))
            return 0
    return 0


def main(molecules_file):
    _, mol_list_dict, _ = molecules_to_lists_dicts(molecules_file)
    mol_names = sorted(mol_list_dict)
    fp_array, mol_indices_dict = molecules_to_array(mol_list_dict, mol_names)

    para = Parallelizer()
    start_end_indices = get_triangle_indices(len(mol_names), para.num_proc - 1)
    kwargs = {"fp_array": fp_array, "mol_names": mol_names,
              "mol_indices_dict": mol_indices_dict}
    para.run(run_batch, start_end_indices, kwargs=kwargs)


if __name__ == '__main__':
    molecules_file = sys.argv[1]
    main(molecules_file)
