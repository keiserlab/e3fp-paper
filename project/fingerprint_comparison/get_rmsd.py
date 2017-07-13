"""Save pairwise RMSD between conformers.

Authors: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import sys
import struct
import itertools
import logging
import argparse

import numpy as np
import rdkit.Chem
from rdkit.Chem import rdMolAlign, rdFMCS
from python_utilities.scripting import setup_logging
from python_utilities.io_tools import smart_open
from python_utilities.parallel import Parallelizer, ALL_PARALLEL_MODES
from get_triangle_indices import get_batch_size, get_triangle_indices

SAVE_FREQ = 100  # min number of mol pairs between saves


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


def get_rmsd(mol1, mol2):
    mols = [mol1, mol2]
    mcs = rdFMCS.FindMCS(mols)
    match = rdkit.Chem.MolFromSmarts(mcs.smartsString)
    match1 = mol1.GetSubstructMatch(match)
    match2 = mol2.GetSubstructMatch(match)
    atom_map = zip(match1, match2)
    if len(atom_map) == 0:
        return np.nan
    return rdMolAlign.GetAlignmentTransform(mol1, mol2, atomMap=atom_map)[0]


def run_batch(start_index, end_index, sdf_file=None, save_freq=SAVE_FREQ,
              overwrite=False):
    base_output_name_strings = ['start-{0}'.format(start_index),
                                'end-{0}'.format(end_index)]
    rmsds_file = (
        '_'.join(['rmsds'] + base_output_name_strings)) + ".bin"
    mol_names_file = (
        '_'.join(['rmsd_mol_names'] + base_output_name_strings) + ".csv")

    batch_size = get_batch_size(start_index, end_index)
    logging.info(
        "Will save {} rmsds to {} and mol names to {}".format(
            batch_size, rmsds_file, mol_names_file))

    total_pairs_searched = 0
    last_save_ind = -1

    # Remove files or resume
    if overwrite:
        logging.info("Removing old files.")
        safe_unlink(rmsds_file)
        safe_unlink(mol_names_file)
    elif all_exist(rmsds_file, mol_names_file):
        logging.info("Resuming from existing files.")
        existing_index = get_num_mols(mol_names_file)
        last_save_ind = existing_index - 1
        total_pairs_searched = get_batch_size(start_index, existing_index - 1)
        start_index = existing_index
        logging.info(
            "Found {0} mol names. Resuming from index {0}.".format(
                existing_index))
    elif any_exist(rmsds_file, mol_names_file):
        sys.exit("Not all files exist, so cannot resume from old run.")

    supp = rdkit.Chem.SDMolSupplier(sdf_file)

    rmsds_tril = [[]]
    search_mol_names = []
    search_mols = []
    pairs_since_last_save = 0
    for index, mol in enumerate(supp):
        name = mol.GetProp("_Name")

        if search_mol_names and index >= start_index:
            rmsds = [get_rmsd(mol, x) for x in search_mols]
            rmsds_tril.append(rmsds)
            pairs_since_last_save += len(search_mol_names)

        # Add mol to search mols
        search_mol_names.append(name)
        search_mols.append(mol)

        # Cache results to file
        if (search_mol_names and index >= start_index and
                (pairs_since_last_save >= save_freq or
                 (end_index and index >= end_index))):
            total_pairs_searched += pairs_since_last_save
            perc_complete = total_pairs_searched / float(batch_size)
            logging.info(("{} molecules recorded. Appending rmsds to {} "
                          "and mol names to {}. ({:.4f}%)"
                          ).format(len(search_mol_names), rmsds_file,
                                   mol_names_file, 100 * perc_complete))
            cache_tcs_to_binary(rmsds_file, rmsds_tril)
            cache_mol_names(mol_names_file,
                            search_mol_names[last_save_ind + 1:])
            pairs_since_last_save = 0
            last_save_ind = index

        if end_index and index >= end_index:
            logging.info(
                "Ending at index {0} as requested ({1})".format(index,
                                                                end_index))
            return 0
    return 0


def main(sdf_file, save_freq=SAVE_FREQ, overwrite=False,
         log=None, parallel_mode=None, num_proc=None):
    setup_logging(log)
    logging.info("Reading mols from SDF.")
    supp = rdkit.Chem.SDMolSupplier(sdf_file)
    num_mol = len(supp)
    del supp

    para = Parallelizer(parallel_mode=parallel_mode, num_proc=num_proc)
    start_end_indices = get_triangle_indices(num_mol,
                                             para.num_proc - 1)
    kwargs = {"sdf_file": sdf_file, "save_freq": save_freq,
              "overwrite": overwrite}
    para.run(run_batch, start_end_indices, kwargs=kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Save pairwise RMSD between conformers.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf_file', type=str,
                        help="""Path to SDF file containing mols.""")
    parser.add_argument('--save_freq', type=int, default=SAVE_FREQ,
                        help="""Minimum number of pairs checked between
                             saves.""")
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
    main(params.sdf_file, save_freq=params.save_freq,
         overwrite=params.overwrite, log=params.log, num_proc=params.num_proc,
         parallel_mode=params.parallel_mode)
