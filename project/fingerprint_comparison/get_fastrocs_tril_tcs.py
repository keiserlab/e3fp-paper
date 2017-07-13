#!/usr/local/fastrocs/bin/python2.7
"""Save pairwise max fastROCS shape and combo Tanimotos.

Output files are binary matrices corresponding to lower triangle of pairwise
max conformer-conformer Tanimotos for any molecule pair and a list of molecule
names in the same order.

Authors: Nick Mew, Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import sys
import struct
import itertools
import logging
import argparse

from openeye.oechem import *
from openeye.oefastrocs import *
from python_utilities.io_tools import smart_open
from python_utilities.scripting import setup_logging
from get_triangle_indices import get_batch_size

oepy = os.path.join(os.path.dirname(__file__), "openeye", "python")
sys.path.insert(0, os.path.realpath(oepy))

SAVE_FREQ = 100000  # min number of mol pairs between saves


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


def group_mols_by_name(mol_iter):
    """Create new mol object because OpenEye reuses mol object for each loop."""
    return ((k, list(g)) for k, g in
            itertools.groupby((OEMol(mol) for mol in mol_iter),
                              key=lambda x: x.GetTitle().split("-")[0]))


def main(mol_dbase, start_index, end_index, skip_inds=set(), skip_next=False,
         save_freq=SAVE_FREQ, overwrite=False, merge_confs=False,
         verbose=False, compress=False):
    base_output_name_strings = ['start-{0}'.format(start_index),
                                'end-{0}'.format(end_index)]
    if compress:
        binext = ".bin.gz"
        csvext = ".csv.gz"
    else:
        binext = ".bin"
        csvext = ".csv"

    log_file = (
        '_'.join(['fastrocs_log'] + base_output_name_strings) + ".txt")
    max_combo_tcs_file = (
        '_'.join(['fastrocs_max_combo_tcs'] +
                 base_output_name_strings)) + binext
    max_shape_tcs_file = (
        '_'.join(['fastrocs_max_shape_tcs'] +
                 base_output_name_strings)) + binext
    mol_names_file = ('_'.join(['mol_names'] + base_output_name_strings) +
                      csvext)
    if overwrite:
        safe_unlink(log_file)
    setup_logging(log_file, verbose=verbose)

    batch_size = get_batch_size(start_index, end_index)
    logging.info(
        "Will save {} max tcs to {} and {} and mol names to {}".format(
            batch_size, max_combo_tcs_file, max_shape_tcs_file,
            mol_names_file))

    total_pairs_searched = 0
    last_save_ind = -1

    # Remove files or resume
    if overwrite:
        logging.info("Removing old files.")
        safe_unlink(max_combo_tcs_file)
        safe_unlink(max_shape_tcs_file)
        safe_unlink(mol_names_file)
    elif all_exist(max_combo_tcs_file, max_shape_tcs_file, mol_names_file):
        logging.info("Resuming from existing files.")
        existing_index = get_num_mols(mol_names_file)
        last_save_ind = existing_index - 1
        total_pairs_searched = get_batch_size(start_index, existing_index - 1)
        start_index = existing_index
        logging.info(
            "Found {0} mol names. Resuming from index {0}.".format(
                existing_index))
    elif any_exist(max_combo_tcs_file, max_shape_tcs_file, mol_names_file):
        sys.exit("Not all files exist, so cannot resume from old run.")

    if skip_next:
        skip_inds.add(start_index)
    if len(skip_inds) > 1:
        logging.debug("Will skip indices: {}".format(skip_inds))

    ifs = oemolistream()
    if not ifs.open(mol_dbase):
        OEThrow.Fatal("Unable to open {} for reading".format(mol_dbase))
    if merge_confs:
        ifs.SetConfTest(OEAbsoluteConfTest())  # detect and merge conformers
        mols_iter = group_mols_by_name(ifs.GetOEMols())
    else:
        mols_iter = ((x.GetTitle(), [x]) for x in ifs.GetOEMols())

    # Configure OpenEye
    dbtype = OEShapeDatabaseType_Default
    options = OEShapeDatabaseOptions()
    options.SetScoreType(dbtype)
    search_db = OEShapeDatabase(dbtype)
    combo_tc_getter = OEShapeDatabaseScore.GetTanimotoCombo
    shape_tc_getter = OEShapeDatabaseScore.GetShapeTanimoto

    max_combo_tcs_tril = [[]]
    max_shape_tcs_tril = [[]]
    search_mol_names = []
    mol_idx_to_index = []
    pairs_since_last_save = 0
    dots = OEDots(save_freq, 20, "looping through molecule scores")
    for index, (mol_name, mols) in enumerate(mols_iter):
        logging.debug("Mol {} ({})".format(index, mol_name))

        mol_idx_to_index.extend([index] * len(mols))
        if search_mol_names and index >= start_index:
            if index not in skip_inds:
                logging.debug("Scoring mol {}".format(index))
                max_combo_tcs = [0.0] * len(search_mol_names)
                max_shape_tcs = [0.0] * len(search_mol_names)
                for conf in (c for mol in mols for c in mol.GetConfs()):
                    for score in search_db.GetScores(conf, options):
                        mol_idx = score.GetMolIdx()
                        mol_id = mol_idx_to_index[mol_idx]
                        max_combo_tcs[mol_id] = max(max_combo_tcs[mol_id],
                                                    combo_tc_getter(score))
                        max_shape_tcs[mol_id] = max(max_shape_tcs[mol_id],
                                                    shape_tc_getter(score))
                logging.debug("Finished scoring mol {}".format(index))
            else:
                logging.info("Skipping index {} ({}) as requested.".format(
                    index, mol_name))
                max_combo_tcs = [-1.0] * len(search_mol_names)
                max_shape_tcs = [-1.0] * len(search_mol_names)
            max_combo_tcs_tril.append(max_combo_tcs)
            max_shape_tcs_tril.append(max_shape_tcs)
            pairs_since_last_save += len(search_mol_names)

        # Add mol to search mols
        search_mol_names.append(mol_name)
        logging.debug("Adding mol {} to search db".format(index))
        for mol in mols:
            search_db.AddMol(mol)
        logging.debug("Finished adding mol {} to search db".format(index))
        dots.Update()

        # Cache results to file
        if (search_mol_names and (
                (index >= start_index and
                 (pairs_since_last_save >= save_freq or
                  end_index and index >= end_index)) or
                index == start_index)):
            total_pairs_searched += pairs_since_last_save
            perc_complete = total_pairs_searched / float(batch_size)
            logging.info(("{} molecules recorded. Appending shape tcs to {}, "
                          "combo tcs to {}, and mol names to {}. ({:.4%})"
                          ).format(
                              len(search_mol_names), max_shape_tcs_file,
                              max_combo_tcs_file, mol_names_file,
                              perc_complete))
            cache_tcs_to_binary(max_combo_tcs_file, max_combo_tcs_tril)
            cache_tcs_to_binary(max_shape_tcs_file, max_shape_tcs_tril)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Compute pairwise FastROCS max shape and combo Tanimoto coefficients
           between molecules.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('db_file', type=str,
                        help="""Path to SDF input file.""")
    parser.add_argument('start_index', type=int,
                        help="""Index to start.""")
    parser.add_argument('end_index', type=int,
                        help="""Index to end.""")
    parser.add_argument('--skip_inds', type=int, nargs="+", default=[],
                        help="""Indices to skip. Their rows will be set to
                             -1.""")
    parser.add_argument('--skip_next', action="store_true",
                        help="""Skip first index actually checked.""")
    parser.add_argument('--save_freq', type=int, default=SAVE_FREQ,
                        help="""Minimum number of pairs checked between
                             saves.""")
    parser.add_argument('--merge_confs', action="store_true",
                        help="""Merge adjacent conformers if they are the
                             same molecule.""")
    parser.add_argument('--compress', action="store_true",
                        help="""Save gzipped files.""")
    parser.add_argument('-O', '--overwrite', action="store_true",
                        help="""Overwrite existing file(s).""")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="""Run with increased verbosity.""")
    params = parser.parse_args()
    main(params.db_file, params.start_index, params.end_index,
         skip_inds=set(params.skip_inds), skip_next=params.skip_next,
         save_freq=params.save_freq, overwrite=params.overwrite,
         merge_confs=params.merge_confs, compress=params.compress,
         verbose=params.verbose)
