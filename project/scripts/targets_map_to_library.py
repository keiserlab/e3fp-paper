"""From targets map and fingerprints, build SEA library.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging
import argparse

import pandas as pd
from python_utilities.scripting import setup_logging
from python_utilities.io_tools import smart_open, touch_dir
from e3fp_paper.sea_utils.util import targets_to_dict, \
                                      molecules_to_lists_dicts, \
                                      filter_molecules_by_targets, \
                                      lists_dicts_to_molecules, \
                                      dict_to_targets, \
                                      targets_to_mol_lists_targets
from e3fp_paper.sea_utils.library import build_library
from e3fp_paper.crossvalidation.sample import sample_mol_lists_files

MAP_HEADER = ["family", "pdsp_name", "tid", "name", "organism"]
KEY_HEADER = "tid"
FIT_FILE = "library.fit"
AFFINITY = None
TMP_PREFIX = "temp"
SAMPLE_SIZE = 50000


def read_targets_map(targets_map_file, key_header=KEY_HEADER,
                     headers=MAP_HEADER):
    """Read targets map file into a dataframe."""
    targets_map = {}
    with smart_open(targets_map_file, "rU") as f:
        for i, line in enumerate(f):
            entries = line.rstrip('\r\n').split('\t')
            if len(entries) != len(headers):
                logging.error(
                    "Line {0} does not look as expected:".format(i + 1))
                logging.error(line)
            if i == 0:
                if entries != headers:
                    logging.warning("Header does not look as expected:")
                    logging.warning(line)
                continue
            entries_dict = dict(zip(headers, entries))
            key_value = entries_dict[key_header]
            targets_map[key_value] = entries_dict
    df = pd.DataFrame(targets_map).T
    df.set_index('name', inplace=True)
    return df


def filter_targets_by_map(targets_dict, targets_map):
    """Filter targets_dict to only targets with a name/tid in targets_map."""
    filtered_targets_dict = {}
    for target_key, set_value in targets_dict.iteritems():
        name = set_value.name
        tid = target_key.tid
        if name in targets_map.index or tid in targets_map.index:
            filtered_targets_dict[target_key] = set_value
    return filtered_targets_dict


def library_from_map(targets_map_file, all_molecules_file, all_targets_file,
                     fit_file=None, sample=None, affinity=None, out_dir='./'):
    """Build SEA library from target map and existing SEA molecules/targets."""
    molecules_file = os.path.join(out_dir, "molecules.csv.bz2")
    targets_file = os.path.join(out_dir, "targets.csv.bz2")
    library_file = os.path.join(out_dir, "library.sea")
    touch_dir(out_dir)

    logging.info("Reading targets map from {0}".format(targets_map_file))
    targets_map = read_targets_map(targets_map_file, key_header=KEY_HEADER,
                                   headers=MAP_HEADER)
    logging.debug("{:d} targets in map".format(len(targets_map)))
    logging.info("Reading targets file from {0}".format("all_targets_file"))
    all_targets_dict = targets_to_dict(all_targets_file, affinity=affinity)
    logging.debug("Read {:d} targets".format(len(all_targets_dict)))
    targets_dict = filter_targets_by_map(all_targets_dict, targets_map)
    logging.debug("{:d} targets after filtering".format(len(targets_dict)))
    logging.info("Reading molecules file from {0}".format(all_molecules_file))
    smiles_dict, all_mol_lists_dict, fp_type = molecules_to_lists_dicts(
        all_molecules_file)
    logging.debug("{:d} molecules in file".format(len(all_mol_lists_dict)))
    mol_lists_targets_dict = targets_to_mol_lists_targets(targets_dict,
                                                          all_mol_lists_dict)
    logging.debug("{:d} mol lists targets".format(
        len(mol_lists_targets_dict)))
    logging.info("Writing targets file")
    dict_to_targets(targets_file, mol_lists_targets_dict)
    mol_lists_dict = filter_molecules_by_targets(all_mol_lists_dict,
                                                 targets_dict)
    del targets_dict
    logging.debug("{:d} filtered molecules".format(len(mol_lists_dict)))
    del mol_lists_targets_dict
    logging.info("Writing molecules file")
    lists_dicts_to_molecules(molecules_file, smiles_dict,
                             mol_lists_dict, fp_type)

    if fit_file is None or not os.path.isfile(fit_file):
        logging.info("Fit file does not exist. Generating fit.")
        if fit_file is None:
            fit_file = os.path.join(out_dir, "library.fit")
        tmp_molecules_file = all_molecules_file
        tmp_targets_file = TMP_PREFIX + "_" + os.path.basename(targets_file)
        tmp_library_file = TMP_PREFIX + "_" + os.path.basename(library_file)
        if sample is not None:
            logging.info("Sampling {} random molecules for fit".format(sample))
            tmp_molecules_file = TMP_PREFIX + "_" + os.path.basename(
                molecules_file)
            sample_mol_lists_files(all_molecules_file, all_targets_file,
                                   sample,
                                   sample_molecules_file=tmp_molecules_file,
                                   sample_targets_file=tmp_targets_file,
                                   overwrite=True)
        else:
            logging.info("Using all molecules for fit")
            all_mol_lists_targets_dict = targets_to_mol_lists_targets(
                all_targets_dict, all_mol_lists_dict)
            logging.info("Writing all targets to file.")
            dict_to_targets(tmp_targets_file,
                            all_mol_lists_targets_dict)
            del all_mol_lists_targets_dict
        logging.info("Building library for fit molecules/targets.")
        build_library(tmp_library_file, tmp_molecules_file, tmp_targets_file,
                      fit_file, log=True, no_plot=False)
    else:
        logging.info("Fit file already exists. Skipping fit generation.")

    del all_mol_lists_dict

    logging.info("Building library")
    build_library(library_file, molecules_file, targets_file, fit_file,
                  log=True)
    logging.info("Library has been built.")


def main(targets_map_file, all_molecules_file, all_targets_file,
         fit_file=None, sample=None, affinity=None,
         log=None, out_dir='./', verbose=False):
    setup_logging(log, verbose=verbose)
    library_from_map(targets_map_file, all_molecules_file, all_targets_file,
                     fit_file=fit_file, sample=sample, affinity=affinity,
                     out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='From targets map and fingerprints, build SEA library.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('targets_map_file', type=str,
                        help='tab delimited file with target ids')
    parser.add_argument('all_targets_file', type=str,
                        help='SEA targets file')
    parser.add_argument('all_molecules_file', type=str,
                        help='SEA molecules file')
    parser.add_argument('-f', '--fit_file', type=str, default=None,
                        help=('fit file to add to library. If not provided, '
                              'fit will be generated'))
    parser.add_argument('--sample', type=int, default=None,
                        help=('number of random molecules to sample for fit '
                              'generation'))
    parser.add_argument('--affinity', type=str, default=None,
                        help='affinity level at which to filter targets')
    parser.add_argument('-o', '--out_dir', type=str, default='./',
                        help='directory to save output files')
    parser.add_argument('-l', '--log', type=str, default=None,
                        help='log file')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='run with increased verbosity')
    all_args = parser.parse_args()
    kwargs = dict(all_args._get_kwargs())

    main(**kwargs)
