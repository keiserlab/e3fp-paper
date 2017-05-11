"""Search molecules against library.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import cPickle as pickle
import logging
import argparse

from python_utilities.scripting import setup_logging
from python_utilities.io_tools import smart_open
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts
from e3fp_paper.sea_utils.run import sea_set_search

TARGET_RESULTS_PICKLE_DEF = "target_results.pkl.bz2"
MOL_RESULTS_PICKLE_DEF = "mol_results.pkl.bz2"


def main(molecules_file, library_file,
         target_results_pickle=TARGET_RESULTS_PICKLE_DEF,
         mol_results_pickle=MOL_RESULTS_PICKLE_DEF, log_file=None,
         verbose=False):
    setup_logging(log_file, verbose=verbose)
    logging.info("Loading molecules file")
    smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
        molecules_file)
    del smiles_dict, fp_type
    logging.info("Running SEA searches with {:d} molecules.".format(
        len(mol_lists_dict)))
    set_searcher = sea_set_search(library_file, mol_lists_dict)
    logging.info("Saving results to pickles.")
    with smart_open(target_results_pickle, "wb") as f:
        pickle.dump(set_searcher.target_results_dict, f)
    with smart_open(mol_results_pickle, "wb") as f:
        pickle.dump(set_searcher.set_results_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search molecules against library.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('molecules_file', type=str,
                        help='SEA molecules file to search against library')
    parser.add_argument('library_file', type=str,
                        help='SEA library file to search against')
    parser.add_argument('--target_results_pickle', type=str,
                        default=TARGET_RESULTS_PICKLE_DEF,
                        help='Pickle file to write dict of target results')
    parser.add_argument('--mol_results_pickle', type=str,
                        default=MOL_RESULTS_PICKLE_DEF,
                        help='Pickle file to write dict of molecule results')
    parser.add_argument('-l', '--log_file', type=str, default=None,
                        help='log file')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='run with increased verbosity')
    all_args = parser.parse_args()
    kwargs = dict(all_args._get_kwargs())

    main(**kwargs)
