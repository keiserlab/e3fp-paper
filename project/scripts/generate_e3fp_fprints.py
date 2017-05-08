"""Generate E3FP fingerprints and save to SEA molecules file.

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
import glob
import os
import logging
import argparse

from python_utilities.parallel import Parallelizer, make_data_iterator, \
                                      ALL_PARALLEL_MODES
from python_utilities.scripting import setup_logging
from e3fp.conformer.util import smiles_to_dict
from e3fp.pipeline import params_to_dicts
from e3fp_paper.sea_utils.util import lists_dicts_to_molecules, \
                                      fprint_params_to_fptype
from e3fp_paper.pipeline import native_tuples_from_sdf, \
                                native_tuples_from_smiles


def name_from_sdf_filename(fn):
    return os.path.basename(fn).split('.')[0]


def main(smiles_file, params_file, sdf_dir=None, out_file="molecules.csv.bz2",
         log=None, num_proc=None, parallel_mode=None, verbose=False):
    """Fingerprint molecules."""
    setup_logging(log, verbose=verbose)
    parallelizer = Parallelizer(parallel_mode="processes")

    # set conformer generation and fingerprinting parameters
    confgen_params, fprint_params = params_to_dicts(params_file)
    kwargs = {"save": False, "fprint_params": fprint_params}

    smiles_dict = smiles_to_dict(smiles_file)
    mol_num = len({x.split('-')[0] for x in smiles_dict})

    if sdf_dir is not None:
        sdf_files = glob.glob(os.path.join(sdf_dir, "*.sdf*"))
        sdf_files = sorted([x for x in sdf_files
                            if name_from_sdf_filename(x) in smiles_dict])
        data_iter = make_data_iterator(sdf_files)
        fp_method = native_tuples_from_sdf
        logging.info("Using SDF files from {}".format(sdf_dir))
    else:
        kwargs["confgen_params"] = confgen_params
        data_iter = ((smiles, name) for name, smiles
                     in smiles_dict.iteritems())
        mol_num = len({x.split('-')[0] for x in smiles_dict})
        fp_method = native_tuples_from_smiles
        logging.info("Will generate conformers.")
        logging.info(
            "Conformer generation params: {!r}.".format(confgen_params))
    logging.info(
        "Fingerprinting params: {!r}.".format(fprint_params))

    # fingerprint in parallel
    logging.info("Fingerprinting {:d} molecules".format(mol_num))
    mol_list_dict = {}
    for result, data in parallelizer.run_gen(fp_method, data_iter,
                                             kwargs=kwargs):
        if not result:
            logging.warning("Fingerprinting failed for {}.".format(data[0]))
            continue
        try:
            _, name = result[0]
            name = name.split('_')[0]
        except IndexError:
            logging.warning("Fingerprinting failed for {}.".format(data[0]))
            continue
        mol_list_dict[name] = result
    logging.info("Finished fingerprinting molecules")

    # save to SEA molecules file
    logging.info("Saving fingerprints to {}".format(out_file))
    fp_type = fprint_params_to_fptype(**fprint_params)
    lists_dicts_to_molecules(out_file, smiles_dict, mol_list_dict, fp_type)
    logging.info("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Generate E3FP fingerprints from SMILES.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('smiles_file', type=str,
                        help="""Path to SMILES file for fingerprinting.""")
    parser.add_argument('params_file', type=str,
                        help="""INI formatted file with parameters. If
                             '--sdf_dir' is provided, conformer generation
                             parameters are ignored.""")
    parser.add_argument('--sdf_dir', type=str, default=None,
                        help="""Directory containing SDF files for conformers
                             to be used.""")
    parser.add_argument('-o', '--out_file', type=str,
                        default="molecules.csv.bz2",
                        help="""SEA-format output molecules file.""")
    parser.add_argument('-l', '--log', type=str, default=None,
                        help="Log filename.")
    parser.add_argument('-p', '--num_proc', type=int, default=None,
                        help="""Set number of processors to use.""")
    parser.add_argument('--parallel_mode', type=str, default=None,
                        choices=list(ALL_PARALLEL_MODES),
                        help="""Set parallelization mode to use.""")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Run with extra verbosity.")
    params = parser.parse_args()

    kwargs = dict(params._get_kwargs())
    smiles_file = kwargs.pop('smiles_file')
    params_file = kwargs.pop('params_file')
    main(smiles_file, params_file, **kwargs)
