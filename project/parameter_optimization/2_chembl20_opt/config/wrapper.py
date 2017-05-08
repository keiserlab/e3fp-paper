"""Cross-validation wrapper to be run with Spearmint.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import re
import logging
import csv

from python_utilities.scripting import setup_logging
from python_utilities.io_tools import touch_dir, smart_open
from python_utilities.parallel import Parallelizer, make_data_iterator
from e3fp.config.params import update_params, write_params
from e3fp.conformer.util import smiles_to_dict, MolItemName
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts, \
                                      lists_dicts_to_molecules, \
                                      targets_to_dict, dict_to_targets, \
                                      filter_targets_by_molecules, \
                                      targets_to_mol_lists_targets, \
                                      fprint_params_to_fptype
from e3fp_paper.pipeline import native_tuples_from_sdf
from e3fp_paper.crossvalidation.run import KFoldCrossValidator, \
                                           ByTargetMoleculeSplitter

PROJECT_DIR = os.environ("E3FP_PROJECT")
MAIN_DIR = os.path.join(PROJECT_DIR, "parameter_optimization")
CV_DIR = os.path.join(MAIN_DIR, "2_chembl20_opt")
MAIN_CONF_DIR = os.path.join(PROJECT_DIR, "conformer_generation")
SMILES_FILE = os.path.join(PROJECT_DIR, "data",
                           "chembl20_proto_smiles.smi.bz2")
MOL_TARGETS_FILE = os.path.join(PROJECT_DIR, "data",
                                "chembl20_binding_targets.csv.bz2")
TARGETS_BASENAME = "targets"
MOL_BASENAME = "molecules"
CSV_EXT = ".csv.bz2"
LOG_FILE = "log.txt"
NUM_PROC = None
AUC_TYPE = 'sum'
AFFINITY = 10000
STEREO = True
CV_K = 5
MIN_MOLS_PER_TARGET = 50
REDUCE_NEGATIVES = True


def unformat_params(params):
    """Format params as needed for Spearmint PB file."""
    return {k: [v, ] for k, v in params.iteritems()}


def format_params(params):
    """Clean up params dict."""
    new_params = {k: v[0] for k, v in params.iteritems()}
    new_params['level'] = int(new_params['level'])
    new_params['bits'] = int(new_params['bits'])
    new_params['first'] = int(new_params['first'])
    return new_params


def params_to_str(params, with_first=True):
    """Create a descriptive string built from params."""
    params_string = "e3fp_{!s}_rad{:.4g}_level{:d}_fold{:d}".format(
        params['conformers'], params['radius_multiplier'], params['level'],
        params['bits'])
    if with_first:
        params_string = "{!s}_first{:d}".format(params_string,
                                                params['first'])
    return params_string


def str_to_params(string):
    """Parse descriptive string to get params."""
    params = {}
    m = re.match("^e3fp_(.+)_rad([0-9\.]+)_level(\d+)_fold(\d+)(.*)", string)
    params['conformers'] = m.group(1)
    params['radius_multiplier'] = float(m.group(2))
    params['level'] = int(m.group(3))
    params['bits'] = int(m.group(4))
    try:
        params['first'] = int(m.group(5).replace('_first', ''))
    except ValueError:
        pass
    return params


def get_existing_fprints(params_string, needed_first, directory):
    """Check directory for fingerprints which can be reused."""
    earlier_results_dirs = [x[:-1] for x
                            in glob.glob("{!s}/*/".format(directory))]
    pre_encoding_match_dirs = [
        x for x in earlier_results_dirs
        if os.path.basename(x).startswith(params_string)]
    if len(pre_encoding_match_dirs) == 0:
        return None

    encoding_num = [(str_to_params(os.path.basename(x))['first'], x)
                    for x in pre_encoding_match_dirs]
    existing_dir_name = None
    for first, dir_name in sorted(encoding_num):
        if first >= needed_first:
            existing_dir_name = dir_name
            break
    if existing_dir_name is None:
        return None

    existing_fprints_file = get_molecules_file(os.path.join(
        directory, existing_dir_name))
    if os.path.isfile(existing_fprints_file):
        return existing_fprints_file
    else:
        return None


def params_to_molecules(params, smiles_file, conf_dir, out_dir,
                        parallelizer=None):
    """Generate molecules_file based on params dict."""
    smiles_dict = smiles_to_dict(smiles_file)
    logging.debug("SMILES file has {:d} unique smiles.".format(
        len(smiles_dict)))
    logging.debug("Example SMILES: {!r}".format(smiles_dict.items()[0]))
    fprint_params = {"radius_multiplier": params["radius_multiplier"],
                     "stereo": STEREO, "bits": params["bits"],
                     "first": params['first'], "level": params['level']}

    conf_dir_files = glob.glob("{!s}/*".format(conf_dir))
    logging.debug("Found {:d} files in conformer directory.".format(
        len(conf_dir_files)))
    sdf_files = [x for x in conf_dir_files
                 if os.path.basename(x).split('.')[0] in smiles_dict]
    logging.debug("{:d} conformer files match SMILES.".format(len(sdf_files)))

    if len(sdf_files) == 0:
        raise Exception("Directory {!s} does not contain any usable SDF "
                        "files.".format(conf_dir))

    kwargs = {"save": False, "fprint_params": fprint_params}

    data_iterator = make_data_iterator(sdf_files)
    if parallelizer is not None:
        results_iter = parallelizer.run_gen(native_tuples_from_sdf,
                                            data_iterator, kwargs=kwargs)
    else:
        results_iter = (native_tuples_from_sdf(*x, **kwargs)
                        for x in data_iterator)

    molecules_file = get_molecules_file(out_dir)
    fp_type = fprint_params_to_fptype(**params)
    with smart_open(molecules_file, "wb") as f:
        writer = csv.writer(f)
        fp_type.write(writer)
        writer.writerow(("molecule id", "smiles", "fingerprint"))
        for results in results_iter:
            try:
                fp_native_list, sdf_file = results
            except ValueError:
                logging.error("Results of fingerprinting did not look as "
                              "expected: {!r}".format(results))
            proto_name = MolItemName.from_str(fp_native_list[0][1]).proto_name
            smiles = smiles_dict[proto_name]
            for fp_native, fp_name in fp_native_list:
                writer.writerow((fp_name, smiles, fp_native))

    del smiles_dict
    filtered_smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
        molecules_file)
    return (filtered_smiles_dict, mol_lists_dict, fp_type)


def get_molecules_file(out_dir):
    """Get molecules filename."""
    return os.path.join(out_dir, MOL_BASENAME + CSV_EXT)


def get_targets_file(out_dir):
    """Get targets filename."""
    return os.path.join(out_dir, TARGETS_BASENAME + CSV_EXT)


def main(job_id, params, main_conf_dir=MAIN_CONF_DIR, main_dir=CV_DIR,
         out_dir=None, smiles_file=SMILES_FILE, check_existing=True,
         mol_targets_file=MOL_TARGETS_FILE, k=CV_K, log_file=LOG_FILE,
         verbose=False, overwrite=False, min_mols=MIN_MOLS_PER_TARGET,
         parallelizer=None):
    params = format_params(params)

    pre_encoding_params_string = params_to_str(params, with_first=False)
    params_string = params_to_str(params)
    if out_dir is None:
        out_dir = os.path.join(main_dir, params_string)
    touch_dir(out_dir)
    if log_file is not None:
        log_file = os.path.join(out_dir, log_file)
    setup_logging(log_file, verbose=verbose)

    params_file = os.path.join(out_dir, "params.cfg")
    config_parser = update_params(params, section_name="fingerprinting")
    write_params(config_parser, params_file)

    if not isinstance(parallelizer, Parallelizer):
        parallelizer = Parallelizer(parallel_mode="processes",
                                    num_proc=NUM_PROC)

    logging.info("Params: {!r}".format(params.items()))
    logging.info("Saving files to {:s}.".format(out_dir))

    logging.info("Checking for usable pre-existing fingerprints.")
    existing_molecules_file = get_existing_fprints(pre_encoding_params_string,
                                                   params['first'], main_dir)

    molecules_file = get_molecules_file(out_dir)
    if os.path.isfile(molecules_file) and not overwrite:
        logging.info("Molecules file already exists. Loading.")
        smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
            molecules_file)
    elif existing_molecules_file is None:
        conf_dir = os.path.join(main_conf_dir, params['conformers'])
        logging.info("Generating fingerprints from conformers in "
                     "{!s}.".format(conf_dir))
        smiles_dict, mol_lists_dict, fp_type = params_to_molecules(
            params, smiles_file, conf_dir, out_dir, parallelizer=parallelizer)
    else:
        logging.info("Using native strings from existing molecules "
                     "file {!s}.".format(existing_molecules_file))
        smiles_dict, mol_lists_dict, fp_type = molecules_to_lists_dicts(
            existing_molecules_file, first=params['first'])
        lists_dicts_to_molecules(get_molecules_file(out_dir),
                                 smiles_dict, mol_lists_dict, fp_type)

    targets_file = get_targets_file(out_dir)
    if overwrite or not os.path.isfile(targets_file):
        logging.info("Reading targets from {!s}.".format(mol_targets_file))
        targets_dict = targets_to_dict(mol_targets_file, affinity=AFFINITY)
        logging.debug("Read {:d} targets.".format(len(targets_dict)))
        logging.info("Filtering targets by molecules.")
        filtered_targets_dict = targets_to_mol_lists_targets(
            filter_targets_by_molecules(targets_dict, mol_lists_dict),
            mol_lists_dict)

        del targets_dict, smiles_dict, mol_lists_dict, fp_type
        logging.info("Saving filtered targets to {!s}.".format(targets_file))
        dict_to_targets(targets_file, filtered_targets_dict)
        del filtered_targets_dict
    else:
        logging.info("Targets file already exists. Skipping.")

    parallel_mode = parallelizer.parallel_mode
    parallelizer = Parallelizer(parallel_mode=parallel_mode, num_proc=k + 1)

    splitter = ByTargetMoleculeSplitter(k, reduce_negatives=REDUCE_NEGATIVES)
    kfold_cv = KFoldCrossValidator(k=k, parallelizer=parallelizer,
                                   splitter=splitter,
                                   return_auc_type=AUC_TYPE, out_dir=out_dir,
                                   overwrite=False)
    auc = kfold_cv.run(molecules_file, targets_file, min_mols=min_mols,
                       affinity=AFFINITY)
    logging.info("CV Mean AUC: {:.4f}".format(auc))
    return 1 - auc
