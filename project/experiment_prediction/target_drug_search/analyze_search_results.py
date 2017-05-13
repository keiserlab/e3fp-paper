"""Read 2D and 3D SEA search results, filter, and write to file.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import cPickle as pickle
import logging

from python_utilities.scripting import setup_logging
from python_utilities.io_tools import smart_open

ECFP_MIN_PVALUE = .1
E3FP_MAX_PVALUE = 1e-25
E3FP_MIN_AFFINITY = 10
ECFP_TC_CUTOFF = 0.3
OUTFILE = "unique_hits_aff{:d}_3dpval{:.4g}_2dpval{:.4g}.txt".format(
    E3FP_MIN_AFFINITY, E3FP_MAX_PVALUE, ECFP_MIN_PVALUE)


def reformat_mol_results(mol_results_dict):
    new_mol_results = {}
    for mol_name, hit_dict in mol_results_dict.iteritems():
        for target_key, result_tuple in hit_dict.iteritems():
            new_mol_results.setdefault(
                mol_name, {}).setdefault(
                    target_key.tid, {})[int(target_key.group)] = result_tuple
    return new_mol_results


if __name__ == "__main__":
    setup_logging()

    logging.info("Loading and reformatting E3FP results.")
    e3fp_mol_results = reformat_mol_results(
        pickle.load(smart_open("e3fp/mol_results.pkl.bz2", "rb")))
    logging.info("Loading and reformatting ECFP4 results.")
    ecfp4_mol_results = reformat_mol_results(
        pickle.load(smart_open("ecfp4/mol_results.pkl.bz2", "rb")))

    logging.info("Getting valid mol/target pairs.")
    e3fp_unique_mol_results = {}
    for mol_name, hit_dict in e3fp_mol_results.iteritems():
        for tid, affinity_results in hit_dict.iteritems():
            if (mol_name not in ecfp4_mol_results or
                    tid not in ecfp4_mol_results[mol_name] or
                    ecfp4_mol_results[mol_name][tid][max(
                        ecfp4_mol_results[mol_name][tid].keys())][0] > ECFP_MIN_PVALUE):
                for affinity, results_tuple in sorted(
                        affinity_results.items()):
                    if affinity > E3FP_MIN_AFFINITY:
                        break
                    try:
                        ecfp_results = ecfp4_mol_results[mol_name][tid][affinity]
                    except KeyError:
                        ecfp_results = (1.0, ECFP_TC_CUTOFF)
                    if results_tuple[0] <= E3FP_MAX_PVALUE:
                        e3fp_unique_mol_results[(
                            mol_name, tid,
                            affinity)] = results_tuple + ecfp_results
    del e3fp_mol_results
    del ecfp4_mol_results

    logging.info("Sorting and writing results to {}.".format(OUTFILE))
    with open(OUTFILE, "w") as f:
        f.write("mol_name\ttid\tgroup\tpvalue\ttc\n")
        for (mol_name, tid, affinity), (pvalue, tc, ecfp_pvalue,
                                        ecfp_tc) in sorted(
                                            e3fp_unique_mol_results.items(),
                                            key=lambda x: (x[1][0], x[0][2],
                                                           x[0][1], x[0][0])):
            f.write("{}\t{}\t{:d}\t{:.4g}\t{:.4f}\t{:.4g}\t{:.4f}\n".format(
                mol_name, tid, affinity, pvalue, tc, ecfp_pvalue, ecfp_tc))
