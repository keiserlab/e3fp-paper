"""Compute TCs between all molecule pairs with ECFP4 and E3FP.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import tempfile
import logging
import itertools
import random
import sys

import numpy as np
import pandas as pd

from seacore.run import sea_c

from python_utilities.io_tools import smart_open
from python_utilities.scripting import setup_logging
from python_utilities.parallel import Parallelizer

from e3fp_paper.sea_utils.util import molecules_to_lists_dicts


MOL_FRAC = 1.
MAX_EXAMPLES_PER_BIN = 10
MAX_CHUNK_SIZE = 1000000
CACHE_FREQ = .0005
ECFP4_SEA_TC_CUTOFF = .3
E3FP_SEA_TC_CUTOFF = .2


def molecules_to_fp_sets(molecules_file):
    _, mol_lists_dict, _ = molecules_to_lists_dicts(molecules_file)
    return {k: sea_c.build_set(zip(*v)[0])
            for k, v in mol_lists_dict.iteritems()}


def split_iterator_into_chunks(iterable, max_num):
    iterable = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterable, max_num))
        if not chunk:
            return
        yield chunk


def select_random_subset(df, n=MAX_EXAMPLES_PER_BIN):
    gb = df.groupby(df.columns.tolist(), as_index=True)
    df2 = pd.DataFrame.from_dict(
        {z: k for k, v in gb.groups.iteritems()
         for z in list(random.sample(v, min(len(v), n)))}, orient='index')
    df2.columns = df.columns
    df2.index.name = df.index.name
    return df2


def compute_tc_pairs(pairs, mol_names=[], fp_sets_dict1={}, fp_sets_dict2={},
                     col_prefixes=["ECFP4", "E3FP Max"],
                     precision=3):
    cols = [col_prefixes[0] + " TC", col_prefixes[1] + " TC"]

    fit1 = sea_c.build_fit(ECFP4_SEA_TC_CUTOFF, (0., 0., 0), (0., 0., 0))
    fit2 = sea_c.build_fit(E3FP_SEA_TC_CUTOFF, (0., 0., 0), (0., 0., 0))
    tcpair_counts = {}
    example_pairs = {}
    for i, pair in enumerate(pairs):
        mol_name1, mol_name2 = mol_names[pair[0]], mol_names[pair[1]]
        tc1 = sea_c.score(fp_sets_dict1[mol_name1],
                          fp_sets_dict1[mol_name2], fit1)[3]
        tc2 = sea_c.score(fp_sets_dict2[mol_name1],
                          fp_sets_dict2[mol_name2], fit2)[3]
        tc_pair = (round(tc1, precision), round(tc2, precision))
        try:
            tcpair_counts[tc_pair] += 1
        except KeyError:
            tcpair_counts[tc_pair] = 1

        if tcpair_counts[tc_pair] <= MAX_EXAMPLES_PER_BIN:
            example_pairs[(mol_name1, mol_name2)] = tc_pair

    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    counts_file = str(f.name)
    f.write("{}\t{}\tCount\n".format(*cols))
    for tcs, count in tcpair_counts.iteritems():
        f.write("{0:.{3}f}\t{1:.{3}f}\t{2:d}\n".format(
            tcs[0], tcs[1], count, precision))
    f.close()
    del tcpair_counts

    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    examples_file = str(f.name)
    f.write("Pair\t{}\t{}\n".format(*cols))
    for pair, tcs in example_pairs.iteritems():
        f.write("{0},{1}\t{2:.{4}f}\t{3:.{4}f}\n".format(
            pair[0], pair[1], tcs[0], tcs[1], precision))
    f.close()
    del example_pairs

    return counts_file, examples_file


if __name__ == "__main__":
    usage = "python compare_fingerprints.py <ecfp4_molecules> <e3fp_molecules>"
    try:
        ecfp_mol_file, e3fp_mol_file = sys.argv[1:]
    except ValueError:
        sys.exit(usage)

    setup_logging("log.txt")
    para = Parallelizer(parallel_mode="mpi")
    if para.rank == 0:
        logging.info("Reading molecules")
    ecfp_fp_sets = molecules_to_fp_sets(ecfp_mol_file)
    e3fp_fp_sets = molecules_to_fp_sets(e3fp_mol_file)

    mutual_mols = sorted(set(ecfp_fp_sets.keys()) & set(e3fp_fp_sets.keys()))
    mol_num = int(MOL_FRAC * len(mutual_mols))
    mols = mutual_mols[:mol_num]

    if para.rank == 0:
        logging.info(
            "Found total of {} mols. Selecting {} for comparison.".format(
                len(mutual_mols), mol_num))
    mols = sorted(np.random.choice(mutual_mols, size=mol_num, replace=False))
    pairs = ((i, j) for i in xrange(mol_num) for j in xrange(i + 1, mol_num))
    pair_groups_iter = split_iterator_into_chunks(pairs,
                                                  max_num=MAX_CHUNK_SIZE)
    pairs_iter = ((x,) for x in pair_groups_iter)
    kwargs = {"mol_names": mols, "fp_sets_dict1": ecfp_fp_sets,
              "fp_sets_dict2": e3fp_fp_sets,
              "col_prefixes": ["ECFP4", "E3FP Max"]}
    results_iter = para.run_gen(compute_tc_pairs, pairs_iter, kwargs=kwargs)

    if para.rank == 0:
        logging.info("Computing TC pairs.")
    counts_df = None
    examples_df = None
    pair_num_tot = mol_num * (mol_num - 1) / 2.
    pair_num_running = 0
    cache_point = int(CACHE_FREQ * pair_num_tot)
    i = 0
    for (counts_file, examples_file), _ in results_iter:
        pair_num_running += len(_[0])
        if para.rank == 0:
            logging.info("{:.2f}% completed".format(
                100 * pair_num_running / float(pair_num_tot)))
            with open(counts_file, "r") as f:
                df = pd.read_csv(f, sep="\t", index_col=(0, 1))
            if counts_df is None:
                counts_df = df
            else:
                counts_df = counts_df.add(df, fill_value=0)
            os.remove(counts_file)

            with open(examples_file, "r") as f:
                df = pd.read_csv(f, sep="\t", index_col=0)
            if examples_df is None:
                examples_df = df
            else:
                examples_df = examples_df.append(df)
                examples_df = select_random_subset(examples_df)

            os.remove(examples_file)
            i += 1
            if ( ((pair_num_running % cache_point) <
                  ((pair_num_running - MAX_CHUNK_SIZE) % cache_point)) or
                 (pair_num_running == pair_num_tot) ):
                counts_df["Count"] = counts_df["Count"].astype(int)
                counts_df.sort(inplace=True)
                counts_file = "ecfp_e3fp_tcs_counts.csv.bz2"
                with smart_open(counts_file, "w") as f:
                    counts_df.to_csv(f, sep="\t")
                logging.info("Cached counts file.")

                examples_file = "ecfp_e3fp_tcs_examples.csv.bz2"
                with smart_open(examples_file, "w") as f:
                    examples_df.to_csv(f, float_format="%.3f", sep="\t")
                logging.info("Cached examples file.")

    logging.info("Finished computing TC pairs. {}".format(para.rank))

    if para.rank == 0:
        logging.info("Saving counts file.")
        counts_df["Count"] = counts_df["Count"].astype(int)
        counts_df.sort(inplace=True)
        counts_file = "ecfp_e3fp_tcs_counts.csv.bz2"
        with smart_open(counts_file, "w") as f:
            counts_df.to_csv(f, sep="\t")
        del counts_df
        logging.info("Saved counts file.")

        logging.info("Saving examples file.")
        examples_file = "ecfp_e3fp_tcs_examples.csv.bz2"
        with smart_open(examples_file, "w") as f:
            examples_df.to_csv(f, float_format="%.3f", sep="\t")
        del examples_df
        logging.info("Saved examples file.")

        logging.info("Done!")
