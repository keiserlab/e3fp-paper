"""Bin and count pairwise TCs from two NumPy memmaps.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import logging
import csv
import argparse
from collections import Counter
try:
    from itertools import izip as zip
except ImportError:  # Python 3
    pass

import numpy as np
from python_utilities.scripting import setup_logging
from python_utilities.parallel import Parallelizer, ALL_PARALLEL_MODES
from python_utilities.io_tools import smart_open

LOG_FREQ = .001
MAX_CHUNK_SIZE = int(1e6)
PRECISION = 3
SEP = "\t"


def load_mmap(fn, log=True):
    if log:
        logging.info("Loading memmap file {}.".format(fn))
    mmap = np.memmap(fn, mode="r", dtype=np.double)
    if log:
        logging.info("Loaded {} pairs from file.".format(mmap.shape[0]))
    return mmap


def count_tcs(start_ind, end_ind, mfile1=None, mfile2=None,
              precision=PRECISION, log_freq=LOG_FREQ):
    if mfile1 is None or mfile2 is None:
        raise ValueError("memmap files are not valid.")
    mmap1 = load_mmap(mfile1, log=False)
    mmap2 = load_mmap(mfile2, log=False)

    pair_num = end_ind - start_ind + 1
    log_freq = int(log_freq * pair_num)
    tc_pair_counts = Counter()
    mult = 10**precision
    i = 0
    indices_since_last_log = 0
    while i < pair_num:
        chunk_start = i + start_ind
        chunk_range = (chunk_start,
                       min(chunk_start + MAX_CHUNK_SIZE, end_ind) + 1)
        chunk_size = chunk_range[1] - chunk_range[0]
        tcs_iter = zip(np.rint(mmap1[chunk_range[0]:chunk_range[1]] * mult),
                       np.rint(mmap2[chunk_range[0]:chunk_range[1]] * mult))

        tc_pair_counts.update(tcs_iter)
        i += chunk_size
        indices_since_last_log += chunk_size
        if indices_since_last_log >= log_freq or i >= pair_num:
            indices_since_last_log = 0
            logging.info("Binned {:d} of {:d} pairs ({:.1%})".format(
                i, pair_num, i / pair_num))
    return tc_pair_counts


def main(mfile1, mfile2, name1, name2, out_file, precision=PRECISION,
         log_freq=LOG_FREQ, num_proc=None, parallel_mode=None):
    setup_logging()
    if not out_file:
        out_file = (name1.lower().replace('\s', '_') + "_" +
                    name2.lower().replace('\s', '_') + "_tcs.csv.gz")

    # Load files
    mmap1 = load_mmap(mfile1)
    mmap2 = load_mmap(mfile2)
    if mmap1.shape != mmap2.shape:
        raise ValueError(
            "Memmaps do not have the same shape: {} {}".format(
                mmap1.shape, mmap2.shape))

    # Count binned pairs
    pair_num = mmap1.shape[0]
    del mmap1, mmap2

    para = Parallelizer(parallel_mode=parallel_mode, num_proc=num_proc)
    num_proc = max(para.num_proc - 1, 1)
    chunk_bounds = np.linspace(-1, pair_num - 1, num_proc + 1, dtype=int)
    chunk_bounds = list(zip(chunk_bounds[:-1] + 1, chunk_bounds[1:]))
    logging.info("Divided into {} chunks with ranges: {}".format(num_proc,
                                                                 chunk_bounds))

    logging.info("Counting TCs in chunks.")
    kwargs = {"mfile1": mfile1, "mfile2": mfile2, "precision": precision,
              "log_freq": log_freq}
    results_iter = para.run_gen(count_tcs, chunk_bounds, kwargs=kwargs)
    tc_pair_counts = Counter()
    for chunk_counts, _ in results_iter:
        if not isinstance(chunk_counts, dict):
            logging.error("Results are not in dict form.")
            continue
        tc_pair_counts.update(chunk_counts)

    # Write pairs to file
    logging.info("Writing binned pairs to {}.".format(out_file))
    mult = 10**precision
    with smart_open(out_file, "wb") as f:
        writer = csv.writer(f, delimiter=SEP)
        writer.writerow([name1, name2, "Count"])
        for pair in sorted(tc_pair_counts):
            writer.writerow([round(pair[0] / mult, precision),
                             round(pair[1] / mult, precision),
                             tc_pair_counts[pair]])

    total_counts = sum(tc_pair_counts.values())
    if total_counts != pair_num:
        logging.warning(
            "Pair counts {} did not match expected number {}".format(
                total_counts, pair_num))
        return
    logging.info("Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Bin and count pairwise TCs from two NumPy memmaps.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mmap_files', type=str, nargs=2,
                        help="""NumPy memmap files with pairwise TCs.""")
    parser.add_argument('--names', type=str, nargs=2, default=["1", "2"],
                        help="""Names for mmap datasets, used for column
                             names.""")
    parser.add_argument('-o', '--out_file', type=str, default=None,
                        help="""File in which to save binned pairs.""")
    parser.add_argument('-p', '--precision', type=int, default=PRECISION,
                        help="""Number of decimal points to use when
                             rounding.""")
    parser.add_argument('--num_proc', type=int, default=None,
                        help="""Set number of processors to use. Defaults to
                             all.""")
    parser.add_argument('--parallel_mode', type=str, default=None,
                        choices=list(ALL_PARALLEL_MODES),
                        help="""Set parallelization mode to use.""")
    params = parser.parse_args()
    args = params.mmap_files + params.names + [params.out_file]
    main(*args, precision=params.precision, num_proc=params.num_proc,
         parallel_mode=params.parallel_mode)
