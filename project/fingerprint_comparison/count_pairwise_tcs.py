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
from python_utilities.io_tools import smart_open

LOG_FREQ = .001
MAX_CHUNK_SIZE = int(1e6)
PRECISION = 3
SEP = "\t"


def load_mmap(fn):
    logging.info("Loading memmap file {}.".format(fn))
    mmap = np.memmap(fn, mode="r", dtype=np.double)
    logging.info("Loaded {} pairs from file.".format(mmap.shape[0]))
    return mmap


def main(mfile1, mfile2, name1, name2, out_file, precision=PRECISION,
         log_freq=LOG_FREQ):
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
    log_freq = int(log_freq * pair_num)
    tc_pair_counts = Counter()
    logging.info("Binning pairs.")
    mult = 10**precision
    i = 0
    indices_since_last_log = 0
    while i < pair_num:
        tcs_iter = zip(np.rint(mmap1[i:i + MAX_CHUNK_SIZE] * mult),
                       np.rint(mmap2[i:i + MAX_CHUNK_SIZE] * mult))

        tc_pair_counts.update(tcs_iter)
        i += MAX_CHUNK_SIZE
        indices_since_last_log += MAX_CHUNK_SIZE
        if indices_since_last_log >= log_freq:
            indices_since_last_log = 0
            logging.info("Binned {:d} of {:d} pairs ({:.1%})".format(
                i, pair_num, i / pair_num))

    # Write pairs to file
    logging.info("Writing binned pairs to {}.".format(out_file))
    with smart_open(out_file, "wb") as f:
        writer = csv.writer(f, delimiter=SEP)
        writer.writerow([name1, name2, "Count"])
        for pair in sorted(tc_pair_counts):
            writer.writerow([round(pair[0] / mult, precision),
                             round(pair[1] / mult, precision),
                             tc_pair_counts[pair]])
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
    params = parser.parse_args()
    args = params.mmap_files + params.names + [params.out_file]
    main(*args, precision=params.precision)
