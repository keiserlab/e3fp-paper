"""Concatenate binary lower triangular matrix data into numpy memmap.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import sys
import logging

import numpy as np
from python_utilities.io_tools import smart_open
from python_utilities.scripting import setup_logging
from get_triangle_indices import get_batch_size


def start_end_indices_from_fn(fn):
    """Parse filenames to get inclusive start and end indices."""
    start_index = int(fn.split("start-")[1].split("_")[0])
    end_index = int(fn.split("end-")[1].split(".")[0])
    return start_index, end_index


def get_mol_name_file_from_bin_file(fn):
    path = os.path.dirname(fn)
    bn = os.path.basename(fn)
    bn = "mol_names" + bn.split("_tcs")[1]
    bn = bn.split(".bin")[0] + ".csv"
    fn = os.path.join(path, bn)
    if not os.path.isfile(fn):
        fn += ".gz"
    return fn


def main(bin_files, np_file, out_mol_names_file):
    setup_logging()
    max_index = 0
    bin_files = sorted(bin_files, key=start_end_indices_from_fn)
    mol_name_files = [get_mol_name_file_from_bin_file(x)
                      for x in bin_files]

    _, max_index = start_end_indices_from_fn(mol_name_files[-1])
    total_expect_size = get_batch_size(0, max_index)
    logging.info("Instantiating memmap of length {} at {}.".format(
        total_expect_size, np_file))
    memmap = np.memmap(np_file, mode="w+", dtype=np.double,
                       shape=(total_expect_size,))

    file_count = len(bin_files)
    data_count = 0
    for i, (bin_file, mol_name_file) in enumerate(zip(bin_files,
                                                      mol_name_files)):
        start_ind, end_ind = start_end_indices_from_fn(bin_file)
        expect_size = get_batch_size(start_ind, end_ind)
        logging.info("Reading from {} ({}/{})".format(bin_file, i + 1,
                                                      file_count))
        with smart_open(bin_file, "rb") as f:
            buff = f.read()
            data = np.frombuffer(buff, dtype=np.double)
            data_size = data.shape[0]
            diff = data_size - expect_size
            if diff > 0:
                if diff == end_ind:  # handle old +1 error
                    logging.info("Trimming redundant row from file.")
                    data = data[:-diff]
                    data_size = data.shape[0]
                else:
                    sys.exit(
                        "Unexpected size difference in file: {}.".format(diff))

            logging.info("Adding values to memmap.")
            memmap[data_count:data_count + data_size] = data
            memmap.flush()
            data_count += data_size
    del memmap

    if data_count != total_expect_size:
        sys.exit("Total entry number {} deviates from expected {}.".format(
            data_count, total_expect_size))

    mol_names_list = []
    logging.info("Reading mol names from {}.".format(mol_name_file))
    with smart_open(mol_name_file, "rb") as f:
        for line in f.readlines():
            mol_names_list.append(line.rstrip())
    if len(mol_names_list) != max_index + 1:
        sys.exit("Number of mol names {} doesn't match data {}.".format(
            len(mol_names_list), max_index + 1))

    logging.info("Saving mol names to {}.".format(out_mol_names_file))
    with smart_open(out_mol_names_file, "w") as f:
        f.write("\n".join(mol_names_list) + "\n")


if __name__ == "__main__":
    usage = ("python binary_to_numpy.py <binary_matrix_files> "
             "<numpy_array_file> <mol_names_file>")
    try:
        bin_files = sys.argv[1:-2]
        np_file = sys.argv[-2]
        mol_names_file = sys.argv[-1]
    except:
        sys.exit(usage)
    if len(bin_files) < 1:
        sys.exit(usage)
    main(bin_files, np_file, mol_names_file)
