"""Concatenate binary lower triangular matrix data into numpy array.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import array
import sys

import numpy as np
from python_utilities.io_tools import smart_open
from get_triangle_indices import get_batch_size


def start_end_indices_from_fn(fn):
    """Parse filenames to get inclusive start and end indices."""
    start_index = int(fn.split("start-")[1].split("_")[0])
    end_index = int(fn.split("end-")[1].split(".")[0])
    return start_index, end_index


def main(bin_files, mol_name_files, np_file):
    data_list = []
    max_index = 0
    bin_files = sorted(bin_files, key=start_end_indices_from_fn)
    mol_name_files = sorted(mol_name_files, key=start_end_indices_from_fn)
    for bin_file, mol_name_file in zip(bin_files, mol_name_files):
        start_ind, end_ind = start_end_indices_from_fn(bin_file)
        expect_size = get_batch_size(start_ind, end_ind)
        max_index = max(max_index, end_ind)
        print("Reading from {}".format(bin_file))
        with smart_open(bin_file, "rb") as f:
            this_data_list = array.array('d', f.read()).tolist()
            diff = len(this_data_list) - expect_size
            if diff > 0:
                if diff == end_ind:  # handle old +1 error
                    print("Trimming redundant row from file.")
                    this_data_list = this_data_list[:-diff]
                else:
                    sys.exit(
                        "Unexpected size difference in file: {}.".format(diff))
            data_list.extend(this_data_list)

    mol_names_list = []
    print("Reading mol names from {}".format(mol_name_file))
    with smart_open(mol_name_file, "rb") as f:
        for line in f.readlines():
            mol_names_list.append(line.rstrip())
    if len(mol_names_list) != max_index + 1:
        sys.exit("Number of mol names {} doesn't match data {}.".format(
            len(mol_names_list), max_index + 1))

    total_expect_size = get_batch_size(0, max_index)
    if len(data_list) != total_expect_size:
        sys.exit("Total entry number {} deviates from expected {}.".format(
            len(data_list), total_expect_size))

    print("Converting to array and saving.")
    data = np.array(data_list, dtype=np.double)
    np.savez_compressed(np_file, data, np.asarray(mol_names_list))


if __name__ == "__main__":
    usage = ("python binary_to_numpy.py <binary_matrix_files> "
             "<mol_name_files> <numpy_array_file>")
    try:
        in_files = sys.argv[1:-1]
        bin_files = in_files[:len(in_files) / 2]
        mol_name_files = in_files[-len(in_files) / 2:]
        np_file = sys.argv[-1]
    except:
        sys.exit(usage)
    if len(bin_files) < 1 or len(mol_name_files) != len(bin_files):
        sys.exit(usage)
    main(bin_files, mol_name_files, np_file)
