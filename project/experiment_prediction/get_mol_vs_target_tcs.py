"""Get max Tanimoto coefficients between a molecule and target molecules.

Authors: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import argparse

import numpy as np
from e3fp.fingerprint.metrics.array_metrics import tanimoto
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts, \
    targets_to_dict, mol_lists_targets_to_targets
from e3fp_paper.crossvalidation.util import molecules_to_array


def main(mol_name, tid, molecules_file1, molecules_file2,
         targets_file, out_file="max_tcs.npz", affinity=None):
    _, mol_list_dict1, _ = molecules_to_lists_dicts(molecules_file1)
    mol_list_dict1 = {mol_name: mol_list_dict1[mol_name]}
    fp_array1, _ = molecules_to_array(mol_list_dict1, [mol_name])

    targets_dict = targets_to_dict(targets_file, affinity=affinity)
    targets_dict = {k: v for k, v in targets_dict.items() if k.tid == tid}
    targets_dict = mol_lists_targets_to_targets(targets_dict)
    cids = set.union(*[set(v.cids) for v in targets_dict.values()])
    _, mol_list_dict2, _ = molecules_to_lists_dicts(molecules_file2)
    mol_list_dict2 = {k: v for k, v in mol_list_dict2.items()
                      if k in cids}
    fp_array2, mol_inds2 = molecules_to_array(mol_list_dict2,
                                              mol_list_dict2.keys())
    mol_start_inds = [
        v[0] for k, v in sorted(mol_inds2.items(), key=lambda x: x[1][0])]
    tcs = tanimoto(fp_array1, fp_array2)
    mol_start_inds
    max_tcs = tcs.max(axis=0)
    max_tcs = np.maximum.reduceat(max_tcs, mol_start_inds)
    np.savez(out_file, max_tcs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Compute pairwise max Tanimoto coefficients between a molecule and a
        target's molecules.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mol_name', type=str, help="""Molecule name""")
    parser.add_argument('tid', type=str, help="""Target IDs""")
    parser.add_argument('molecules_file1', type=str,
                        help="""Path to SEA-format molecules file 1.""")
    parser.add_argument('molecules_file2', type=str,
                        help="""Path to SEA-format molecules file 2.""")
    parser.add_argument('targets_file', type=str,
                        help="""Targets file.""")
    parser.add_argument('--affinity', type=int, default=None,
                        help="""Target affinity group""")
    parser.add_argument('-o', '--out_file', type=str, default="max_tcs.npz",
                        help="""NumPy array with max tcs.""")

    params = parser.parse_args()
    main(params.mol_name, params.tid, params.molecules_file1,
         params.molecules_file2, params.targets_file, out_file=params.out_file,
         affinity=params.affinity)
