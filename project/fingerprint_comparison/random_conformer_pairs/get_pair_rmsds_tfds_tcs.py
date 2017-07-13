"""Save pairwise RMSD, TFDs, and E3FP TCs between conformer pairs.

Authors: Seth Axen
E-mail: seth.axen@gmail.com
"""
import argparse

import numpy as np
import rdkit.Chem
from rdkit.Chem import rdMolAlign, TorsionFingerprints
from e3fp.pipeline import params_to_dicts, fprints_from_mol
from e3fp_paper.pipeline import load_params
from e3fp.fingerprint.metrics import tanimoto


_, FPRINT_PARAMS = params_to_dicts(load_params())
LOG_FREQ = 20  # min number of mol pairs between saves


def get_rmsd(mol):
    rms, tmat = rdMolAlign.GetAlignmentTransform(mol, mol, prbCid=0, refCid=1)
    return rms


def get_tfd(mol):
    return TorsionFingerprints.GetTFDBetweenConformers(mol, [0], [1])


def get_e3fp_tc(mol):
    fprints = fprints_from_mol(mol, fprint_params=FPRINT_PARAMS)
    return tanimoto(fprints[0], fprints[1])


def main(sdf_file, tfds_file='tfds.bin', rmsds_file='rmsds.bin',
         e3fp_tcs_file='e3fp_tcs.bin', log_freq=LOG_FREQ):
    rmsds = []
    tfds = []
    e3fp_tcs = []
    i = 0
    supp = rdkit.Chem.SDMolSupplier(sdf_file)
    while True:
        try:
            mol = next(supp)
            tmp_mol = next(supp)
        except StopIteration:
            break
        mol.AddConformer(tmp_mol.GetConformer(0), assignId=True)
        rmsds.append(get_rmsd(mol))
        tfds.append(get_tfd(mol))
        e3fp_tcs.append(get_e3fp_tc(mol))
        i += 1

        if i > 0 and i % log_freq == 0:
            print(i)

    np.asarray(rmsds, dtype=np.double).tofile(rmsds_file, format="d")
    np.asarray(tfds, dtype=np.double).tofile(tfds_file, format="d")
    np.asarray(e3fp_tcs, dtype=np.double).tofile(e3fp_tcs_file, format="d")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Save pairwise RMSD, TFDs, and E3FP TCs between conformer pairs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf_file', type=str,
                        help="""Path to SDF file containing mols.""")
    parser.add_argument('--rmsds_file', type=str,
                        help="""Path to output file with RMSDs.""")
    parser.add_argument('--tfds_file', type=str,
                        help="""Path to output file with Torsion Fingerprint
                             Deviations (TFDs).""")
    parser.add_argument('--e3fp_tcs_file', type=str,
                        help="""Path to output file with E3FP TCs.""")
    params = parser.parse_args()
    main(params.sdf_file, rmsds_file=params.rmsds_file,
         tfds_file=params.tfds_file, e3fp_tcs_file=params.e3fp_tcs_file)
