"""Compare ECFP4 and E3FP runtimes.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import time
import sys
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from python_utilities.parallel import Parallelizer
from e3fp.conformer.util import smiles_to_dict, mol_from_sdf
from e3fp.pipeline import params_to_dicts, fprints_from_mol
from e3fp_paper.pipeline import load_params


def benchmark_fprinting(smiles, sdf_file, name, fprint_params={}):
    mol = mol_from_sdf(sdf_file, conf_num=fprint_params.get('first', None))
    num_confs = mol.GetNumConformers()
    num_rot = AllChem.CalcNumRotatableBonds(mol)
    num_heavy = mol.GetNumHeavyAtoms()

    start_time = time.time()
    Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    fprint_2d_time = time.time() - start_time

    start_time = time.time()
    fprints_from_mol(mol, fprint_params=fprint_params, save=False)
    fprint_3d_time = time.time() - start_time

    return (fprint_2d_time, fprint_3d_time, num_heavy, num_confs, num_rot)


def get_sdf_file(mol_name, sdf_dir):
    return os.path.join(sdf_dir, mol_name + ".sdf.bz2")


def main(smiles_file, sdf_dir, out_file):
    _, fprint_params = params_to_dicts(load_params())
    smiles_dict = smiles_to_dict(smiles_file)

    para = Parallelizer()
    smiles_iter = ((smiles, get_sdf_file(name, sdf_dir), name)
                   for name, smiles in smiles_dict.items())
    kwargs = {"fprint_params": fprint_params}
    results_iter = para.run_gen(benchmark_fprinting, smiles_iter,
                                kwargs=kwargs)

    with open(out_file, "w") as f:
        f.write("\t".join(["Name", "ECFP4 Time", "E3FP Time", "Num Heavy",
                           "Num Confs", "Num Rot"]) + "\n")
        for results, (_, _, name) in results_iter:
            print(results)
            f.write("{}\t{:.4g}\t{:.4g}\t{:d}\t{:d}\t{:d}\n".format(
                name, *results))


if __name__ == "__main__":
    usage = "python run_benchmark.py <smiles_file> <sdf_dir> <out_file>"
    try:
        smiles_file, sdf_dir, out_file = sys.argv[1:]
    except:
        sys.exit(usage)
    main(smiles_file, sdf_dir, out_file)
