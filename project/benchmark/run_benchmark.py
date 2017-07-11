"""Compare ECFP4 and E3FP runtimes.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import time
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from python_utilities.parallel import Parallelizer
from e3fp.conformer.util import smiles_to_dict, mol_from_smiles
from e3fp.conformer.generator import ConformerGenerator
from e3fp.pipeline import params_to_dicts, fprints_from_mol
from e3fp_paper.pipeline import load_params


def get_random_smiles(smiles_file, n):
    smiles_dict = smiles_to_dict(smiles_file)
    return {k: smiles_dict[k] for k
            in np.random.choice(smiles_dict.keys(), size=n, replace=False)}


def benchmark_fprinting(smiles, name, confgen_params={},
                        fprint_params={}):
    mol = mol_from_smiles(smiles, name)
    num_rot = AllChem.CalcNumRotatableBonds(mol)
    num_heavy = mol.GetNumHeavyAtoms()

    start_time = time.time()
    Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    fprint_2d_time = time.time() - start_time

    start_time = time.time()
    conf_gen = ConformerGenerator(**confgen_params)
    mol = conf_gen.generate_conformers(mol)
    confgen_time = time.time() - start_time
    num_confs = mol.GetNumConformers()

    start_time = time.time()
    fprints_from_mol(mol, fprint_params=fprint_params, save=False)
    fprint_3d_time = time.time() - start_time

    return (fprint_2d_time, confgen_time, fprint_3d_time, num_heavy,
            num_confs, num_rot)


def main(smiles_file, out_file, num_mols):
    confgen_params, fprint_params = params_to_dicts(load_params())
    del confgen_params['out_dir'], confgen_params['compress']
    smiles_dict = get_random_smiles(smiles_file, n=num_mols)

    para = Parallelizer()
    smiles_iter = ((smiles, name) for name, smiles in smiles_dict.items())
    kwargs = {"confgen_params": confgen_params, "fprint_params": fprint_params}
    results_iter = para.run_gen(benchmark_fprinting, smiles_iter,
                                kwargs=kwargs)

    with open(out_file, "w") as f:
        f.write("\t".join(["ECFP4 Time", "ConfGen Time", "E3FP Time",
                           "Num Heavy", "Num Confs", "Num Rot"]) + "\n")
        for results, (_, name) in results_iter:
            f.write(
                "{:.4g}\t{:.4g}\t{:.4g}\t{:d}\t{:d}\t{:d}\n".format(*results))


if __name__ == "__main__":
    usage = "python run_benchmark.py <smiles_file> <num_mols> <out_file>"
    try:
        smiles_file, num_mols, out_file = sys.argv[1:]
        num_mols = int(num_mols)
    except:
        sys.exit(usage)
    main(smiles_file, out_file, num_mols)
