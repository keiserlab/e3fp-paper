"""Get random subset of molecules and compute properties.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import random
import logging
import argparse

from rdkit.Chem import AllChem
from python_utilities.io_tools import smart_open
from e3fp.conformer.util import mol_from_sdf, smiles_to_dict, \
    dict_to_smiles, mol_from_smiles


def split_conf_name(conf_name):
    try:
        proto_name, conf_id = conf_name.split("_")
    except ValueError:
        proto_name = conf_name
    mol_name, proto_id = proto_name.split("-")
    try:
        return mol_name, int(proto_id), int(conf_id)
    except NameError:
        return mol_name, int(proto_id)


def join_conf_name(mol_name, proto_id, conf_id=None):
    if conf_id is None:
        return "{}-{}".format(mol_name, proto_id)
    else:
        return "{}-{}_{}".format(mol_name, proto_id, conf_id)


def get_sdf_file(sdf_dir, proto_name):
    return os.path.join(sdf_dir, proto_name + ".sdf.bz2")


def main(sdf_dir, smiles_file, num_mols=10000, first=3,
         out_props_file="random_mols_props.txt",
         out_smiles_file="random_mols.csv.bz2"):
    mol_names = set()
    if os.path.isfile(out_smiles_file):
        logging.info("Loading existing random molecules.")
        smiles_dict = smiles_to_dict(out_smiles_file)
        mol_names.update(set(smiles_dict))
        out_sdf_files_dict = {k: get_sdf_file(sdf_dir, k) for k in mol_names}
    else:
        logging.info("Loading SMILES file.")
        smiles_dict = smiles_to_dict(smiles_file)
        remaining_mol_names = set(smiles_dict.keys())
        out_smiles_dict = {}
        out_sdf_files_dict = {}
        logging.info("Picking random molecules.")
        while len(mol_names) < num_mols:
            print(len(mol_names))
            proto_name = random.choice(smiles_dict.keys())
            if proto_name not in remaining_mol_names:
                continue
            remaining_mol_names.remove(proto_name)
            sdf_file = get_sdf_file(sdf_dir, proto_name)
            if not os.path.isfile(sdf_file):
                continue
            mol_names.add(proto_name)
            out_smiles_dict[proto_name] = smiles_dict[proto_name]
            out_sdf_files_dict[proto_name] = sdf_file

            if len(mol_names) % 100 == 0:
                logging.info(len(mol_names))

        dict_to_smiles(out_smiles_file, out_smiles_dict)

    mol_names = sorted(mol_names)

    logging.info("Computing mol properties.")
    mol_props = {}
    for name, smiles in smiles_dict.items():
        mol = mol_from_smiles(smiles, name)
        nheavy = mol.GetNumHeavyAtoms()
        nrot = AllChem.CalcNumRotatableBonds(mol)
        mol_props[name] = (nheavy, nrot)

    with open(out_props_file, "w") as f:
        f.write("mol_name\tnheavy\tnrot\n")
        for mol_name in mol_names:
            nheavy, nrot = mol_props[mol_name]
            f.write("{}\t{:d}\t{:d}\n".format(mol_name, nheavy, nrot))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Get random subset of molecules and compute properties.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf_dir', type=str,
                        help="""Path to directory with input SDF files.""")
    parser.add_argument('smiles_file', type=str,
                        help="""Path to SMILES file
                             corresponding to same conformers in 'sdf_dir'.""")
    parser.add_argument('--num_mols', type=int, default=10000,
                        help="""Number of random conformers to get.""")
    parser.add_argument('--first', type=str, default=3,
                        help="""Number of first conformers to take.""")
    parser.add_argument('--out_props_file', type=str,
                        default="random_mols_props.txt",
                        help="""Output list of mol names with properties.""")
    parser.add_argument('--out_smiles_file', type=str,
                        default="random_mols.smi",
                        help="""Output SMILES file.""")
    params = parser.parse_args()
    main(params.sdf_dir, params.smiles_file, num_mols=params.num_mols,
         first=params.first, out_props_file=params.out_props_file,
         out_smiles_file=params.out_smiles_file)
