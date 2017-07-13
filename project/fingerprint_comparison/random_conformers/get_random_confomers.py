"""Get random subset of conformers.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import random
import logging
import argparse

import rdkit.Chem
from python_utilities.io_tools import smart_open
from e3fp.conformer.util import mol_from_sdf
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts, \
    lists_dicts_to_molecules


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


def main(sdf_dir, mol_file, num_confs=10000,
         out_conf_file="random_conformers.txt",
         out_sdf_file="random_conformers.sdf.bz2",
         out_mol_file="random_conformers.csv.bz2"):
    confs = set()
    if os.path.isfile(out_mol_file):
        logging.info("Loading existing random molecules.")
        _, conf_mol_list_dict, _ = molecules_to_lists_dicts(out_mol_file,
                                                            merge_proto=False)
        for proto_name in conf_mol_list_dict:
            for _, conf_name in conf_mol_list_dict[proto_name]:
                confs.add(split_conf_name(conf_name))
    else:
        logging.info("Loading molecules file.")
        smiles_dict, mol_list_dict, fp_type = molecules_to_lists_dicts(
            mol_file, merge_proto=False)
        mol_name_to_proto_names = {}
        for proto_name in mol_list_dict:
            mol_name, _ = split_conf_name(proto_name)
            mol_name_to_proto_names.setdefault(mol_name, []).append(proto_name)
        conf_mol_list_dict = {}
        logging.info("Picking random molecules.")
        while len(confs) < num_confs:
            mol_name = random.choice(mol_name_to_proto_names.keys())
            proto_name = random.choice(mol_name_to_proto_names[mol_name])
            _, conf_name = random.choice(mol_list_dict[proto_name])
            conf = split_conf_name(conf_name)
            confs.add(conf)
            conf_mol_list_dict.setdefault(proto_name, set()).add(
                mol_list_dict[proto_name][conf[2]])
            if len(confs) % 100 == 0:
                logging.info(len(confs))
        conf_mol_list_dict = {k: sorted(v) for k, v
                              in conf_mol_list_dict.items()}
        lists_dicts_to_molecules(out_mol_file, smiles_dict, conf_mol_list_dict,
                                 fp_type)
    confs = sorted(confs)

    logging.info("Writing mol names to file.")
    with open(out_conf_file, "w") as f:
        for conf in confs:
            f.write("{}\n".format(join_conf_name(*conf)))

    logging.info("Saving mols to SDF file.")
    with smart_open(out_sdf_file, "wb") as f:
        writer = rdkit.Chem.SDWriter(f)
        for j, conf in enumerate(confs):
            mol_name, proto_id, conf_id = conf
            sdf_file = glob.glob(os.path.join(
                sdf_dir, "{}.sdf*".format(
                    join_conf_name(mol_name, proto_id))))[0]
            mol = mol_from_sdf(sdf_file, conf_num=conf_id + 1)
            name = join_conf_name(*conf)
            mol.SetProp("_Name", name)
            writer.write(mol, confId=conf_id)
            if j > 0 and j % 10 == 0:
                logging.info(j)
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Get random subset of conformers.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf_dir', type=str,
                        help="""Path to directory with input SDF files.""")
    parser.add_argument('mol_file', type=str,
                        help="""Path to SEA-format E3FP molecules file
                             corresponding to same conformers in 'sdf_dir'.""")
    parser.add_argument('--num_confs', type=int, default=10000,
                        help="""Number of random conformers to get.""")
    parser.add_argument('--out_conf_file', type=str,
                        default="random_conformers.txt",
                        help="""Output list of conformers names.""")
    parser.add_argument('--out_sdf_file', type=str,
                        default="random_conformers.sdf.bz2",
                        help="""Output conformers SDF file.""")
    parser.add_argument('--out_mol_file', type=str,
                        default="random_conformers.csv.bz2",
                        help="""Output SEA-format molecules file.""")
    params = parser.parse_args()
    main(params.sdf_dir, params.mol_file, num_confs=params.num_confs,
         out_conf_file=params.out_conf_file, out_sdf_file=params.out_sdf_file,
         out_mol_file=params.out_mol_file)
