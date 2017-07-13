"""Get random pairs of same-molecule conformers.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import os
import glob
import logging
import argparse

import numpy as np
import rdkit.Chem
from python_utilities.io_tools import smart_open
from e3fp.conformer.util import mol_from_sdf
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts


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


def get_random_pairs(proto_name, sdf_dir):
    mol_name, proto_id = split_conf_name(proto_name)
    sdf_file = glob.glob(os.path.join(
        sdf_dir, "{}.sdf*".format(
            join_conf_name(mol_name, proto_id))))[0]
    mol = mol_from_sdf(sdf_file)
    conf_nums = mol.GetNumConformers()

    conf_id_pair = np.random.choice(np.arange(conf_nums), size=2,
                                    replace=False)
    conf_id1, conf_id2 = conf_id_pair
    omol1 = rdkit.Chem.Mol(mol)
    omol1.RemoveAllConformers()
    omol2 = rdkit.Chem.Mol(omol1)
    omol1.AddConformer(mol.GetConformer(conf_id1))
    omol1.SetProp("_Name", join_conf_name(mol_name, proto_id, conf_id1))
    omol2.AddConformer(mol.GetConformer(conf_id2))
    omol2.SetProp("_Name", join_conf_name(mol_name, proto_id, conf_id2))
    return omol1, omol2


def main(sdf_dir, mol_file, num_pairs=10000,
         out_sdf_file="random_pairs.sdf.bz2"):
    logging.info("Loading molecules file.")
    smiles_dict, mol_list_dict, fp_type = molecules_to_lists_dicts(
        mol_file, merge_proto=False)
    mol_list_dict = {k: v for k, v in mol_list_dict.items() if len(v) > 1}

    logging.info("Picking random molecules.")
    mol_proto_num = {}
    for proto_name in mol_list_dict.keys():
        mol_name = proto_name.split("-")[0]
        if mol_name in mol_proto_num:
            mol_proto_num[mol_name] += 1
        else:
            mol_proto_num[mol_name] = 1
    proto_names, proto_nums = zip(*[(k, mol_proto_num[k.split("-")[0]])
                                    for k in mol_list_dict.keys()])
    proto_probs = 1. / np.asanyarray(proto_nums)
    proto_probs /= np.sum(proto_probs)

    random_proto_names = np.random.choice(proto_names, size=num_pairs,
                                          replace=False, p=proto_probs)

    with smart_open(out_sdf_file, "wb") as f:
        writer = rdkit.Chem.SDWriter(f)

        for i, proto_name in enumerate(sorted(random_proto_names)):
            mol1, mol2 = get_random_pairs(proto_name, sdf_dir)
            writer.write(mol1)
            writer.write(mol2)

            if i > 0 and i % 100 == 0:
                logging.info(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Get random pairs of same-molecule conformers.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf_dir', type=str,
                        help="""Path to directory with input SDF files.""")
    parser.add_argument('mol_file', type=str,
                        help="""Path to SEA-format E3FP molecules file
                             corresponding to same conformers in 'sdf_dir'.""")
    parser.add_argument('--num_pairs', type=int, default=10000,
                        help="""Number of random conformer pairs to get.""")
    parser.add_argument('--out_sdf_file', type=str,
                        default="random_pairs.sdf.bz2",
                        help="""Output conformers pairs SDF file.""")
    params = parser.parse_args()
    main(params.sdf_dir, params.mol_file, num_pairs=params.num_pairs,
         out_sdf_file=params.out_sdf_file)
