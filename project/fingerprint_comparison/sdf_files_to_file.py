"""Concatenate conformers from each SDF file in directory into one SDF file.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import sys
import os
import glob

import rdkit
import rdkit.Chem
from python_utilities.io_tools import smart_open
from e3fp.conformer.util import mol_from_sdf


FIRST = 3


def mol_conf_id_from_fn(fn):
    root = os.path.basename(fn).split('.')[0]
    try:
        mol_name, proto_id = root.split('-')
        return mol_name, int(proto_id)
    except ValueError:
        return root, 0


def main(sdf_dir, out_sdf_file, first=3):
    sdf_files = glob.glob(os.path.join(sdf_dir, "*sdf*"))
    sdf_files = sorted(sdf_files, key=mol_conf_id_from_fn)

    with smart_open(out_sdf_file, "wb") as fobj:
        writer = rdkit.Chem.SDWriter(fobj)
        for j, sdf_file in enumerate(sdf_files):
            mol = mol_from_sdf(sdf_file, conf_num=FIRST + 1)
            proto_name = mol.GetProp("_Name")
            mol_name, _ = mol_conf_id_from_fn(proto_name)
            mol.SetProp("_Name", mol_name)
            conf_ids = [conf.GetId() for conf in mol.GetConformers()]
            for i in conf_ids:
                if i >= first and first not in (-1, None):
                    break
                writer.write(mol, confId=i)
            if j > 0 and j % 100 == 0:
                print(j)
        writer.close()


if __name__ == "__main__":
    usage = ("python sdf_files_to_file.py <in_sdf_dir> <out_sdf_file> "
             "[<num_first>]")
    try:
        sdf_dir, out_sdf_file = sys.argv[1:3]
    except:
        sys.exit(usage)
    try:
        first = int(sys.argv[3])
    except:
        first = FIRST
    main(sdf_dir, out_sdf_file, first=first)
