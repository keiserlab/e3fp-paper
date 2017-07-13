"""Create input files for PDB/ChEMBL20 enrichment.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
from itertools import product
import cPickle as pkl

import pandas as pd
from rdkit.Chem import MolToSmiles
from rdkit.Chem.rdMolTransforms import CanonicalizeMol

from seacore.util.library import SetValue
from python_utilities.io_tools import smart_open, touch_dir
from e3fp.conformer.util import smiles_to_dict, dict_to_smiles, \
    mol_from_smiles, mol_from_mol2, mol_to_sdf, mol_to_standardised_mol
from e3fp_paper.sea_utils.util import targets_to_dict, dict_to_targets

PDB_DIR = "./scPDB"
PDB_ANNOT_FILE = os.path.join(PDB_DIR, "scPDB_Annotation.csv")
CHEMBL_TARGETS_FILE = os.path.join(os.environ['E3FP_PROJECT'], "data",
                                   "chembl20_binding_targets.csv.bz2")
CHEMBL_SMILES_FILE = os.path.join(os.environ['E3FP_PROJECT'], "data",
                                  "chembl20_proto_smiles.smi.bz2")
PDB_SMILES_FILE = "pdb_smiles.smi"
PDB_CANON_SMILES_FILE = "pdb_canon_smiles.smi"
OUT_CHEMBL_SMILES_FILE = "chembl_smiles.smi"
CHEMBL_CANON_SMILES_FILE = "chembl_canon_smiles.smi"
PDB_CHEMBL_MOL_MAP = "pdb_chembl_mol_map.pkl.bz2"
CHEMBL_TARGETS = "chembl_targets.csv.bz2"
PDB_TARGETS = "pdb_targets.csv.bz2"
PDB_CONF_DIR = os.path.join(os.environ['E3FP_PROJECT'], "conformer_generation",
                            "conformers_proto_rms0")
MIN_MOLS = 25


def read_pdb_annotation(annot_file):
    df = pd.read_csv(annot_file, sep="\t")
    return df


def get_canonical_smiles(smiles_dict):
    canonical_smiles_dict = {}
    for i, (mol_name, smiles) in enumerate(smiles_dict.items()):
        try:
            mol = mol_from_smiles(smiles, mol_name, standardise=True)
            canon_smiles = MolToSmiles(mol, isomericSmiles=True)
            canonical_smiles_dict[mol_name] = canon_smiles
        except:
            canonical_smiles_dict[mol_name] = smiles
    return canonical_smiles_dict


def get_canonical_smiles_from_mol2(mol2_dict):
    canonical_smiles_dict = {}
    for i, (mol_name, mol2_file) in enumerate(mol2_dict.items()):
        mol = mol_from_mol2(mol2_file, mol_name)
        try:
            mol = mol_to_standardised_mol(mol)
        except:
            pass
        canon_smiles = MolToSmiles(mol, isomericSmiles=True)
        canonical_smiles_dict[mol_name] = canon_smiles
    return canonical_smiles_dict


def reverse_smiles_dict(smiles_dict):
    d = {}
    for k, v in smiles_dict.items():
        d.setdefault(v, set()).add(k)
    return d


def mol_map_from_smiles(smiles_dict1, smiles_dict2):
    smiles_to_mols1 = reverse_smiles_dict(smiles_dict1)
    smiles_to_mols2 = reverse_smiles_dict(smiles_dict2)
    mol_map1 = {}
    mol_map2 = {}
    for smiles, mol_set1 in smiles_to_mols1.items():
        if smiles not in smiles_to_mols2:
            continue
        mol_set2 = smiles_to_mols2[smiles]
        for mol1, mol2 in product(mol_set1, mol_set2):
            mol_map1.setdefault(mol1, set()).add(mol2)
            mol_map2.setdefault(mol2, set()).add(mol1)
    mol_map1 = {k: sorted(v) for k, v in mol_map1.items()}
    mol_map2 = {k: sorted(v) for k, v in mol_map2.items()}
    return mol_map1, mol_map2

if __name__ == "__main__":
    # Read input files
    annot_df = read_pdb_annotation(PDB_ANNOT_FILE)
    annot_df['mol_name'] = ["{}-{}".format(row['HET_CODE'], row['PDB_ID'])
                            for i, row in annot_df.iterrows()]
    annot_df.set_index(['Uniprot_ID', 'mol_name'], inplace=True)

    chembl_targets_dict = targets_to_dict(CHEMBL_TARGETS_FILE, affinity=10000)
    chembl_smiles_dict = smiles_to_dict(CHEMBL_SMILES_FILE)

    # Add PDB data to useful maps
    pdb_pdb_to_mol = {}
    pdb_pdb_to_name = {}
    pdb_smiles_dict = {}
    for (uid, mol_name), row in annot_df.iterrows():
        smiles = row['SMILES']
        pdb_smiles_dict[mol_name] = smiles
        pdb_pdb_to_mol.setdefault(uid, set()).add((mol_name, smiles))
        pdb_pdb_to_name[uid] = (row['Uniprot_AC'], row['Uniprot_Name'])
    for uid, mol_set in pdb_pdb_to_mol.items():
        if len(mol_set) < MIN_MOLS:
            del pdb_pdb_to_mol[uid]
            del pdb_pdb_to_name[uid]

    # Build SEA-format PDB targets dict
    pdb_mol_names = set()
    pdb_mol2_files = {}
    pdb_targets_dict = {}
    for k, v in chembl_targets_dict.items():
        uid = v.name
        if uid in pdb_pdb_to_name:
            acc, desc = pdb_pdb_to_name[uid]
            cids = {x[0] for x in pdb_pdb_to_mol[uid]}
            mol2_cids = set()
            for cid in cids:
                pair = (uid, cid)
                pdb_id = annot_df.loc[pair]['PDB_ID']
                mol2_file = os.path.join(PDB_DIR, pdb_id, "ligand.mol2")
                try:
                    mol = mol_from_mol2(mol2_file, cid)
                except:
                    continue
                mol2_cids.add(cid)
                pdb_mol2_files[cid] = mol2_file
            if len(mol2_cids) < MIN_MOLS:
                continue
            pdb_mol_names.update(mol2_cids)
            cids = sorted(mol2_cids)
            pdb_set_value = SetValue(uid, cids, desc)
            pdb_targets_dict[k] = pdb_set_value
    dict_to_targets(PDB_TARGETS, pdb_targets_dict)

    # Build/load maps between PDB and CHEMBL
    if not os.path.isfile(PDB_CHEMBL_MOL_MAP):
        if not os.path.isfile(PDB_CANON_SMILES_FILE):
            pdb_canonical_smiles_dict = get_canonical_smiles_from_mol2(
                {k: v for k, v in pdb_mol2_files.items()})
            dict_to_smiles(PDB_CANON_SMILES_FILE, pdb_canonical_smiles_dict)
        pdb_canonical_smiles_dict = smiles_to_dict(PDB_CANON_SMILES_FILE)

        if not (os.path.isfile(CHEMBL_CANON_SMILES_FILE)):
            chembl_canonical_smiles_dict = get_canonical_smiles(
                chembl_smiles_dict)
            dict_to_smiles(CHEMBL_CANON_SMILES_FILE,
                           chembl_canonical_smiles_dict)
        chembl_canonical_smiles_dict = smiles_to_dict(CHEMBL_CANON_SMILES_FILE)

        pdb_to_chembl_mol_map, chembl_to_pdb_mol_map = mol_map_from_smiles(
            pdb_canonical_smiles_dict, chembl_canonical_smiles_dict)
        with smart_open(PDB_CHEMBL_MOL_MAP, "w") as f:
            pkl.dump((pdb_to_chembl_mol_map, chembl_to_pdb_mol_map), f)
    else:
        with smart_open(PDB_CHEMBL_MOL_MAP, "r") as f:
            pdb_to_chembl_mol_map, chembl_to_pdb_mol_map = pkl.load(f)

    # Save query ligands to SDF files and get SMILES
    pdb_smiles_dict = {}
    skip_pairs = set()
    touch_dir(PDB_CONF_DIR)
    for mol_name, mol2_file in pdb_mol2_files.items():
        mol = mol_from_mol2(mol2_file, mol_name)
        smiles = MolToSmiles(mol, isomericSmiles=True)
        pdb_smiles_dict[mol_name] = smiles
        CanonicalizeMol(mol)
        sdf_file = os.path.join(PDB_CONF_DIR, "{}.sdf.bz2".format(mol_name))
        mol_to_sdf(mol, sdf_file)
    dict_to_smiles(PDB_SMILES_FILE, pdb_smiles_dict)

    # Build filtered CHEMBL targets dict
    chembl_smiles_cids = {k.split("-")[0] for k in chembl_smiles_dict.keys()}
    chembl_cids = set()
    if not os.path.isfile(CHEMBL_TARGETS):
        filtered_chembl_targets_dict = {}
        for k, v in pdb_targets_dict.items():
            chembl_set_value = chembl_targets_dict[k]
            chembl_pdb_cids = set.union(*[
                set(pdb_to_chembl_mol_map.get(x, [])) for x in v.cids])
            cids = [x for x in chembl_set_value.cids
                    if (x not in chembl_pdb_cids and
                        x in chembl_smiles_cids)]
            chembl_cids.update(set(cids))
            set_value = SetValue(chembl_set_value.name, cids,
                                 chembl_set_value.description)
            filtered_chembl_targets_dict[k] = set_value

        dict_to_targets(CHEMBL_TARGETS, filtered_chembl_targets_dict)
    else:
        filtered_chembl_targets_dict = targets_to_dict(CHEMBL_TARGETS)

    chembl_smiles_dict = {k: v for k, v in chembl_smiles_dict.items()
                          if k.split("-")[0] in chembl_cids}
    dict_to_smiles(OUT_CHEMBL_SMILES_FILE, chembl_smiles_dict)
