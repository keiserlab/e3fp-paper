"""Generate ECFP fingerprints and save to SEA molecules file.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import argparse
import logging

from rdkit import Chem
from seashell.cli.fputil import FingerprintType

from python_utilities.scripting import setup_logging
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.conformer.util import smiles_to_dict, mol_from_smiles
from e3fp_paper.pipeline import fprint_to_native_tuple
from e3fp_paper.sea_utils.util import lists_dicts_to_molecules


def get_fprint2d_fptype(bits=1024, radius=2, use_chiral=False):
    fp_type = FingerprintType()
    fp_type.data[fp_type.KEYS[0]] = 'rdkit_ecfp'
    if use_chiral:
        fp_type.data[fp_type.KEYS[0]] = fp_type.data[fp_type.KEYS[0]] + "_chiral"
    fp_type.data[fp_type.KEYS[1]] = 'sea_native'
    fp_type.data[fp_type.KEYS[2]] = bits
    fp_params = {'bit_length': bits, 'circle_radius': radius}
    fp_type.data[fp_type.KEYS[3]] = str(sorted(fp_params.items()))
    return fp_type


def fprint2d_from_mol(mol, bits=1024, radius=2, use_chiral=False):
    bitvect = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, bits, useChirality=use_chiral)
    fp = Fingerprint.from_rdkit(bitvect)
    fp.name = mol.GetProp("_Name")
    return fp


def run(smiles_file, bits=1024, radius=2, use_chiral=False,
        out_file="molecules.csv.bz2", log=None):
    setup_logging(log)

    smiles_dict = smiles_to_dict(smiles_file)
    mol_list_dict = {}
    for name, smiles in smiles_dict.iteritems():
        try:
            mol = mol_from_smiles(smiles, name)
            logging.info("Generating fingerprint for {}".format(name))
            fp = fprint2d_from_mol(mol, bits=bits, radius=radius,
                                   use_chiral=use_chiral)
            logging.info("Generated fingerprint for {}".format(name))
            mol_list_dict.setdefault(name, []).append(
                fprint_to_native_tuple(fp))

        except Exception:
            logging.warning("Fingerprinting {} failed.".format(name))
    fp_type = get_fprint2d_fptype(bits=bits, radius=radius,
                                  use_chiral=use_chiral)
    lists_dicts_to_molecules(out_file, smiles_dict, mol_list_dict, fp_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Generate ECFP fingerprints from SMILES.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('smiles_file', type=str,
                        help="""Path to SMILES file for fingerprinting.""")
    parser.add_argument('-b', '--bits', type=int, default=1024,
                        help="""Set number of bits for final folded
                             fingerprint.""")
    parser.add_argument('-r', '--radius', type=int, default=2,
                        help="""Set maximum radius in bond number.""")
    parser.add_argument('--use_chiral', type=bool, default=False,
                        help="""Use use_chiralchemical information.""")
    parser.add_argument('-o', '--out_file', type=str,
                        default="molecules.csv.bz2",
                        help="""Output SEA molecules file.""")
    parser.add_argument('-l', '--log', type=str, default=None,
                        help="Log filename.")

    params = parser.parse_args()

    kwargs = dict(params._get_kwargs())
    smiles_file = kwargs.pop('smiles_file')
    run(smiles_file, **kwargs)
