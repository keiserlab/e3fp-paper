"""Functions for to connect SEA outputs to the E3FP pipeline.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os

from rdkit import Chem
from fpcore.fconvert import string2ascii, ascii2string

from e3fp.config.params import read_params
from e3fp.conformer.util import mol_from_sdf
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.pipeline import fprints_from_smiles, fprints_from_sdf, \
                          fprints_from_mol

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "config")
PARAM_FILE = os.path.join(CONFIG_DIR, "best_params.cfg")
FIT_FILE = os.path.join(CONFIG_DIR, "best_params.fit")


def load_params():
    """Load best parameter set from file."""
    return read_params(PARAM_FILE)


def get_fit_file():
    """Get path to SEA library fit file for best parameter set."""
    return FIT_FILE


def fprint_to_native_tuple(fprint):
    """Convert fingerprint to tuple with native string, name."""
    bitstring = fprint.to_bitstring()
    native = string2ascii(bitstring)
    return (native, fprint.name)


def native_tuple_to_fprint(native_tuple, bits=None):
    """Convert native tuple to fingerprint."""
    native, name = native_tuple
    bitstring = ascii2string(native)
    fprint = Fingerprint.from_bitstring(bitstring, bits=bits)
    fprint.name = name
    return fprint


def native_tuples_from_mol(mol, fprint_params={}, save=False):
    """Fingerprint molecule and convert to native encoding."""
    if not mol.HasProp("_Name"):
        raise ValueError(
            "mol must have a '_Name' property or `name` must be provided")

    fprints_list = fprints_from_mol(mol, fprint_params=fprint_params,
                                    save=save)
    native_tuples = list(map(fprint_to_native_tuple, fprints_list))
    return native_tuples


def native_tuples_from_smiles(smiles, name, confgen_params={},
                              fprint_params={}, save=False):
    """Generate conformers, fprints, and native encoding from SMILES string."""
    fprints_list = fprints_from_smiles(smiles, name,
                                       confgen_params=confgen_params,
                                       fprint_params=fprint_params, save=save)
    native_tuples = list(map(fprint_to_native_tuple, fprints_list))
    return native_tuples


def native_tuples_from_sdf(sdf_file, fprint_params={}, save=False):
    """Fingerprint conformers from SDF file and convert to native_tuples."""
    fprints_list = fprints_from_sdf(sdf_file, fprint_params=fprint_params,
                                    save=save)
    native_tuples = list(map(fprint_to_native_tuple, fprints_list))
    return native_tuples


def fprint2d_from_mol(mol, bits=1024, radius=2):
    """Generate ECFP fingerprint from mol."""
    bitvect = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                                  bits)
    fp = Fingerprint.from_rdkit(bitvect)
    fp.name = mol.GetProp("_Name")
    return fp


def fprint2d_from_sdf(sdf_file, bits=1024, radius=2):
    """Generate ECFP fingerprint from sdf file."""
    mol = mol_from_sdf(sdf_file)
    return fprint2d_from_mol(mol, bits=bits, radius=radius)
