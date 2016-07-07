"""Functions for to connect SEA outputs to the E3FP pipeline.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os

from fpcore.fconvert import string2ascii, ascii2string

from e3fp.config.params import read_params
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.pipeline import fprints_from_smiles, fprints_from_sdf

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "config")
PARAM_FILE = os.path.join(CONFIG_DIR, "best_params.cfg")


def load_params():
    """Load best parameter set from file."""
    return read_params(PARAM_FILE)


def fprint_to_native_tuple(fprint):
    """Convert fingerprint to tuple with native string, name."""
    bitstring = fprint.to_bitstring()
    native = string2ascii(bitstring)
    return (native, fprint.name)


def native_tuple_to_fprint(native_tuple):
    """Convert native tuple to fingerprint."""
    native, name = native_tuple
    bitstring = ascii2string(native)
    fprint = Fingerprint.from_bitstring(bitstring)
    fprint.name = name
    return fprint


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
