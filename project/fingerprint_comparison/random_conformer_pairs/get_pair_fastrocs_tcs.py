#!/usr/local/fastrocs/bin/python2.7
"""Save FastROCS shape and combo Tanimotos between consecutive conformer pairs.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import sys
import logging
import argparse

import numpy as np
from openeye.oechem import *
from openeye.oefastrocs import *
from python_utilities.scripting import setup_logging

oepy = os.path.join(os.path.dirname(__file__), "openeye", "python")
sys.path.insert(0, os.path.realpath(oepy))


def main(mol_dbase, combo_tcs_file='fastrocs_combo_tcs.bin',
         shape_tcs_file='fastrocs_shape_tcs.bin',
         mol_names_file='fastrocs_mol_names.csv', overwrite=False, log=None,
         log_freq=100):
    setup_logging(log)

    logging.info(
        "Will save tcs to {} and {} and mol names to {}".format(
            combo_tcs_file, shape_tcs_file, mol_names_file))

    ifs = oemolistream()
    if not ifs.open(mol_dbase):
        OEThrow.Fatal("Unable to open {} for reading".format(mol_dbase))

    # Configure OpenEye
    dbtype = OEShapeDatabaseType_Default
    options = OEShapeDatabaseOptions()
    options.SetScoreType(dbtype)
    combo_tc_getter = OEShapeDatabaseScore.GetTanimotoCombo
    shape_tc_getter = OEShapeDatabaseScore.GetShapeTanimoto

    combo_tcs = []
    shape_tcs = []
    dots = OEDots(log_freq, 20, "looping through molecule scores")
    last_name = None
    search_db = None
    for index, mol in enumerate(ifs.GetOEMols()):
        conf_name = mol.GetTitle()
        proto_name = conf_name.split("_")[0]
        if proto_name != last_name:
            last_name = proto_name
            search_db = OEShapeDatabase(dbtype)
            search_db.AddMol(mol)
            continue

        combo_tc = shape_tc = 0
        i = 0
        for conf in mol.GetConfs():
            for score in search_db.GetScores(conf, options):
                combo_tc = combo_tc_getter(score)
                shape_tc = shape_tc_getter(score)
                if i > 0:
                    sys.exit("More than one conformer was found in database.")
                i += 1

        dots.Update()
        combo_tcs.append(combo_tc)
        shape_tcs.append(shape_tc)
        assert(len(combo_tcs) == (index + 1) / 2)

    np.asarray(combo_tcs, dtype=np.double).tofile(combo_tcs_file, format="d")
    np.asarray(shape_tcs, dtype=np.double).tofile(shape_tcs_file, format="d")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        """Save FastROCS shape and combo Tanimotos between consecutive
        conformer pairs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sdf_file', type=str,
                        help="""Path to SDF.""")
    parser.add_argument('--mol_names_file', type=str,
                        default='fastrocs_mol_names.csv',
                        help="""Output list of searched mol names.""")
    parser.add_argument('--shape_tcs_file', type=str,
                        default='fastrocs_shape_tcs.bin',
                        help="""Output ROCS shape Tanimotos.""")
    parser.add_argument('--combo_tcs_file', type=str,
                        default='fastrocs_combo_tcs.bin',
                        help="""Output ROCS combo (shape+color) Tanimotos.""")
    parser.add_argument('-O', '--overwrite', action="store_true",
                        help="""Overwrite existing file(s).""")
    params = parser.parse_args()
    main(params.sdf_file, mol_names_file=params.mol_names_file,
         shape_tcs_file=params.shape_tcs_file,
         combo_tcs_file=params.combo_tcs_file, overwrite=params.overwrite)
