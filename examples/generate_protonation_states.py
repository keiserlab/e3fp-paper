"""Write protonation SMILES from input SMILES file.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import sys

from python_utilities.parallel import Parallelizer
from python_utilities.scripting import setup_logging
from e3fp.conformer.util import smiles_to_dict, dict_to_smiles
from e3fp.conformer.protonation import smiles_dict_to_proto_smiles_dict, \
                                       smiles_to_proto_smiles

setup_logging(verbose=True)
in_smiles_file, out_smiles_file = sys.argv[1:]
in_smiles_dict = smiles_to_dict(in_smiles_file)
parallelizer = Parallelizer(parallel_mode="processes", num_proc=5)
# smiles_iter = ((smiles, mol_name) for mol_name, smiles
#                in sorted(in_smiles_dict.items()))
proto_smiles_dict = smiles_dict_to_proto_smiles_dict(in_smiles_dict,
                                                     parallelizer=parallelizer)
# proto_smiles_dict = {smiles_tuple[1]: smiles_tuple[0]
#                      for proto_smiles_list, smiles_tuple
#                      in parallelizer.run_gen(smiles_to_proto_smiles,
#                                              smiles_iter,
#                                              kwargs={"dist_cutoff": -1})
#                      if len(proto_smiles_list) > 0}
dict_to_smiles(out_smiles_file, proto_smiles_dict)
