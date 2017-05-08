# Conformer Generation

Two individual conformer libraries were used in the paper. They may each
be generated using the following commands:
- rms0.5
    `python $E3FP_PACKAGE/conformer/generate.py -s ../data/chembl20_proto_smiles.smi.bz2 -o conformers_proto_rms0.5 --pool_multiplier 2 -r 0.5 -C 2`
- rms1_e20
    `python $E3FP_PACKAGE/conformer/generate.py -s ../data/chembl20_proto_smiles.smi.bz2 -o conformers_proto_rms1_e20 --pool_multiplier 2 -r 1.0 -e 20 -C 2`

* Note: it is highly recommended that you run conformer generation using
one of the parallelization options. Also, if you only need the subset
of conformers used for the final E3FP parameter set (3), then use the
`--first` parameter for early termination.

