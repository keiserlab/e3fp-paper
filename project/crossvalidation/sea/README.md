# K-fold Cross-Validation Using SEA

Cross-validation using SEA is performed in two steps. First, we generate the
fingerprints with the desired parameter set. Second, we run the actual
*k*-fold cross-validation.

## Generating Fingerprints

For each of the E3FP-based fingerprints (E3FP, E3FP-nostereo, E2FP, E2FP-
stereo, E3FP-RDKit), generate fingerprints by running within the directory:

```bash
python $E3FP_PROJECT/scripts/generate_e3fp_fprints.py $E3FP_PROJECT/data/chembl20_proto_smiles.smi.bz2 params.cfg --sdf_dir $E3FP_PROJECT/conformer_generation/conformers_proto_rms0.5 -l fp_log.txt
```

For each of the ECFP-based fingerprints (ECFP4, ECFP4-chiral), generate
fingerprints by running within the directory:

```bash
python $E3FP_PROJECT/scripts/generate_ecfp_fprints.py $E3FP_PROJECT/data/chembl20_proto_smiles.smi.bz2 -l fp_log.txt
```

Add the `--use_chiral` parameter for ECFP4-chiral.

## Running 5-fold Cross-Validation

Within each directory, run cross-validation with:

```bash
python $E3FP_PROJECT/scripts/run_cv.py molecules.csv.bz2 $E3FP_PROJECT/data/chembl20_binding_targets.csv.bz2 --reduce_negatives -l cv_log.txt
```
