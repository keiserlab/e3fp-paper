# Benchmarking ECFP, E3FP, and ROCS

In this analysis, we determine the compute time of ECFP4 and E3FP
fingerprinting, as well as the [conformer generation](../conformer_generation)
protocol included with E3FP. We additionally determine the compute time of
pairwise comparisons of these fingerprints, as well as pairwise ROCS combo
Tanimotos.

## Selecting Random Molecules

```bash
python get_random_subset.py $E3FP_PROJECT/conformer_generation/conformers_proto_rms0.5 $E3FP_PROJECT/data/chembl20_proto_smiles.smi.bz2
```

This script saves 10,000 random SMILES to `random_mols.smi`.

## Generate ECFP4 Fingerprints

```bash
python $E3FP_PROJECT/scripts/generate_ecfp_fprints.py random_mols.smi -o ecfp_molecules.csv.bz2 -l ecfp_bench_log.txt
```

## Generate Conformers

```bash
python $E3FP_PACKAGE/conformer/generate.py -s random_mols.smi --pool_multiplier 2 --first 3
```

## Generate E3FP Fingerprints

```bash
python $E3FP_PROJECT/scripts/generate_e3fp_fprints.py random_mols.smi $E3FP_PAPER/e3fp_paper/config/best_params.cfg --sdf_dir conformers -o e3fp_molecules.csv.bz2 -l e3fp_bench_log.txt
```

## Compute pairwise TCs

```bash
python ../fingerprint_comparison/get_fingerprint_tril_tcs.py ecfp_molecules.csv.bz2 --merge_confs --parallel_mode=serial -l ecfp_pairwise_tc.log --num_proc 2 
python ../fingerprint_comparison/get_fingerprint_tril_tcs.py e3fp_molecules.csv.bz2 --merge_confs --parallel_mode=serial -l ecfp_pairwise_tc.log --num_proc 2 
```

## Compute ROCS Combo Tanimotos

### Concatenate SDF files

```bash
python ../fingerprint_comparison/sdf_files_to_file.py conformers combined.sdf
```

### Run FastROCS

```bash
python ../fingerprint_comparison/get_fastrocs_tril_tcs.py combined.sdf 0 9999 --merge_confs --save_freq 10000
```
