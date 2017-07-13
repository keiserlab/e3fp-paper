# Pairwise Comparisons of Random Conformers of the Same Molecules

## Getting Random Pairs of Conformers

First, we need to select 10,000 random conformer pairs. Assuming we have
already generated conformers and performed E3FP fingerprinting, run

```bash
python get_random_pairs.py <sdf_dir> <e3fp_mol_file> --num_pairs 10000
```

## Computing Pairwise E3FP TCs, RMSDs, and Torsion Fingerprint Deviation (TFD)

```bash
python get_pair_rmsds_tfds_tcs.py random_pairs.sdf.bz2
```

## Computing Pairwise FastROCS Shape and Combo Tanimotos

```bash
python get_pair_fastrocs_tcs.py random_pairs.sdf.bz2
```
