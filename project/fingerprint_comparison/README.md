# Comparing E3FP and ECFP4 Tanimoto Coefficients (TCs)

* Note: Here we assume that E3FP and ECFP4 fingerprints have already
been generated and saved to `e3fp_molecules.csv.bz2` and
`ecfp4_molecules.csv.bz2`, respectively. See the discussion
in [`crossvalidation`](../crossvalidation/sea) for fingerprinting
instructions.

The script `compare_fingerprints.py` computes TCs between fingerprints
for all pairs of molecules common between the two provided files.
Where multiple fingerprints exist for a given molecule pair (e.g. 
multiple conformers for E3FP or tautomers for ECFP4), the maximum
TC between all fingerprint pairs across the two molecules is
computed. This can be run with:

```bash
python compare_fingerprints.py ecfp4_molecules.csv.bz2 e3fp_molecules.csv.bz2
```

This produces two output files:

- `ecfp_e3fp_tcs_counts.csv.bz2` contains counts of number of molecule pairs
  per TC bin. Bins are used for space efficiency.
- `ecfp_e3fp_tcs_examples.csv.bz2` contains up to 10 example molecules for
  each TC bin. This allows examination of the types of molecules that appear
  in different regions of TC space (see paper Figures 1a and 3).

As this comparison takes a long time to run, these files are regularly cached.
