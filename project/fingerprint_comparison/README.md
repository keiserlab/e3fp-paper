# Comparing E3FP and ECFP4 Tanimoto Coefficients (TCs)

* Note: Here we assume that E3FP and ECFP4 fingerprints have already
been generated and saved to `e3fp_molecules.csv.bz2` and
`ecfp4_molecules.csv.bz2`, respectively. See the discussion
in [`crossvalidation`](../crossvalidation/sea) for fingerprinting
instructions.

## Computing Pairwise TCs

For both sets of molecules files, run the following series of scripts. Where
multiple fingerprints exist for a given molecule pair (e.g. multiple
conformers for E3FP or tautomers for ECFP4), the maximum TC between all
fingerprint pairs across the two molecules is computed. The results are
multiple flat binary files, each consisting of a flattened lower (below-
diagonal) pairwise TCs triangle matrix, corresponding to a contiguous chunk of
the triangle matrix.

```bash
python get_fingerprint_tril_tcs.py <molecules_file>
```

Assemble binary files into a
[NumPy memmap file](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html).
A memmap enables relatively quick look-up without needing to read many doubles
into memory.

```bash
python binary_to_numpy.py *bin.gz <out_memmap_file> <out_names_file>
```

## Binning TCs Comparisons

For comparing two sets of pairwise TCs, first use the above approach to build
the two pairwise TCs memmaps. Then, count the number of TCs pairs that map to
specific values at a specified precision (number of decimal places). For ECFP4
and E3FP, run

```bash
python count_pairwise_tcs.py <ecfp4_mmap_file> <e3fp_mmap_file> --names ECFP4 E3FP
```
which produces an output file `ecfp4_e3fp_tcs.csv.gz`.
