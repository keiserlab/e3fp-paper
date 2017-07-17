# Comparing E3FP and ECFP4 Tanimoto Coefficients (TCs)

* Note: Here we assume that E3FP and ECFP4 fingerprints have already
been generated and saved to `e3fp_molecules.csv.bz2` and
`ecfp4_molecules.csv.bz2`, respectively. See the discussion
in [`crossvalidation`](../crossvalidation/sea) for fingerprinting
instructions.

## Computing Pairwise Fingerprint TCs

For both sets of molecules files, run the following series of scripts. Where
multiple fingerprints exist for a given molecule pair (e.g. multiple
conformers for E3FP or tautomers for ECFP4), the maximum TC between all
fingerprint pairs across the two molecules is computed. The results are
multiple flat binary files, each consisting of a flattened lower (below-
diagonal) pairwise TCs triangle matrix, corresponding to a contiguous chunk of
the triangle matrix.

```bash
python get_fingerprint_tril_tcs.py <molecules_file> --merge_confs
```

## Computing Pairwise FastROCS Shape and Combo Tanimotos

Run [FastROCS](https://docs.eyesopen.com/toolkits/python/fastrocstk/index.html)
on the same set of conformers used for computing pairwise E3FP TCs above.
First, concatenate the SDF files:

```bash
python sdf_files_to_file.py $E3FP_PROJECT/conformer_generation/conformers_proto_rms0.5 combined.sdf
```

We must manually split the lower triangle matrix into batches consisting of
roughly equal numbers of pairs. 

```bash
python get_triangle_indices.py <mol_num> <batch_num>
```

This script prints out start and stop row indices (inclusive) for each batch.
For each batch, run

```bash
python get_fastrocs_tril_tcs.py combined.sdf <start_index> <stop_index> --merge_confs
```

## Merging Batch Files

Each of the above scripts produces multiple binary files. We need to assemble
these into into a
[NumPy memmap file](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html).
A memmap enables relatively quick look-up without needing to read many doubles
into memory. The format for a NumPy memmap file on disk is identical to the binary
files generated above, so we only need to concatenate them.

```bash
cat $(ls *bin | sort -n -t - -k 2) > <out_memmap_file>
```

## Binning TCs Comparisons

For comparing two sets of pairwise TCs, first use the above approach to build
the two pairwise TCs memmaps. Then, count the number of TCs pairs that map to
specific values at a specified precision (number of decimal places). For a
given memmap pair, run

```bash
python count_pairwise_tcs.py <mmap_file1> <mmap_file2> --names Name1 Name2
```
which produces an output file `name1_name2_tcs.csv.gz`.

## Computing Enrichment Curves

FastROCS output are pairwise shape and combo (shape + color) Tanimotos, as
opposed to ECFP4's and E3FP's individual fingerprints. To compare the
performance of all three approaches at enriching for actives, we perform
*k*-fold cross-validation as [previously described](../crossvalidation), using
as the threshold for fingerprints the maximum pairwise TC computed between the
set of query molecule fingerprints and target molecule fingerprints, and for
ROCS the shape or combo max TC between the query molecule conformers and target
molecule conformers. The enrichment curve plots fraction of actives recovered
(sensitivity) vs fraction of database screened.

The MaxTC "classifier" can either dynamically compute the Tanimotos or read
from a pre-computed memmap (see above). The former is significantly faster for
fingerprints, due to the efficiency of large-scale TC computation.

For fingerprints, run

```bash
python $E3FP_PROJECT/scripts/run_cv.py <molecules_file> $E3FP_PROJECT/data/chembl20_binding_targets.csv.bz2 --method maxtc --reduce_negatives -l cv_log.txt
```

For FastROCS comparisons, run

```bash
python $E3FP_PROJECT/scripts/run_cv.py <molecules_file> $E3FP_PROJECT/data/chembl20_binding_targets.csv.bz2 --method maxtc --tc_files <tcs_memmap_file> <mol_names_file> --reduce_negatives -l cv_log.txt
```

## All-By-All Comparisons of Random Conformers from Different Molecules

See [`random_conformers`](./random_conformers) for specific instructions.

## Pairwise Comparisons of Random Conformers of the Same Molecules

See [`random_conformer_pairs`](./random_conformer_pairs) for specific
instructions.
