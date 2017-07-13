# All-By-All Comparisons of Random Conformers from Different Molecules

## Getting Random Subset

First, we need to select 1,000 random conformers. Assuming we have already
generated conformers and performed E3FP fingerprinting, run

```bash
python get_random_conformers <sdf_dir> <e3fp_mol_file> --num_confs 1000
```

## Computing Pairwise Tanimoto Coefficients (TCs)

See instructions in [above section](../) for computing pairwise E3FP TCs and
FastROCS shape and combo TCs.

## Computing Pairwise RMSDs

The following script computes the maximum common substructure (MCS) between each
conformer pair and the RMSD between the pairs of atoms in the MCS.

```bash
python ../get_rmsd.py random_conformers.sdf.bz2
```

The output files may be concatenated, and pairwise comparisons may be computed
using the instructions in the [above section](../).
