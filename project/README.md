# Project Files and Scripts

All necessary files and scripts to reproduce the work in the E3FP paper are
included here.

## Setup

Be sure that both `e3fp` and `e3fp_paper` packages and all required and
optional dependencies are installed. Then, set the path to this directory to
the environmental variable `$E3FP_PROJECT`, the path to the E3FP package to
the variable `$E3FP_PACKAGE`, and the path to the `e3fp_paper` repo to
`$E3FP_PAPER`.

## Summary

The sections are best understood when browsed in the following order:

1. [Conformer Generation](conformer_generation): Before performing 3D
   fingerprinting, one must generate a conformer library. While a user may
   provide their own conformer library, here we provide instructions for
   generating a library according to the method in the paper.
2. [Parameter Optimization](parameter_optimization): E3FP has several tunable
   parameters, including the input conformer library. Here, we use
   [Spearmint](https://github.com/JasperSnoek/spearmint) to optimize the
   parameter set using *5*-fold cross-validation to determine the optimal
   parameters that produce high-performing fingerprints.
3. [Fingerprint Comparison](fingerprint_comparison): Compare E3FP and ECFP4
   fingerprints with each other and with other 3D similarity approaches.
4. [Benchmark](benchmark): Compare compute times of ECFP4, E3FP, and FastROCS.
5. [Cross-Validation](crossvalidation): E3FP and ECFP4 fingerprints are
   generated, and *k*-fold cross-validation is applied to determine the
   performance of these fingerprints using various classifiers.
6. [PDB Enrichment](pdb_enrichment): Enrich for actives in ChEMBL20 using PDB
   conformations.
7. [Experiment Prediction](experiment_prediction): Having optimized E3FP and
   assessed its performance relative to ECFP4 *in silico*, we here identify
   molecule-target pairs that are predicted by the Similarity Ensemble
   Approach (SEA) to bind with E3FP but not with ECFP4. A subset of these
   predictions were verified experimentally.
8. [Analysis and Figures](analysis): This section contains all scripts
   necessary to generate figures, tables, and statistics that appeared in the
   paper.

The additional [`data`](data) and [`scripts`](scripts) directories contain all
data files and general purpose scripts needed to perform the analyses in these
sections, except where otherwise noted.
