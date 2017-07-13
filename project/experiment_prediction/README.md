# Predicting Novel Binding Pairs with E3FP

The goal of this analysis is to make novel compound-target 
binding predictions using SEA with E3FP that cannot be made
using SEA with the 2D fingerprint ECFP.

## Relevant Data Files
- [`chembl20_zinc_in-stock_mw800_unique_proto.smi.bz2`](../data/chembl20_zinc_in-stock_mw800_unique_proto.smi.bz2): SMILES file
  for all ["in-stock"](http://zinc15.docking.org/substances/subsets/in-stock/)
  compounds in the [ZINC](http://zinc15.docking.org) database, as of 2015-09-24,
  mapped to ChEMBL20
- [`available_targets_pdsp_chembl20_map.tab.bz2`](../data/available_targets_pdsp_chembl20_map.tab.bz2): Table of human, mouse,
  and rat analogs to protein targets for which
  [NIMH PDSP](http://pdspdb.unc.edu/pdspWeb/) has a binding assay, mapped to
  ChEMBL20 target IDs.
- [`early_params.cfg`](../data/early_params.cfg): Parameters chosen early into
  the [optimization](../parameter_optimization) process that were originally
  used for this analysis.

## Generating Fingerprints

See `README` instructions within [`drug_selection`](drug_selection) and
[`target_selection`](target_selection) for ECFP4 and E3FP fingerprint
generation.

## Searching Molecules against Targets

See [`target_drug_search`](target_drug_search).

## Computing All Pairwise Tanimoto Coefficients (TCs)

To compute all pairwise TCs between target-associated molecules and drug
molecules for either E3FP or ECFP4, run
```bash
python ../fingerprint_comparison/get_pairwise_tcs.py <drug_molecules> <target_molecules> --memmap_file <mmap_file>
```

Then, to count the number of TCs pairs that map to specific values at a
specified precision
```bash
python ../fingerprint_comparison/count_pairwise_tcs.py <ecfp4_mmap_file> <e3fp_mmap_file> --names ECFP4 E3FP
```
which produces an output file `ecfp4_e3fp_tcs.csv.gz`.

## Computing Pairwise TCs Between Specific Target/Molecule Pairs

To compute pairwise max TCs for a specific drug/target pair, run:

```bash
python get_mol_vs_target_tcs.py <mol_name> <target_id> <drug_molecules> <target_molecules> <target_targets> --affinity <affinity> --out_file <out_file>
```
