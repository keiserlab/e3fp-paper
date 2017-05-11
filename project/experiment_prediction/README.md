# Predicting Novel Binding Pairs with E3FP

The goal of this analysis is to make novel compound-target 
binding predictions using SEA with E3FP that cannot be made
using SEA with the 2D fingerprint ECFP.

## Relevant Data Files
- [`chembl20_zinc_in-stock_mw800_unique_proto.smi.bz2`](../data/chembl20_zinc_in-stock_mw800_unique_proto.smi.bz2): SMILES file
for all ["in-stock"](http://zinc15.docking.org/substances/subsets/in-stock/) compounds in the [ZINC](http://zinc15.docking.org) database, as of 2015-09-24, mapped to ChEMBL20
- [`available_targets_pdsp_chembl20_map.tab.bz2`](../data/available_targets_pdsp_chembl20_map.tab.bz2): Table of human, mouse,
and rat analogs to protein targets for which [NIMH PDSP](http://pdspdb.unc.edu/pdspWeb/) has a binding assay, mapped to ChEMBL20 target IDs.
- [`early_params.cfg`](../data/early_params.cfg): Parameters chosen early into the [optimization](../parameter_optimization) process that were originally used for this analysis.

## Generating Fingerprints

See `README` instructions within [`drug_selection`](drug_selection) and [`target_selection`](target_selection) for ECFP4 and E3FP fingerprint generation.

## Searching Molecules against Targets

See [`target_drug_search`](target_drug_search).
