# Enriching for Actives using PDB Conformers

In this analysis, we seek to enrich for active compound/target pairs using
E3FP fingerprints of bioactive conformers in the PDB and compare this
enrichment with that attainable using ECFP4 and E3FP fingerprints of RDKit-
generated conformers.

*Note: We assume here that [conformer generation](../conformer_generation) of
       ChEMBL20 compounds has already been performed.

## Download Dataset

First, download a copy of the [scPDB](http://cheminfo.u-strasbg.fr/scPDB) dataset,
and place it in `./scPDB`.

## Generate Input Files

Run

```bash
python create_inputs.py
```

## Searching ChEMBL20 Compounds against PDB Targets

Run the following commands:

```bash
python validate.py chembl_ecfp4_mols.csv.bz2 chembl_targets.csv.bz2 pdb_ecfp4_mols.csv.bz2 pdb_targets.csv.bz2 --fit_file ~/e3fp-paper/project/data/ecfp4.fit --out_dir ecfp4_chembl_pdb
python validate.py chembl_e3fp_mols.csv.bz2 chembl_targets.csv.bz2 pdb_crystalconf_e3fp_mols.csv.bz2 pdb_targets.csv.bz2 --fit_file /Users/saxen/e3fp-paper/e3fp_paper/config/best_params.fit --out_dir e3fp_chembl_pdb
python validate.py chembl_e3fp_mols.csv.bz2 chembl_targets.csv.bz2 pdb_rdkitconf_e3fp_mols.csv.bz2 pdb_targets.csv.bz2 --fit_file /Users/saxen/e3fp-paper/e3fp_paper/config/best_params.fit --out_dir e3fp_rdkit_chembl_pdb
```

## Performing 5-fold Cross-validation on PDB compounds

Run the following commands:

```bash
mkdir ecfp4_cv_sea
python $E3FP_PROJECT/scripts/run_cv.py pdb_ecfp4_mols.csv.bz2 pdb_targets.csv.bz2 --out_dir ecfp4_cv_sea -l ecfp4_cv_sea/cv_log.txt
mkdir e3fp_cv_sea
python $E3FP_PROJECT/scripts/run_cv.py pdb_crystalconf_e3fp_mols.csv.bz2 pdb_targets.csv.bz2 --out_dir e3fp_cv_sea -l e3fp_cv_sea/cv_log.txt
mkdir e3fp_rdkit_cv_sea
python $E3FP_PROJECT/scripts/run_cv.py pdb_rdkitconf_e3fp_mols.csv.bz2 pdb_targets.csv.bz2 --out_dir e3fp_rdkit_cv_sea -l e3fp_rdkit_cv_sea/cv_log.txt
```

* Note: these datasets are too small to generate a reliable SEA library fit.
        It is recommended to inject a fit generated using ChEMBL20 into these
        libraries. For a more in-depth explanation, see the section on
        [K-Fold Cross-validation using SEA](../crossvalidation/sea).
