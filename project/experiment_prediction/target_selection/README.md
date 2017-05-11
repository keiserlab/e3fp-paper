# Preparing Targets for Searching

* Note: Here we assume that earlier instructions on
[conformer generation](../../conformer_generation) and
[fingerprinting](../../crossvalidation/sea/README.md) have already
been followed, producing ChEMBL20 molecules sets
`e3fp_molecules.csv.bz2` and `ecfp4_molecules.csv.bz2`.


## Create Directory Structure

```bash
mkdir e3fp
mkdir ecfp4
```

## Build SEA Libraries

To search with SEA, the targets and known binding molecules must
be assembled into a library, along with a fit that is used to
compute the *p-value* of any particular match. While this code will
generate a fit automatically, fit files are provided.

### Build the ECFP4 library

```bash
python $E3FP_PROJECT/scripts/targets_map_to_library.py $E3FP_PROJECT/data/available_targets_pdsp_chembl20_map
.tab $E3FP_PROJECT/data/chembl20_binding_targets.csv.bz2 $E3FP_PROJECT/data/ecfp4_molecules.csv.bz2 --fit_file $E3FP_PROJECT/data/ecfp4.fit -o ecfp4 -l ecfp4/lib_log.txt
```

### Build the E3FP library

```bash
python $E3FP_PROJECT/scripts/targets_map_to_library.py $E3FP_PROJECT/data/available_targets_pdsp_chembl20_map
.tab $E3FP_PROJECT/data/chembl20_binding_targets.csv.bz2 $E3FP_PROJECT/data/e3fp_molecules.csv.bz2 --fit_file $E3FP_PROJECT/data/early_params.fit -o e3fp -l e3fp/lib_log.txt
```
