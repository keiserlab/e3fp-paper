# K-fold Cross-validation Using Mean Fingerprints

Here we pre-process the fingerprints before running cross-
validation, by computing the mean of all fingerprints for
a molecule to form a 'float' fingerprint.

## Running 5-fold Cross-Validation

Within each directory, run cross-validation with the following command,
replacing `<molecules_file>` with `../sea/e3fp/molecules.csv.bz2` for E3FP
or `../sea/ecfp4/molecules.csv.bz2` for ECFP4 and replacing `<cv_method>`
with the appropriate argument for `--method` ('nb', 'rf', 'linsvm', or 'nn'):

```bash
python $E3FP_PROJECT/scripts/run_cv.py <molecules_file> $E3FP_PROJECT/data/chembl20_binding_targets.csv.bz2 --reduce_negatives -l cv_log.txt --method <cv_method> --process_inputs mean
```
