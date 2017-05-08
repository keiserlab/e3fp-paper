# Generate Plots/Tables to Compare E3FP and ECFP4 Variants

These scripts generate the paper figures and tables
used for comparing E3FP and ECFP4 in the paper. They
assume that the
[fingerprint comparison](../../fingerprint_comparison)
and [cross-validation](../../crossvalidation) analyses
have already been run.

## Generate the Comparison/Summary Table (Table 1)

```bash
python make_cv_summary_table.py
```

## Generate the Main Text Figure (Figure 1)

```bash
python make_plots.py
```

## Generate all Supporting Figures (Figures S5-S8)

```bash
python make_supp_plots.py
```
