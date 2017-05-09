# Generate Plots/Tables to Compare E3FP and ECFP4 Variants

These scripts generate the paper figures, tables, and miscellaneous metrics
used for comparing E3FP and ECFP4 in the paper. They assume that the
[fingerprint comparison](../../fingerprint_comparison) and
[cross-validation](../../crossvalidation) analyses have already been run.

## Generate the Comparison/Summary Table (Table 1)

```bash
python make_cv_summary_table.py
```

## Generate the Main Text Figure (Figure 1)

```bash
python make_plots.py
```

## Generate All Supporting Figures (Figures S5-S8)

```bash
python make_supp_plots.py
```

## Compute Summary Metrics from Confusion Matrix

Compute SEA p-value threshold, sensitivity, specificity, and precision
corresponding to maximum E3FP
[F1-score](https://en.wikipedia.org/wiki/F1_score) (a common metric for
determining the performance of a classifier at precision and recall) and
determine the same metrics when the same threshold is applied to ECFP4.
Likewise, compute the same metrics at the threshold corresponding to ECFP4's
maximum F1-score.

```bash
python get_confusion_stats.py $E3FP_PROJECT/crossvalidation/sea/e3fp $E3FP_PROJECT/crossvalidation/sea/ecfp4
```
