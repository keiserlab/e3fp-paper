# Plotting Results of Parameter Optimization

This assumes that [parameter optimization](../../parameter_optimization) has
already been performed. To parse all log files and summarize fold results,
run:

```bash
python get_cv_results.py $E3FP_PROJECT/parameter_optimization/1_chembl20_opt 2_chembl20_rand100000_aucsum_fold_results.txt
python get_cv_results.py $E3FP_PROJECT/parameter_optimization/2_chembl20_opt 3_chembl20_rand100000_aucsum_fold_results.txt
```

Subsequently to generate all plots, run:

```bash
python plot_cv_performance.py
```

* Note: this script expects an additional summary file, corresponding to a run
  with a small subset of ChEMBL17. These files are not included but can be
  generated.