# Searching Molecules against Libraries

## Create Directory Structure

```bash
mkdir e3fp
mkdir ecfp4
```

## Search ECFP4 Molecules

```bash
python $E3FP_PROJECT/scripts/search_library.py ../drug_selection/ecfp4/molecules.csv.bz2 ../target_selection/ecfp4/library.sea --target_results_pickle ecfp4/target_results.pkl.bz2 --mol_results_pickle ecfp4/mol_results.pkl.bz2 -l ecfp4/search_log.txt
```

## Search E3FP Molecules

```bash
python $E3FP_PROJECT/scripts/search_library.py ../drug_selection/e3fp/molecules.csv.bz2 ../target_selection/e3fp/library.sea --target_results_pickle e3fp/target_results.pkl.bz2 --mol_results_pickle e3fp/mol_results.pkl.bz2 -l e3fp/search_log.txt
```

# Filter Results to Unique E3FP Results

Here, we filter the predicted molecule/target pairs
to those which have a SEA *p-value* < 1e-20 using E3FP
against molecules that bind to the target with affinity
of 10 nM or better and a *p-value* > 0.1 using ECFP4
at any affinity level up to 10 uM.

```bash
python analyze_search_results.py
```
