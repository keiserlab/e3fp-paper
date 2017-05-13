# Generate Plots/Table for Experimental Results

These scripts generate the paper figures and table for visualizing results of
binding, agonist, and antagonist assays. They assume that the data files and
curve fits (in GraphPad Prism 5.0 format exported to text file) are in a
folder named `results` (not provided) located in the
[experiment prediction](../../experiment_prediction) section.

## Generate the lower panels of Figure 4

```bash
python make_plots.py
```

## Generate all experimental supporting plots/table

```bash
python make_supp_mat.py
```
