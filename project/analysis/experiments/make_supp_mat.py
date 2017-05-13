"""Create supporting info experimental curves and binding summary table.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import math
from collections import Counter

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from e3fp_paper.plotting.defaults import DefaultColors
from e3fp_paper.plotting.experiments import data_df_from_file, \
                                            fit_df_from_file, \
                                            plot_experiments, \
                                            plot_schild
from e3fp_paper.plotting.util import add_panel_labels


RESULTS_DIR = "../../experiment_prediction/results"
BINDING_POOL_DATA_FILES = glob.glob(os.path.join(RESULTS_DIR,
                                                 "*binding*/*pool*data.txt"))
BINDING_DATA_FILES = [x for x in glob.glob(os.path.join(RESULTS_DIR,
                                                        "*binding*/*data.txt"))
                      if x not in BINDING_POOL_DATA_FILES]
TANGO_POOL_DATA_FILES = glob.glob(os.path.join(RESULTS_DIR,
                                               "*tango*/*pool*data.txt"))
ANTAGONIST_POOL_DATA_FILES = glob.glob(
    os.path.join(RESULTS_DIR, "*antagonist*/*pool*data.txt"))
BINDING_STRING_MATCH = {'muscarinic': 'm'}
BINDING_BASENAMES = {'muscarinic': 'fig_s9'}
BINDING_TABLE_FILE = 'table_s7.txt'
TANGO_BASENAME = "fig_s10"
ANTAGONIST_BASENAMES = {"m5": "fig_s11"}
TARGET_NAME_FORMAT = {'m1': r"$M_1$", 'm2': r"$M_2$", 'm3': r"$M_3$",
                      'm4': r"$M_4$", 'm5': r"$M_5$"}
REF_MOLS = {'Atropine', 'Carbachol', 'Acetylcholine'}
NAME_DF = pd.DataFrame.from_csv(
    os.path.join(RESULTS_DIR, "compound_name_map.txt"), sep="\t",
    header=-1)
NAME_MAP = {str(k): v for k, v in NAME_DF.to_dict().values()[0].iteritems()}

colors = DefaultColors()


def sort_columns(df, ref_mols={}):
    if df is None:
        return
    col_inds = zip(*sorted(enumerate(df.columns),
                           key=lambda x: (x[1] not in REF_MOLS,
                                          x[1].split(' ')[0], x[0])))[0]
    return df.iloc[:, col_inds]


def convert_columns_to_type(df, astype=float):
    cols = []
    for x in df.columns:
        try:
            col = astype(x)
        except Exception:
            col = x
        cols.append(col)
    df.columns = cols
    return df


def read_files(data_files, expect_one=False, sort_cols=True):
    exp_prefixes = [x.split('_data')[0] for x in data_files]
    fit_files = [x + "_fit.txt" for x in exp_prefixes]
    target_data_fit = {}
    for prefix, data_file, fit_file in zip(exp_prefixes, data_files,
                                           fit_files):
        prefix = os.path.basename(prefix)
        tmp = prefix.split("_")
        target_name = tmp[0]
        try:
            exp_name = tmp[1]
        except IndexError:
            exp_name = ""
        data_df = data_df_from_file(data_file, name_map=NAME_MAP)
        fit_df = fit_df_from_file(fit_file, name_map=NAME_MAP)
        if sort_cols:
            data_df = sort_columns(data_df, ref_mols=REF_MOLS)
        if expect_one:
            assert(target_name not in target_data_fit)
            target_data_fit[target_name] = (exp_name, data_df, fit_df)
        else:
            target_data_fit.setdefault(target_name, []).append(
                (exp_name, data_df, fit_df))
    return target_data_fit


def count_replicates(data_dfs):
    all_counts = {}
    for df in data_dfs:
        counts = Counter(df.columns)
        for name, count in counts.items():
            name = name.split()[0]
            all_counts.setdefault(name, []).append(count)
    return all_counts


def compute_exp_statistics(data_dfs, fit_dfs, pool_fit_df=None, precision=4):
    d = {}
    for name, rep_counts in count_replicates(data_dfs).items():
        d.setdefault("Compound", []).append(name)
        rep_counts = sorted(rep_counts)
        exp_count = len(rep_counts)
        exp_count_str = "{:d} ({:s})".format(exp_count,
                                             ", ".join(map(str, rep_counts)))
        d.setdefault("Experiment Number", []).append(exp_count_str)

    logKis = {}
    for df in fit_dfs:
        df = df.loc["Best-fit values"]
        for col in df.columns:
            if 'Global' in col:
                continue
            name = col.split()[0]
            logKi = df[col].loc["logKi"]
            logKis.setdefault(name, []).append(logKi)
    for name in d["Compound"]:
        try:
            vals = logKis[name]
            logKi_mean, logKi_std = np.mean(vals), np.std(vals)
            logKi_str = "{1:.{0:d}f} +/- {2:.{0:d}f}".format(
                precision, logKi_mean, logKi_std)
        except KeyError:
            logKi_str = np.nan
        logKi_mean, logKi_std = np.mean(vals), np.std(vals)
        d.setdefault("Mean LogKi", []).append(logKi_str)

    df = pd.DataFrame.from_dict(d)
    if pool_fit_df is not None:
        pooled_logKis = []
        pooled_logIC50s = []
        pooled_Kis = []
        for name in df['Compound']:
            if name not in pool_fit_df.columns:
                pooled_logIC50s.append(np.nan)
                pooled_logKis.append(np.nan)
                pooled_Kis.append(np.nan)
                continue
            try:
                logKi = float(
                    pool_fit_df[name].loc['Best-fit values'].loc['logKi'])
                logKi_std = float(
                    pool_fit_df[name].loc['Std. Error'].loc['logKi'])
                pooled_logKis.append("{1:.{0:d}f} +/- {2:.{0:d}f}".format(
                    precision, logKi, logKi_std))
                pooled_Kis.append(10**logKi * 1e9)
                pooled_logIC50s.append(np.nan)
            except KeyError:
                logIC50 = float(
                    pool_fit_df[name].loc['Best-fit values'].loc['LogIC50'])
                logIC50_std = float(
                    pool_fit_df[name].loc['Std. Error'].loc['LogIC50'])
                pooled_logIC50s.append("{1:.{0:d}f} +/- {2:.{0:d}f}".format(
                    precision, logIC50, logIC50_std))
                pooled_logKis.append(np.nan)
                pooled_Kis.append(np.nan)
        df['Pooled LogKi'] = pooled_logKis
        df['Pooled Ki (nM)'] = pooled_Kis
        df['Pooled LogIC50'] = pooled_logIC50s
    return df


if __name__ == "__main__":
    # Read binding data
    binding_data = read_files(BINDING_DATA_FILES)
    pooled_binding_data = read_files(BINDING_POOL_DATA_FILES, expect_one=True)

    # Save binding pooled table
    exp_stats_dfs = []
    for target in pooled_binding_data:
        data_dfs = [x[1] for x in binding_data[target]]
        fit_dfs = [x[2] for x in binding_data[target]]
        exp_stats_df = compute_exp_statistics(
            data_dfs, fit_dfs, pool_fit_df=pooled_binding_data[target][2])
        exp_stats_df['Target'] = target
        exp_stats_dfs.append(exp_stats_df)
    exp_stats_df = pd.concat(exp_stats_dfs, axis=0)
    exp_stats_df = exp_stats_df[['Target', 'Compound', 'Experiment Number',
                                 'Mean LogKi', 'Pooled LogKi',
                                 'Pooled Ki (nM)', 'Pooled LogIC50']]
    exp_stats_df.sort_values(by=['Target', 'Compound'], inplace=True)
    exp_stats_df.set_index(['Target', 'Compound'], inplace=True)
    exp_stats_df.to_csv(BINDING_TABLE_FILE, sep='\t')

    # Plot radioligand assays
    for target_family, target_sub in BINDING_STRING_MATCH.items():
        targets = sorted([x for x in pooled_binding_data if target_sub in x])
        if len(targets) == 0:
            continue
        num_subplots = len(targets)
        num_rows = math.ceil(num_subplots / 2.)
        fig = plt.figure(figsize=(5.8, 2.5 * num_rows))
        for i, target in enumerate(targets):
            exp_name, data_df, fit_df = pooled_binding_data[target]
            if np.nanmax(np.abs(data_df)) > 150:
                normalize = True
            else:
                normalize = False
            ax = fig.add_subplot(num_rows, 2, i + 1)
            title = TARGET_NAME_FORMAT.get(target, target)
            plot_experiments(data_df, ax, fit_df=fit_df.loc['Best-fit values'],
                             colors_dict=colors.mol_colors,
                             invert=True, normalize=normalize, title=title,
                             ylabel="Specific Binding (%)")
        sns.despine(fig=fig, offset=10)
        add_panel_labels(fig=fig, xoffset=.21)
        fig.tight_layout()
        fig.savefig(BINDING_BASENAMES[target_family] + ".png", dpi=300)
        fig.savefig(BINDING_BASENAMES[target_family] + ".tif", dpi=300)

    # Plot Tango results
    target_data_fit = read_files(TANGO_POOL_DATA_FILES, expect_one=True)
    num_subplots = len(target_data_fit)
    num_rows = math.ceil(num_subplots / 2.)
    fig = plt.figure(figsize=(7, 3 * num_rows))
    for i, (target, (exp_name, data_df, fit_df)) in enumerate(
            sorted(target_data_fit.items())):
        if np.nanmax(np.abs(data_df)) > 150:
            normalize = True
        else:
            normalize = False
        ax = fig.add_subplot(num_rows, 2, i + 1)
        title = TARGET_NAME_FORMAT.get(target, target)
        try:
            fit_df = fit_df.loc['Best-fit values']
        except AttributeError:
            fit_df = None
        plot_experiments(data_df, ax, fit_df=fit_df,
                         colors_dict=colors.mol_colors, invert=False,
                         normalize=normalize, title=title,
                         ylabel="Relative Response (%)")
    sns.despine(fig=fig, offset=10)
    add_panel_labels(fig=fig)
    fig.tight_layout()
    fig.savefig(TANGO_BASENAME + ".png", dpi=300)
    fig.savefig(TANGO_BASENAME + ".tif", dpi=300)

    # Plot antagonist results
    target_data_fit = read_files(ANTAGONIST_POOL_DATA_FILES, sort_cols=False)
    for target, dfs_list in target_data_fit.items():
        num_subplots = len(dfs_list)
        num_rows = 2 * math.ceil(num_subplots / 2.)
        fig = plt.figure(figsize=(7, 3 * num_rows))
        for i, (exp_name, data_df, fit_df) in enumerate(dfs_list):
            convert_columns_to_type(data_df, float)
            convert_columns_to_type(fit_df, float)
            unique_cols = sorted(set(data_df.columns))
            colors_dict = dict(zip(
                unique_cols,
                np.tile(np.linspace(0, .75, len(unique_cols)), (3, 1)).T))
            ax = fig.add_subplot(num_rows, 2, 2 * i + 1)
            if "ACh" in exp_name:
                title = "Against Acetylcholine"
            elif "CCh" in exp_name:
                title = "Against Carbachol"

            try:
                fit_df = fit_df.loc['Best-fit values']
            except AttributeError:
                fit_df = None
            plot_experiments(data_df, ax, fit_df=fit_df,
                             colors_dict=colors_dict, invert=False,
                             normalize=True, title=title,
                             ylabel="Relative Activity (%)")

            ax = fig.add_subplot(num_rows, 2, 2 * i + 2)
            plot_schild(fit_df, ax)

        add_panel_labels(fig=fig)
        sns.despine(fig=fig, offset=10)
        fig.tight_layout()
        fig.savefig(ANTAGONIST_BASENAMES[target] + ".png", dpi=300)
        fig.savefig(ANTAGONIST_BASENAMES[target] + ".tif", dpi=300)
