"""Plot all cross-validation results.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from e3fp_paper.plotting.defaults import DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels

fonts = DefaultFonts()


RESULT_TABLE_FILES = ["1_chembl17_rand10000_auprc_fold_results.txt",
                      "2_chembl20_rand100000_aucsum_fold_results.txt",
                      "3_chembl20_rand100000_aucsum_fold_results.txt"]
NUM_MOLS = [10000, 100000, 100000]
MAX_STEPS = [350, 100, 100]
MAX_ECFP4_AUROC = 0.9809
MAX_ECFP4_AUPRC = 0.6323
CV_TRAJECTORY_BASENAME = "fig_s2_v6"
AUC_SCATTER_BASENAME = "fig_s3_v6"
BOXPLOT_BASENAME = "fig_s4_v6"


def fold_df_to_df(df):
    cols = ["Bits", "Conformers", "First", "Level", "Radius Multiplier"]
    df = df.loc[(df[cols].shift() != df[cols]).any(axis=1)]
    df.reset_index(drop=True, inplace=True)
    return df


def fold_df_to_best_fold_df(df):
    cols = ["Bits", "Conformers", "First", "Level", "Radius Multiplier"]
    df = df.groupby(by=cols).max()
    df.reset_index(inplace=True)
    return df


def plot_performance(df, ax, show_legend=False):
    auprc_line, = ax.plot(df.index + 1, df["Mean AUPRC"], color="b",
                          linewidth=1.5, alpha=.8, zorder=2)
    ax.fill_between(df.index + 1, df["Mean AUPRC"] - df["Std AUPRC"],
                    df["Mean AUPRC"] + df["Std AUPRC"], color="b",
                    alpha=.3, zorder=2, linewidth=0)

    auroc_line, = ax.plot(df.index + 1, df["Mean AUROC"], color="r",
                          linewidth=1.5, alpha=.8, zorder=2)
    ax.fill_between(df.index + 1, df["Mean AUROC"] - df["Std AUROC"],
                    df["Mean AUROC"] + df["Std AUROC"], color="r",
                    alpha=.3, zorder=2, linewidth=0)

    max_aucsum_ind = np.argmax(df["AUCSUM"])
    max_aucsum = df["AUCSUM"][max_aucsum_ind]
    aucsum_line, = ax.plot(df.index + 1, df["AUCSUM"], color="k",
                           linewidth=1.5, alpha=.8, zorder=1)
    max_aucsum_line, = ax.plot(df.index + 1,
                               max_aucsum * np.ones_like(df.index), color="k",
                               linestyle='--', linewidth=1.5, alpha=.8,
                               zorder=1)
    ax.set_xlim(0, max(df.index) + 2)
    ax.set_yticks(np.linspace(0, 2, 5))
    ax.set_ylim(0, 2.2)
    ax.set_ylabel("AUC", fontsize=fonts.ax_label_fontsize)

    if show_legend:
        ax.legend(
            (auprc_line, auroc_line, aucsum_line, max_aucsum_line),
            ('Mean AUPRC', 'Mean AUROC', r'$AUC_{SUM}$', r'Max $AUC_{SUM}$'),
            loc='upper center', ncol=4, borderaxespad=0)


def plot_runtime(df, ax, show_legend=False):
    time_line, = ax.plot(df.index + 1, df["Runtime"], linewidth=1.5,
                         color="k", zorder=2)
    mean_runtime = np.mean(df["Runtime"])
    print("Average runtime is {:.2f} CPU hours".format(mean_runtime))
    mean_time_line, = ax.plot(df.index + 1,
                              mean_runtime * np.ones_like(df.index),
                              linewidth=1.5, linestyle='--', color="k",
                              zorder=1)
    ax.set_xlim(0, max(df.index) + 2)
    ax.set_ylim(0, max(df["Runtime"]) * 1.15)
    ax.set_ylabel("CPU Hours", fontsize=fonts.ax_label_fontsize)
    ax.set_xlabel("Iteration", fontsize=fonts.ax_label_fontsize)

    if show_legend:
        ax.legend(
            (time_line, mean_time_line),
            ('Runtime', 'Average Runtime'),
            loc='upper center', ncol=2, borderaxespad=0)


def plot_auc_scatter(df, ax, alpha=1, ref_auc_pair=None, show_legend=True,
                     vmin=None, vmax=None):
    ax.scatter(df["Fold AUROC"], df["Fold AUPRC"], c=df["Fold TC Cutoff"],
               alpha=alpha, marker='.', s=50, vmin=vmin, vmax=vmax)
    if ref_auc_pair is not None:
        ref_line = ax.axvline(ref_auc_pair[0], color='r', linestyle='--')
        ax.axhline(ref_auc_pair[1], color='r', linestyle='--')
        if show_legend:
            ax.legend([ref_line], ["ECFP4 AUCs"], loc=(0, .95), borderpad=0,
                      fontsize=fonts.legend_fontsize)
    ax.set_xlabel("AUROC", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel("AUPRC", fontsize=fonts.ax_label_fontsize)
    ax.tick_params(labelsize=fonts.ax_ticks_fontsize)


def plot_param_boxplot(df, ax, xcol, ycol, xlabel=None, ylabel=None):
    flierprops = dict(marker='o', markersize=4, markerfacecolor="k")
    sns.boxplot(x=xcol, y=ycol, data=df, ax=ax, palette="Blues",
                flierprops=flierprops, whis=np.inf)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def reformat_range_label(x, type=str):
    b, e = x.replace("(", "").replace("]", "").split(", ")
    if type is int:
        b, e = float(b), float(e)
        if b + 1 < e:
            b = int(b + 1)
        else:
            b = int(b)
        e = int(e)
    return "{}-{}".format(b, e)


if __name__ == "__main__":
    final_dfs = []
    # CV trajectory figure
    panel_axes = []
    fig = plt.figure(figsize=(6, 8))
    for i, fn in enumerate(RESULT_TABLE_FILES):
        df = pd.DataFrame.from_csv(fn, sep="\t")
        df.dropna(how='any', axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["Mol Num"] = NUM_MOLS[i]
        if i > 0:
            final_dfs.append(df)
        df = fold_df_to_df(df)
        df = df.iloc[:MAX_STEPS[i]]
        ax = fig.add_subplot(6, 1, 2 * i + 1)
        sns.despine(ax=ax)
        panel_axes.append(ax)

        plot_performance(df, ax, show_legend=True)
        ax = fig.add_subplot(6, 1, 2 * i + 2)
        plot_runtime(df, ax, show_legend=True)
        sns.despine(ax=ax)
    add_panel_labels(fig, axes=panel_axes, xoffset=.055)
    fig.tight_layout(rect=[0, 0, 1, .97], h_pad=0)
    fig.savefig(CV_TRAJECTORY_BASENAME + ".png", dpi=300)
    fig.savefig(CV_TRAJECTORY_BASENAME + ".tif", dpi=300)

    # AUC scatter figure
    df = pd.concat(final_dfs, axis=0)
    df = df.loc[df["Mol Num"] == 100000]
    min_tc_cutoff = df["Fold TC Cutoff"].min()
    max_tc_cutoff = df["Fold TC Cutoff"].max()
    fig = plt.figure(figsize=(7, 2.4))
    ax = fig.add_subplot(131)
    plot_auc_scatter(df, ax, alpha=.8, vmin=min_tc_cutoff,
                     vmax=max_tc_cutoff)
    ax.set_ylim(0.4, .99)
    ax.set_xlim(0.4, .99)
    ax.set_aspect(1)
    sns.despine(ax=ax, offset=10)

    ax = fig.add_subplot(132)
    tmp_df = df[((df["Fold AUROC"] < .95) &
                 (df["Fold AUPRC"] > .5))]
    plot_auc_scatter(tmp_df, ax, alpha=1,
                     ref_auc_pair=(MAX_ECFP4_AUROC, MAX_ECFP4_AUPRC),
                     vmin=min_tc_cutoff, vmax=max_tc_cutoff)
    ax.set_aspect(1. / ax.get_data_ratio())
    sns.despine(ax=ax, offset=10)

    ax = fig.add_subplot(133)
    tmp_df = df[df["Fold AUROC"] > .95]
    plot_auc_scatter(tmp_df, ax, alpha=1,
                     ref_auc_pair=(MAX_ECFP4_AUROC, MAX_ECFP4_AUPRC),
                     vmin=min_tc_cutoff, vmax=max_tc_cutoff)
    ax.set_aspect(1. / ax.get_data_ratio())
    sns.despine(ax=ax, offset=10)

    add_panel_labels(fig, xoffset=.32)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(AUC_SCATTER_BASENAME + ".png", dpi=300)
    fig.savefig(AUC_SCATTER_BASENAME + ".tif", dpi=300)

    linear_df = df[df["Fold AUROC"] > .95].copy()
    best_df = fold_df_to_best_fold_df(linear_df)
    best_df["Rad_Cat"] = pd.cut(best_df["Radius Multiplier"],
                                bins=np.linspace(1.6, 2.8, 5))
    best_df["Rad_Cat"] = [reformat_range_label(x) for x in best_df["Rad_Cat"]]
    best_df["First_Cat"] = pd.cut(best_df["First"],
                                  bins=np.linspace(0.9, 35, 9))
    best_df["First_Cat"] = [reformat_range_label(x, type=int)
                            for x in best_df["First_Cat"]]

    col_names = ["Rad_Cat", "Level", "First_Cat"]
    prop_names = ["Radius Multiplier", "Fingerprinting Level",
                  "Number of First Conformers"]
    fig = plt.figure(figsize=(6.25, 7))
    for i, (col_name, prop_name) in enumerate(zip(col_names, prop_names)):
        ax = fig.add_subplot(3, 2, 2 * i + 1)
        plot_param_boxplot(best_df, ax, col_name, "Fold AUPRC",
                           xlabel=prop_name, ylabel="AUPRC")
        sns.despine(ax=ax, offset=10)

        ax = fig.add_subplot(3, 2, 2 * i + 2)
        plot_param_boxplot(best_df, ax, col_name, "Fold AUROC",
                           xlabel=prop_name, ylabel="AUROC")
        sns.despine(ax=ax, offset=10)

    add_panel_labels(fig, xoffset=.23)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(BOXPLOT_BASENAME + ".png", dpi=300)
    fig.savefig(BOXPLOT_BASENAME + ".tif", dpi=300)
