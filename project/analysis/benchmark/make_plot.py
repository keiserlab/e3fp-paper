"""Plot benchmark results and save table.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from e3fp_paper.plotting.defaults import DefaultColors, DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels


fonts = DefaultFonts()
def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
FPRINT_TIMES_FILE = os.path.join(PROJECT_DIR, "benchmark", "time_values.txt")
ROCS_LOG_FILE = "fastrocs_log.txt"
OUT_IMAGE_BASENAME = "benchmark_plot"
OUT_TABLE_BASENAME = "total_times"
MIN_MOLS = 10


def get_stats_df(df, min_mols=0):
    grp = df.groupby(by='Num Rot')
    mean_df, std_df, count_df = grp.mean(), grp.std(), grp.count()
    counts = count_df[count_df.columns[0]]
    return mean_df.loc[counts >= min_mols], std_df[counts >= min_mols]


def plot_times(mean_df, ycol, ax=None, std_df=None, title=""):
    x, y = mean_df.index, mean_df[ycol]
    if std_df is None:
        ymax = y.max()
        ax.plot(x, y)
    else:
        yerr = std_df[ycol]
        ymax = (y + yerr).max()
        ax.plot(x, y)
        ax.fill_between(x, y + yerr, y - yerr, alpha=.3)
    ax.set_ylim(-ymax * .05, ymax * 1.05)
    ax.set_xlim(0, x.max())
    ax.set_ylabel('CPU Time (s)', fontsize=fonts.ax_label_fontsize)
    ax.set_xlabel('Number of Rotatable Bonds',
                  fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)


def get_time_from_line(line):
    time_str = line.split('|')[0].split(','[0])[0]
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return time


def get_runtime_stats_from_rocs_log(fn):
    with open(fn, "r") as f:
        oldtime = None
        runtimes = []
        for i, line in enumerate(f):
            time = get_time_from_line(line)
            if oldtime is None:
                oldtime = time
                continue

            try:
                db_size = int(
                    line.split(" molecules recorded")[0].split("|")[-1])
            except:
                oldtime = time
                continue

            dtime = (time - oldtime).total_seconds()
            rt = dtime / float(db_size)
            runtimes.append(rt)
            oldtime = time

        return np.mean(runtimes), np.std(runtimes)


if __name__ == "__main__":
    cols = ['ECFP4 Time', 'ConfGen Time', 'E3FP Time', 'E3FP Total Time']
    titles = ['ECFP4 Fingerprinting', 'Conformer Generation',
              'E3FP Fingerprinting', 'E3FP Total']
    df = pd.DataFrame.from_csv(FPRINT_TIMES_FILE, sep="\t", index_col=None)
    df['E3FP Total Time'] = df['ConfGen Time'] + df['E3FP Time']
    df.sort_values(by='Num Rot', inplace=True)
    mean_df, std_df = get_stats_df(df, min_mols=MIN_MOLS)

    fig = plt.figure(figsize=(6.4, 6.4))
    for i, col in enumerate(cols):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_times(mean_df, col, ax=ax, std_df=std_df, title=titles[i])
        sns.despine(ax=ax, offset=10)

    fig.tight_layout(rect=[0, 0, 1, .97])
    add_panel_labels(fig=fig, xoffset=.24)
    fig.savefig(OUT_IMAGE_BASENAME + ".png", dpi=300)
    fig.savefig(OUT_IMAGE_BASENAME + ".tif", dpi=300)

    mean_rocs_time, std_rocs_time = get_runtime_stats_from_rocs_log(
        ROCS_LOG_FILE)
    total_df = df[cols].T
    total_mean = total_df.mean(axis=1)
    total_std = total_df.std(axis=1)
    with open(OUT_TABLE_BASENAME + ".txt", "w") as f:
        f.write("Method\tTime (s)\n")
        for i, col in enumerate(cols):
            f.write("{}\t{:.4f} +/- {:.4f}\n".format(
                col, total_mean[i], total_std[i]))
        f.write("ROCS Pairwise*\t{:.4g} +/- {:.4g}\n".format(
            mean_rocs_time, std_rocs_time))
        f.write("* Per molecule pairwise comparison")
