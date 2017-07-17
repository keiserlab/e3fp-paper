"""Plot benchmark results and save table.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from __future__ import division
import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from e3fp_paper.plotting.defaults import DefaultColors, DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels


fonts = DefaultFonts()
def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
BENCHMARK_DIR = os.path.join(PROJECT_DIR, "benchmark")
FPRINT_TIMES_FILE = os.path.join(BENCHMARK_DIR, "time_values.txt")
CONFGEN_LOG_FILE = os.path.join(BENCHMARK_DIR, "confgen_log.txt")
ROCS_TCS_LOG_FILES = glob.glob(os.path.join(BENCHMARK_DIR, "fastrocs_log*"))
SCALED_ROCS_TCS_LOG_FILES = glob.glob(
    os.path.join(os.environ['E3FP_PROJECT'],
                 'fingerprint_comparison', "fastrocs_log*"))
ECFP_TCS_LOG_FILES = [os.path.join(BENCHMARK_DIR, "ecfp_pairwise_tc.log")]
E3FP_TCS_LOG_FILES = [os.path.join(BENCHMARK_DIR, "e3fp_pairwise_tc.log")]

OUT_IMAGE_BASENAME = "benchmark_plot_v2"
OUT_TABLE_BASENAME = "total_times_v2"
MIN_MOLS = 5
SCALE_FROM = 10000
SCALE_TO = 308315


def gpu_secs_to_cpu_secs(s):
    return 824.9 * s


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
    ax.set_xlabel('Number of Heavy Atoms',
                  fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)


def get_time_from_line(line):
    line = line.lstrip('[')
    if '2017-' in line:
        line = '2017-' + line.split('2017-')[1]
    time_str = line.split('|')[0]
    time_str, ms = time_str.split(',')
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return time + timedelta(milliseconds=int(ms))


def get_runtime_stats_from_fprint_log(fn):
    with open(fn, "r") as f:
        start_times = {}
        mol_runtimes = {}
        for line in f:
            line = line.rstrip()
            if 'Generating' in line:
                name = line.split(" for ")[1].split('.')[0]
                try:
                    time = get_time_from_line(line)
                except:
                    print(line)
                start_times[name] = time
            elif 'Generated' in line:
                name = line.split(" for ")[1].split('.')[0]
                try:
                    time = get_time_from_line(line)
                    mol_runtimes[name] = (
                        time - start_times[name]).total_seconds()
                except:
                    continue
    return mol_runtimes


def get_runtime_stats_from_pairwise_log(fns):
    total_time = 0
    for fn in fns:
        start_time = None
        end_time = None
        with open(fn, "r") as f:
            for line in f:
                if 'Will save' in line:
                    start_time = get_time_from_line(line)
                if 'Ending' in line:
                    end_time = get_time_from_line(line)
                    break
            if end_time is None:
                end_time = get_time_from_line(line)
        dtime = (end_time - start_time).total_seconds()
        total_time += dtime
    return total_time


if __name__ == "__main__":
    confgen_runtimes = get_runtime_stats_from_fprint_log(CONFGEN_LOG_FILE)

    cols = ['ECFP4 Time', 'ConfGen Time', 'E3FP Time', 'E3FP Total Time']
    titles = ['ECFP4 Fingerprinting', 'Conformer Generation',
              'E3FP Fingerprinting', 'E3FP Total']
    df = pd.DataFrame.from_csv(FPRINT_TIMES_FILE, sep="\t", index_col=None)
    df['ConfGen Time'] = [confgen_runtimes.get(x, np.nan) for x in df['Name']]
    df['E3FP Total Time'] = df['ConfGen Time'] + df['E3FP Time']
    df.sort_values(by='Num Heavy', inplace=True)
    comp_df = df.dropna(axis=0, how='any')
    mean_df, std_df = get_stats_df(comp_df, min_mols=MIN_MOLS)

    fig = plt.figure(figsize=(4.4, 4.4))
    for i, col in enumerate(cols):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_times(mean_df, col, ax=ax, std_df=std_df, title=titles[i])
        sns.despine(ax=ax, offset=10)

    fig.tight_layout(rect=[0, 0, 1, .97])
    add_panel_labels(fig=fig, xoffset=.24)
    fig.savefig(OUT_IMAGE_BASENAME + ".png", dpi=300)
    fig.savefig(OUT_IMAGE_BASENAME + ".tif", dpi=300)

    linear_cols = ['ECFP4 Time', 'ConfGen Time', 'E3FP Time',
                   'E3FP Total Time']
    expon_cols = ['ECFP4 Pairwise', 'E3FP Pairwise',
                  'ROCS Pairwise (GPU)', 'ROCS Pairwise (CPU)']
    total_df = (comp_df[linear_cols].sum() * SCALE_FROM /
                float(comp_df.shape[0]))
    total_df['ECFP4 Pairwise'] = get_runtime_stats_from_pairwise_log(
        ECFP_TCS_LOG_FILES)
    total_df['E3FP Pairwise'] = get_runtime_stats_from_pairwise_log(
        E3FP_TCS_LOG_FILES)
    total_df['ROCS Pairwise (GPU)'] = get_runtime_stats_from_pairwise_log(
        ROCS_TCS_LOG_FILES)
    total_df['ROCS Pairwise (CPU)'] = gpu_secs_to_cpu_secs(
        total_df['ROCS Pairwise (GPU)'])

    scaled_total_df = total_df.copy()
    scaled_total_df[linear_cols] = (
        scaled_total_df[linear_cols] * SCALE_TO / SCALE_FROM)
    scaled_total_df[expon_cols] = scaled_total_df[expon_cols] * (
        SCALE_TO * (SCALE_TO - 1)) / (SCALE_FROM * (SCALE_FROM - 1))
    scaled_total_df['ROCS Pairwise (GPU)'] = get_runtime_stats_from_pairwise_log(
        SCALED_ROCS_TCS_LOG_FILES)
    scaled_total_df['ROCS Pairwise (CPU)'] = gpu_secs_to_cpu_secs(
        scaled_total_df['ROCS Pairwise (GPU)'])

    total_df = pd.concat([total_df, scaled_total_df], axis=1)
    total_df.columns = ['10,000 Molecules', '308,315 Molecules']
    total_df.to_csv(OUT_TABLE_BASENAME + ".txt", float_format='%.1f', sep="\t")
