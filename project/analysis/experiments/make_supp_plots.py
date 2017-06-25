"""Make main text E3FP/ECFP comparison/crossvalidation figure.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from python_utilities.scripting import setup_logging
from e3fp_paper.plotting.defaults import DefaultColors, DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels
from e3fp_paper.plotting.comparison import plot_tc_scatter, plot_tc_hists, \
                                           plot_tc_heatmap

fonts = DefaultFonts()
def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
EXP_BASEDIR = os.path.join(PROJECT_DIR, 'experiment_prediction')
TC_COUNTS_FILE = os.path.join(EXP_BASEDIR, "ecfp_e3fp_tcs_counts.csv.bz2")
OUT_IMAGE_BASENAME = "tcs_supp"
ECFP_TC_THRESH = .3
E3FP_TC_THRESH = .18
setup_logging(verbose=True)


if __name__ == "__main__":
    panel_axes = []
    panel_xoffsets = []
    fig = plt.figure(figsize=(6.4, 3))
    tc_counts_df = pd.DataFrame.from_csv(TC_COUNTS_FILE, sep="\t",
                                         index_col=None)
    ax = fig.add_subplot(121)
    plot_tc_heatmap(tc_counts_df, ax=ax, outliers_df=None,
                    cols=["ECFP4", "E3FP"],
                    thresholds=[ECFP_TC_THRESH, E3FP_TC_THRESH])
    # plot_tc_scatter(tc_examples_df, ax)
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.23)

    ax = fig.add_subplot(122)
    plot_tc_hists(tc_counts_df, cols=["ECFP4", "E3FP"], ax=ax,
                  colors=[def_colors.ecfp_color, def_colors.e3fp_color],
                  thresholds=[ECFP_TC_THRESH, E3FP_TC_THRESH])
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.20)

    add_panel_labels(axes=panel_axes, xoffset=panel_xoffsets)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(OUT_IMAGE_BASENAME + ".png", dpi=300)
    fig.savefig(OUT_IMAGE_BASENAME + ".tif", dpi=300)
