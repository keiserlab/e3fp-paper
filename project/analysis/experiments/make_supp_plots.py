"""Make supplementary TCs plots for experimental pairs

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from python_utilities.scripting import setup_logging
from e3fp_paper.plotting.defaults import DefaultColors, DefaultFonts
from e3fp_paper.plotting.util import add_panel_labels
from e3fp_paper.plotting.comparison import plot_tc_hists, plot_tc_heatmap

fonts = DefaultFonts()
def_colors = DefaultColors()

PROJECT_DIR = os.environ['E3FP_PROJECT']
EXP_BASEDIR = os.path.join(PROJECT_DIR, 'experiment_prediction')
TC_COUNTS_FILE = os.path.join(EXP_BASEDIR, "ecfp_e3fp_tcs_counts.csv.bz2")
MOL_TARGET_PAIRS = {("Alphaprodine", "M3"), ("Anpirtoline", "a4b2"),
                    ("Cypenamine", "a2b4")}
TARGET_NAMES = {"M3": r"M3", "a4b2": r"$\alpha 4 \beta 2$",
                "a2b4": r"$\alpha 2 \beta 4$"}
OUT_IMAGE_BASENAME = "tcs_supp"
ECFP_TC_THRESH = .3
E3FP_TC_THRESH = .18
setup_logging(verbose=True)


if __name__ == "__main__":
    panel_axes = []
    panel_xoffsets = []
    fig = plt.figure(figsize=(6.4, 5))
    gs = gridspec.GridSpec(5, 6)
    tc_counts_df = pd.DataFrame.from_csv(TC_COUNTS_FILE, sep="\t",
                                         index_col=None)
    ax = plt.subplot(gs[:3, :3])
    plot_tc_heatmap(tc_counts_df, ax=ax, outliers_df=None,
                    cols=["ECFP4", "E3FP"],
                    thresholds=[ECFP_TC_THRESH, E3FP_TC_THRESH])
    # plot_tc_scatter(tc_examples_df, ax)
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.19)

    ax = plt.subplot(gs[:3, 3:])
    plot_tc_hists(tc_counts_df, cols=["ECFP4", "E3FP"], ax=ax,
                  colors=[def_colors.ecfp_color, def_colors.e3fp_color],
                  thresholds=[ECFP_TC_THRESH, E3FP_TC_THRESH])
    sns.despine(ax=ax, offset=10)
    panel_axes.append(ax)
    panel_xoffsets.append(.19)

    for i, (mol_name, target_name) in enumerate(sorted(MOL_TARGET_PAIRS)):
        ax = plt.subplot(gs[3:, 2 * i:2 * i + 2])
        panel_axes.append(ax)
        panel_xoffsets.append(.20)
        basename = "{}_{}".format(mol_name.lower(), target_name.lower())
        names = ["ECFP4", "E3FP"]
        values = []
        for name in names:
            fn = os.path.join(
                EXP_BASEDIR, basename + "_{}_maxtcs.npz".format(name.lower()))
            data = np.load(fn)
            values.append(data['arr_0'])
            data.close()
        df = pd.DataFrame(data=np.asarray(values, np.double).T, columns=names)
        max_tc = df.max().max()
        df['Count'] = np.ones(df.shape[0])
        plot_tc_hists(df, cols=names, ax=ax,
                      colors=[def_colors.ecfp_color, def_colors.e3fp_color],
                      thresholds=[ECFP_TC_THRESH, E3FP_TC_THRESH],
                      logscale=False, show_legend=True,
                      legend_fontsize=fonts.legend_fontsize - 2,
                      title=r"{} vs {}".format(mol_name,
                                               TARGET_NAMES[target_name]))
        ax.set_xlim(0, max([max_tc, ECFP_TC_THRESH, ECFP_TC_THRESH]) * 1.2)

    add_panel_labels(axes=panel_axes, xoffset=panel_xoffsets, )
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(OUT_IMAGE_BASENAME + ".png", dpi=300)
    fig.savefig(OUT_IMAGE_BASENAME + ".tif", dpi=300)
