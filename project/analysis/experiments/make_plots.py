"""Create main text experimental curves.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from e3fp_paper.plotting.defaults import DefaultColors
from e3fp_paper.plotting.experiments import data_df_from_file, \
                                            fit_df_from_file, \
                                            plot_experiments
from e3fp_paper.plotting.util import add_panel_labels

RESULTS_DIR = "../../experiment_prediction/results"
M5_BASENAME = "M1-5_binding/m5_pooled"
A2B4_BASENAME = "a2-4b4_binding/a2b4_pooled"
A3B4_BASENAME = "a2-4b4_binding/a3b4_pooled"
A4B4_BASENAME = "a2-4b4_binding/a4b4_pooled"
FIG_PREFIX = "fig_4b-e"

NAME_DF = pd.DataFrame.from_csv(
    os.path.join(RESULTS_DIR, "compound_name_map.txt"), sep="\t", header=-1)
NAME_MAP = {str(k): v for k, v in NAME_DF.to_dict().values()[0].iteritems()}
TARGET_NAMES = [r"$M_5$", r"$\alpha 2 \beta 4$", r"$\alpha 3 \beta 4$",
                r"$\alpha 4 \beta 4$"]

colors = DefaultColors()

basenames = [M5_BASENAME, A2B4_BASENAME, A3B4_BASENAME, A4B4_BASENAME]
data_files = [os.path.join(RESULTS_DIR, x) + "_data.txt" for x in basenames]
fit_files = [os.path.join(RESULTS_DIR, x) + "_fit.txt" for x in basenames]
fig = plt.figure(figsize=(5, 8))
for i, (name, data_file, fit_file) in enumerate(zip(TARGET_NAMES,
                                                    data_files, fit_files)):
    data_df = data_df_from_file(data_file, name_map=NAME_MAP)
    fit_df = fit_df_from_file(fit_file, name_map=NAME_MAP)
    if np.nanmax(np.abs(data_df)) > 150:
        normalize = True
    else:
        normalize = False
    ax = fig.add_subplot(4, 2, i + 1 + 4)
    plot_experiments(data_df, ax, fit_df=fit_df.loc['Best-fit values'],
                     colors_dict=colors.mol_colors,
                     invert=True, normalize=normalize, title=name,
                     ylabel="Specific Binding (%)")

sns.despine(fig=fig, offset=8)
add_panel_labels(fig, xoffset=.26, label_offset=1)
fig.tight_layout(rect=[0, 0, 1, .97])
fig.savefig("{}.png".format(FIG_PREFIX), dpi=300)
fig.savefig("{}.svg".format(FIG_PREFIX), dpi=300)
fig.savefig("{}.tif".format(FIG_PREFIX), dpi=300)
