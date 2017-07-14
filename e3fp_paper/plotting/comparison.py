"""Methods for plotting fingerprint comparisons.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import seaborn as sns
from e3fp_paper.plotting.defaults import DefaultFonts


fonts = DefaultFonts()


def calculate_line(x, m=1., b=0.):
    return m * x + b


def get_weighted_leastsq(xdata, ydata, weights=None):
    if weights is None:
        sigma = None
        weights = np.ones_like(xdata)
    else:
        sigma = 1. / np.sqrt(weights)
    popt, pcov = curve_fit(calculate_line, xdata, ydata, [1., 0.], sigma=sigma)
    residuals = ydata - calculate_line(xdata, *popt)
    resid_sumsq = np.dot(weights, residuals**2)
    mean_y = np.average(ydata, weights=weights)
    total_sumsq = np.dot(weights, (ydata - mean_y)**2)
    r2 = 1. - (resid_sumsq / total_sumsq)
    return popt[0], popt[1], r2


def plot_tc_scatter(outlier_df, ax):
    ul = outlier_df.loc["Upper left"]
    ur = outlier_df.loc["Upper"]
    lr = outlier_df.loc["Lower Right"]
    fr = outlier_df.loc["Far Right"]
    for df, ms in zip([ul, ur, lr, fr], ['X', 'P', 's', 'D']):
        ax.scatter(df["ECFP4 TC"], df["E3FP TC"], s=20, linewidths=1,
                   marker=ms, facecolors='r', edgecolors='none', alpha=.8,
                   zorder=100)


def plot_tc_heatmap(counts_df, ax, outliers_df=None, cols=[], title="",
                    ref_line=True, fit_line=True, thresholds=[], cmap="bone_r",
                    limit=.5, set_auto_limits=False, logscale=True):
    x, y = counts_df[cols[0]], counts_df[cols[1]]
    non_nan_inds = ~(np.isnan(x) | np.isnan(y))
    x, y = x[non_nan_inds], y[non_nan_inds]
    try:
        count = counts_df["Count"]
        count = count[non_nan_inds]
    except:
        count = None

    if ref_line:
        ax.plot([0, 1], [0, 1], linewidth=1, color="gray", alpha=.5,
                linestyle="--", label="Equal", zorder=2)
        if len(thresholds) > 0:
            xthresh, ythresh = thresholds
            ax.axvline(xthresh, linewidth=1, color="royalblue", alpha=.75,
                       linestyle='--', zorder=2)
            ax.axhline(ythresh, linewidth=1, color="forestgreen",
                       alpha=.75, linestyle='--', zorder=2)

    xmax, ymax = max(max(x), limit), max(max(y), limit)
    if xmax <= 1 and ymax <= 1:
        xmax = ymax = 1
    xmin = ymin = 0
    if set_auto_limits:
        xmin, ymin = round(min(x)), round(min(y))
        xmax, ymax = round(xmax), round(ymax)
    extent = (xmin, xmax, ymin, ymax)

    norm = None
    if logscale:
        norm = matplotlib.colors.LogNorm()
    ax.hexbin(x, y, C=count, cmap=cmap, norm=norm, gridsize=50, extent=extent,
              edgecolors='none', zorder=1)

    if outliers_df is not None:
        x, y = outliers_df[cols[0]], outliers_df[cols[1]]
        if len(x) > 0:
            ax.scatter(x, y, s=15, marker="x", facecolors="r",
                       edgecolors='none', alpha=.3, zorder=3)

    if fit_line:
        slope, intercept, r2 = get_weighted_leastsq(x, y, weights=count)
        ax.plot([xmin, xmax],
                [slope * xmin + intercept, slope * xmax + intercept],
                linewidth=1, color="r", alpha=.75, linestyle="--",
                label="Trend", zorder=2)
        print("Fit with slope {:.4f}, intercept {:.4f}, and R^2 {:.4f}".format(
            slope, intercept, r2))

    ax.set_xlim(xmin, xmax + .02)
    ax.set_ylim(ymin, ymax + .02)
    if xmax <= 1 and ymax <= 1:
        ax.set_aspect('equal')
    xticks = np.linspace(xmin, xmax, 5)
    ax.set_xticks(xticks)
    yticks = np.linspace(ymin, ymax, 5)
    ax.set_yticks(yticks)
    ax.set_xlabel(cols[0], fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel(cols[1], fontsize=fonts.ax_label_fontsize)
    ax.set_title(title, fontsize=fonts.title_fontsize)


def plot_tc_hists(counts_df, ax, cols=[], title="", colors=[], thresholds=[],
                  logscale=True, show_legend=True,
                  legend_fontsize=fonts.legend_fontsize):
    if logscale:
        ax.set_yscale("log")
    count = counts_df["Count"]
    ref_line_colors = ["royalblue", "forestgreen"]
    for i, col in enumerate(cols):
        name = col.split(" ")[0]
        tcs = counts_df[col]
        alpha = 1.
        if i > 0:
            alpha = .9
        else:
            alpha = 1.
        try:
            color = colors[i]
        except IndexError:
            color = None
        sns.distplot(tcs, label=name, kde=False, ax=ax, norm_hist=True,
                     color=color, hist_kws={"linewidth": 0,
                                            "histtype": "stepfilled",
                                            "alpha": alpha,
                                            "zorder": 2 * i + 1,
                                            "weights": count})
        if len(thresholds) > 0:
            ax.axvline(thresholds[i], linewidth=1, color=ref_line_colors[i],
                       alpha=.75, linestyle='--', zorder=2 * i + 2)

    if logscale:
        ylabel = "Log Frequency"
    else:
        ylabel = "Frequency"
    ax.set_xlabel("TC", fontsize=fonts.ax_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=fonts.ax_label_fontsize)
    ax.set_xlim(0, counts_df[cols].values.max())
    ax.set_title(title, fontsize=fonts.title_fontsize)
    if show_legend:
        legend = ax.legend(loc=1, fontsize=legend_fontsize, frameon=True)
        legend.get_frame().set_linewidth(0)
