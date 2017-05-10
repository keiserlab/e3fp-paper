"""Methods for plotting experimental results.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import itertools
import csv
from collections import OrderedDict

import scipy as sc
import numpy as np
import pandas as pd
from .defaults import DefaultFonts

fonts = DefaultFonts()


def _replace_substring(string, name_map):
    """Replace first instance of substring that appears in map with value."""
    for x in name_map:
        if x == string or x in string:
            return string.replace(x, name_map[x])
    return string


def data_df_from_file(fn, name_map={}):
    """Build dataframe of data points from file exported from Prism.

    Parameters
    ----------
    fn : str
        Filename
    name_map : dict, optional
        Dict mapping a column name to a replacement

    Returns
    -------
    Pandas DataFrame
        Dataframe of data
    """
    df = pd.read_csv(fn, sep="\t", index_col=0)
    df.replace(to_replace=r"\d+\*", value=pd.np.nan, regex=True, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = [str(x).split(".")[0] for x in df.columns]
    df.columns = [_replace_substring(x, name_map=name_map) for x in df.columns]
    return df.apply(pd.to_numeric, errors='ignore')


def fit_df_from_file(fn, name_map={}):
    """Build dataframe of fit statistics from file exported from Prism.

    Parameters
    ----------
    fn : str
        Filename
    name_map : dict, optional
        Dict mapping a column name to a replacement

    Returns
    -------
    Pandas DataFrame
        Dataframe of fit statistics
    str
        Type of experiment
    """
    data = {}
    with open(fn, "rb") as f:
        section_name = None
        col_names = None
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            if i == 0:
                col_names = list(row)
                col_names[0] = "Name"
                continue
            elif all([len(x) == 0 for x in row[1:]]):
                section_name = row[0]
            else:
                for col, val in zip(col_names, row):
                    data.setdefault(col, []).append(
                        val.strip().split("= ")[-1])
                data.setdefault("Section", []).append(section_name)
    df = pd.DataFrame(data=data, columns=['Section'] + col_names)
    df.set_index(['Section', 'Name'], inplace=True)
    df.columns = [_replace_substring(x, name_map=name_map) for x in df.columns]
    df.loc["Best-fit values"] = [pd.to_numeric(x) for x
                                 in df.loc["Best-fit values"].values]
    return df


def binding_curve_logIC50(logD, top, bottom, logIC50):
    """Compute the binding curve when logIC50 is fit.

    Used to fit dissociation constant of unlabeled ligand by measuring its
    competition for binding against a radioligand.

    Parameters
    ----------
    logD : float or iterable of float
        Log concentration of unlabeled ligand
    top : float
        Upper plateau of curve corresponding to maximum binding
    bottom : float
        Lower plateau of curve corresponding to minimum binding
    logIC50 : float
        Log concentration of unlabeled ligand that results in binding
        half-way between `bottom` and `top`.

    Returns
    -------
    float or iterable of float
        Binding at log concentrations of `logD`

    References
    ----------
    https://www.graphpad.com/guides/prism/5/user-guide/prism5help.html?reg_one_site_competition_ic50.htm
    """
    logD = np.asarray(logD)
    return bottom + (top - bottom) / (1 + 10**(logD - logIC50))


def binding_curve_logKi(logD, top, bottom, logKi, radioligandNM, hotKdNM):
    """Compute the binding curve when logKi is fit.

    Used to fit dissociation constant of unlabeled ligand by measuring its
    competition for binding against a radioligand when concentration of
    radioligand and its Kd are known.

    Parameters
    ----------
    logD : float or iterable of float
        Log concentration of unlabeled ligand
    top : float
        Upper plateau of curve corresponding to maximum binding
    bottom : float
        Lower plateau of curve corresponding to minimum binding
    logKi : float
        Log of the equilibrium dissociation constant of an unlabeled ligand
    radioligandNM : float
        The concentration of labeled ligand in nM, an experimental constant
    hotKdNM : float
        The equilibrium dissociation constant of the labeled ligand in nM

    Returns
    -------
    float or iterable of float
        Binding at log concentrations of `logD`

    References
    ----------
    https://www.graphpad.com/guides/prism/5/user-guide/prism5help.html?reg_one_site_competition_ki.htm
    """
    logD = np.asarray(logD)
    logEC50 = np.log10(10**logKi * (1 + radioligandNM / hotKdNM))
    return bottom + (top - bottom) / (1 + 10**(logD - logEC50))


def agonist_curve(logD, top, bottom, logEC50, hill_slope=1.):
    """Compute the log[dose]-response curve when agonist logEC50 is fit.

    Parameters
    ----------
    logD : float or iterable of float
        Log concentration of agonist dose
    top : float
        Upper plateau of curve corresponding to maximum response
    bottom : float
        Lower plateau of curve corresponding to minimum response
    logEC50 : float
        Concentration of agonist corresponding to half-maximal response
    hill_slope : float
        Slope of dose-response curve, defaults to 1.0.

    Returns
    -------
    float or iterable of float
        Response at log agonist concentrations of `logD`

    References
    ----------
    https://www.graphpad.com/guides/prism/5/user-guide/prism5help.html?reg_one_site_competition_ic50.htm
    """
    logD = np.asarray(logD)
    return bottom + (top - bottom) / (1 + 10**((logEC50 - logD) * hill_slope))


def antagonist_shift(B, pA2, schild_slope=1.):
    """Compute the shift to agonist logEC50, logEC90, etc. with antagonist.

    Parameters
    ----------
    B : float
        Concentration of antagonist
    pA2 : float
        -log agonist concentration needed to shift curve by factor of 2
    schild_slope : float, optional
        Slope of Schild plot, defaults to 1.0

    Returns
    -------
    float or iterable of float
        Change in log equiactive agonist concentration at antagonist
        concentration

    References
    ----------
    http://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?reg_gaddumschild.htm
    """
    return np.log10(1 + (B / (10**(-pA2)))**schild_slope)


def get_normalized(val, min_val, max_val, mult=100.):
    """Normalize a value to be between 0 and `mult`."""
    return mult * (val - min_val) / (max_val - min_val)


def plot_schild(fit_df, ax, num_fit_points=1000, color="k"):
    Bs = fit_df.loc["B"]
    fit_logBs = np.linspace(np.log10(Bs[1]) - 1, np.log10(Bs.max()),
                            num_fit_points)
    logEC50 = fit_df.loc["LogEC50"][0]
    pA2 = fit_df.loc["pA2"][0]
    schild_slope = fit_df.loc["SchildSlope"][0]
    fitEC50s = logEC50 + antagonist_shift(10**fit_logBs, pA2,
                                          schild_slope=schild_slope)
    EC50_ratio = 10**(fitEC50s - logEC50)
    ax.plot(fit_logBs, np.log10(EC50_ratio - 1), color=color, linewidth=1)
    ax.set_ylabel(r"Agonist $\log{{\left[EC_{{50}}\ Ratio - 1\right]}}$ (M)",
                  fontsize=fonts.ax_label_fontsize)
    ax.set_xlabel(r"$\log{{\left[Antagonist\right]}}$ (M)",
                  fontsize=fonts.ax_label_fontsize)
    ax.set_xticks(np.arange(int(fit_logBs.min()), int(fit_logBs.max() + 1)))


def plot_experiments(data_df, ax, fit_df=None, colors_dict={},
                     num_fit_points=1000, invert=True, normalize=False,
                     ylabel="", title=""):
    """Plot all results from a set of experiments.

    Parameters
    ----------
    data_df : Pandas DataFrame
        Dataframe where row name corresponds to log concentration, and column
        index corresponds to compound name. Where multiple columns have the
        same name, column values are averaged and error bars are shown.
    ax : Axis
        Axis of matplotlib figure on which to plot curves
    fit_df : Pandas DataFrame
        Dataframe containing parameters of curve fit to data.
    colors_dict : dict
        dict matching compound name to color
    num_fit_points : int, optional
        Description
    invert : bool, optional
        Invert curve by subtracting from 100.
    normalize : bool, optional
        Normalize curves to between 0 and 100. Set to false if binding has
        already been normalized to this range.
    ylabel : str, optional
        Label of y-axis data.
    title : str, optional
        Title of subplot
    """
    cols = list(OrderedDict.fromkeys(data_df.columns))
    marker_iter = itertools.cycle(('s', '^', 'D', 'v'))
    logd_min = min(data_df.index)
    logd_max = max(data_df.index)
    fit_logds = np.linspace(logd_min, logd_max, num_fit_points)
    dots = []
    labels = []
    max_val = data_df.values.max()
    min_val = data_df.values.min()
    max_plot_vals = []
    min_plot_vals = []
    for i, col in enumerate(cols):
        label = col
        is_ref = False
        if isinstance(col, str):
            mol_name = str(col).split(' ')[0]
        else:
            is_ref = True
            mol_name = col
        if mol_name in colors_dict:
            color = colors_dict[mol_name]
        else:
            color = 'k'
            is_ref = True
        col_df = data_df.loc[:, col].dropna(axis=0, how='all')
        logds = np.asarray(col_df.index)

        if fit_df is not None and col in fit_df.columns:
            col_fit_df = fit_df[col]
            top = col_fit_df.loc['Top']
            bottom = col_fit_df.loc['Bottom']
            max_val = top
            min_val = bottom

            if 'LogIC50' in col_fit_df.index:
                logIC50 = col_fit_df.loc['LogIC50']
                fit = binding_curve_logIC50(fit_logds, top, bottom,
                                            col_fit_df.loc["LogIC50"])
                label = r"{} ($\log{{IC_{{50}}}}$: {:.2f})".format(label,
                                                                   logIC50)
            elif "logKi" in col_fit_df.index and "HotNM" in col_fit_df.index:
                logKi = col_fit_df.loc['logKi']
                fit = binding_curve_logKi(fit_logds, top, bottom, logKi,
                                          col_fit_df.loc["HotNM"],
                                          col_fit_df.loc["HotKdNM"])
                label = r"{} ($\log{{K_i}}$: {:.2f})".format(label, logKi)
            elif ("LogEC50" in col_fit_df.index and
                  "HillSlope" in col_fit_df.index):
                logEC50 = col_fit_df.loc['LogEC50']
                if "SchildSlope" in col_fit_df.index:
                    logEC50 = logEC50 + antagonist_shift(
                        col_fit_df.loc["B"], col_fit_df.loc["pA2"],
                        schild_slope=col_fit_df.loc["SchildSlope"])
                fit = agonist_curve(fit_logds, top, bottom, logEC50,
                                    hill_slope=col_fit_df.loc["HillSlope"])
                if label != 0:
                    try:
                        label = r"${:.0E}}}$".format(label).replace(
                            r"E-", r"x10^{-")
                    except ValueError:
                        label = str(label)
                else:
                    label = r"$0$"

            label = label.replace("#", "")

            if normalize:
                fit = get_normalized(fit, min_val, max_val)

            if invert:
                fit = 100 - fit

            max_plot_vals.append(fit.max())
            min_plot_vals.append(fit.min())

            ax.plot(fit_logds, fit, color=color, linewidth=1,
                    zorder=2 * i + 1)

        if col_df.ndim < 2:
            means = np.asarray(col_df)
            stds = np.zeros_like(means)
        else:
            means = np.asarray(col_df.mean(axis=1, skipna=True))
            stds = np.asarray(sc.stats.sem(col_df, axis=1, nan_policy='omit'))

        if normalize:
            means = get_normalized(means, min_val, max_val)
            stds = get_normalized(stds, 0, max_val - min_val)

        if invert:
            means = 100 - means

        max_plot_vals.append((means + stds).max())
        min_plot_vals.append((means - stds).min())

        error_inds = (stds > 0.)
        (_, caps, _) = ax.errorbar(logds[error_inds], means[error_inds],
                                   yerr=stds[error_inds], fmt='none',
                                   ecolor=color, capsize=2, linewidth=1,
                                   alpha=.9, zorder=2 * i)
        for cap in caps:
            cap.set_markeredgewidth(1)

        if is_ref:
            marker = 'o'
        else:
            marker = next(marker_iter)
        dot = ax.scatter(logds, means, s=20, lw=0, marker=marker, c=color,
                         label=col, zorder=2 * i + 1)
        dots.append(dot)

        labels.append(label)

    max_plot_val = max(max_plot_vals + [100])
    min_plot_val = min(min_plot_vals + [0])
    legend_loc = 'upper left'
    ax.set_ylim(min_plot_val - 3, max_plot_val + 23)

    ax.legend(dots, labels, loc=legend_loc,
              fontsize=fonts.legend_fontsize, borderaxespad=-1,
              handletextpad=-.5)
    ax.set_xlim(logd_min - .5, logd_max + .5)
    ax.set_xlabel(r"$\log{{\left[Drug\right]}}$ (M)",
                  fontsize=fonts.ax_label_fontsize)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel, fontsize=fonts.ax_label_fontsize)
    ax.set_yticks(np.linspace(0, 100, 6))
    ax.set_title(title, fontsize=fonts.title_fontsize, y=1.03)
