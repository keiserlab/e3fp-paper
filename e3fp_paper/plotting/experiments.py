"""Methods for plotting experimental results.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from collections import OrderedDict

import scipy as sc
import numpy as np
import pandas as pd


def results_df_from_files(files_dict, name_map):
    df_dict = {}
    for target, fs in files_dict.items():
        df_dict[target] = pd.concat(
            [pd.DataFrame.from_csv(x, sep="\t") for x in fs],
            keys=fs, axis=1)

    targets, dfs = zip(*sorted(df_dict.items()))
    df = pd.concat(dfs, axis=1, keys=targets,
                   names=['target', 'file', 'compound'])
    mol_names = df.columns.levels[-1]
    mol_name_reformat_map = {k: str(k).split(".")[0].split(" #")[0]
                             for k in mol_names}
    mol_name_reformat_map = {k: name_map.get(v, v)
                             for k, v in mol_name_reformat_map.items()}

    df.rename(columns=mol_name_reformat_map, inplace=True)
    df = df.convert_objects(convert_numeric=True)
    df.sort_index(axis=1, inplace=True)
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


def normalize(val, min_val, max_val, mult=1.):
    """Normalize a value to be between 0 and `mult`."""
    return mult * (val - min_val) / (max_val - min_val)


def plot_results_df(df, ax, colors_dict, fit_df, ref_col=None,
                    fit_type="logIC50", norm=False, as_perc=False, title="",
                    num_fit_points=1000):
    """Plot all results from a set of experiments with or without controls.

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe where row name corresponds to log concentration, and column
        index corresponds to compound name. Where multiple columns have the
        same name, column values are averaged and error bars are shown.
    ax : Axis
        Axis of matplotlib figure on which to plot curves
    colors_dict : dict
        dict matching compound name to color
    fit_df : Pandas DataFrame
        Dataframe containing parameters of curve fit to data.
    ref_col : str, optional
        Column name of reference compound/control.
    fit_type : str, optional
        Type of fit that was performed. Valid options are 'logIC50' and
        'logKi'.
    norm : bool, optional
        Normalize curves to between 0 and 100. Set to false if binding has
        already been normalized to this range.
    as_perc : bool, optional
        Plot values as percent binding of unlabeled compound. Vertically flips
        curves by subtracting them by 100.
    title : str, optional
        Description
    num_fit_points : int, optional
        Description
    """
    cols = list(OrderedDict.fromkeys(df.columns))

    for i, col in enumerate(cols):
        col_df = df.loc[:, col].dropna(axis=0, how='all')
        num_cols = df.columns.size
        ys = np.concatenate(np.atleast_2d(np.array(col_df).T))
        real_inds = np.where(~np.isnan(ys))
        ys = ys[real_inds]
        logds = np.concatenate([col_df.index for x in col_df])[real_inds]

        if num_cols > 1:
            means = col_df.mean(axis=1, skipna=True)
            stds = sc.stats.sem(col_df, axis=1, nan_policy='omit')
        else:
            means = ys
            stds = np.zeros_like(means)

        top, bottom = fit_df.loc[col]['Top'], fit_df.loc[col]['Bottom']
        if norm:
            means = normalize(means, bottom, top, mult=100.)
            stds = normalize(stds, bottom, top, mult=100.)
        if as_perc:
            means = 100 - means

        color = colors_dict.get(col, "k")
        logds = np.linspace(min(df.index), max(df.index), num_fit_points)
        label = col
        if fit_type == "logIC50":
            logIC50 = fit_df.loc[col]['logIC50']
            curve_fit = binding_curve_logIC50(logds, top, bottom, logIC50)
            label += " (logIC50: {:.2f})".format(logIC50)
        else:  # logKi
            logKi = fit_df.loc[col]['logKi']
            curve_fit = binding_curve_logKi(logds, top, bottom, logKi,
                                            fit_df.loc[col]['HotNM'],
                                            fit_df.loc[col]['HotKdNM'])
            label += " (logKi: {:.2f})".format(logKi)

        ax.errorbar(col_df.index, means, yerr=stds, color=color, capsize=2)
        ax.scatter(col_df.index, means, marker='o', s=25, color=color,
                   label=label)

        if norm:
            curve_fit = normalize(curve_fit, bottom, top, mult=100.)
        if as_perc:
            curve_fit = 100 - curve_fit
        ax.plot(logds, curve_fit, color=color, linewidth=1.5)

    ax.legend(loc=2, fontsize=10, borderaxespad=-1)
    ax.set_xlim(min(df.index) - .5, max(df.index) + .5)
    ax.set_xlabel(r"log[Compound] (M)")
    ax.set_ylabel(r"Competitive Binding")
    # ax.set_yticks([0., 0.5, 1.])
    # ax.set_yticklabels(['0', '50', '100'])
    # ax.set_ylim(-.35, 1.15)
    ax.set_title(title)
    return ax


