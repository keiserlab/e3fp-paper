"""Defaults used for plotting.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from matplotlib import rc, rcParams
try:
    import seaborn as sns
    sns.set_style("white")
except ImportError:
     pass

rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}',
                                   r'\sisetup{detect-all}',
                                   r'\usepackage{helvet}',
                                   r'\usepackage{sansmath}',
                                   r'\sansmath']
rcParams['path.simplify'] = True
rcParams['path.simplify_threshold'] = 1.
rc('text', usetex=False)
rc('font', family='sans-serif', size=8)
rc('mathtext', default='regular')
rc('xtick', labelsize=7)
rc('ytick', labelsize=7)
rc('legend', fontsize=7, labelspacing=.25)
rc('axes', labelsize=8, titlesize=8)


class DefaultColors(object):

    """Default colors for plotting."""

    def __init__(self):
        self.ecfp_color = (0, 0, 0)
        self.ecfp_chiral_color = (.39, .49, .39)
        self.e2fp_color = (.12, .47, .71)
        self.e2fp_stereo_color = (.2, .63, .17)
        self.e3fp_nostereo_color = (.94, .35, .35)
        self.e3fp_color = (1, .6, .18)
        self.rocs_shape_color = (.41, .34, .52)
        self.rocs_combo_color = (0, .18, .31)
        self.mol_colors = {'Alphaprodine': (0.216, 0.495, 0.720),
                           'Anpirtoline': (0.894, 0.102, 0.110),
                           'Cypenamine': (0.304, 0.683, 0.293)}


class DefaultFonts(object):

    """Default fonts/sizes for plotting."""

    def __init__(self):
        self.panel_label_fontsize = 12
        self.title_fontsize = 8
        self.ax_label_fontsize = 8
        self.ax_ticks_fontsize = 7
        self.legend_fontsize = 7
