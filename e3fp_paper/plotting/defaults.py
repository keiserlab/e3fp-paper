"""Defaults used for plotting.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from matplotlib import rc, rcParams
import seaborn as sns

sns.set_style("white")
rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}',
                                   r'\sisetup{detect-all}',
                                   r'\usepackage{helvet}',
                                   r'\usepackage{sansmath}',
                                   r'\sansmath']
rc('text', usetex=False)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('mathtext', default='regular')


class DefaultColors(object):

    """Default colors for plotting."""

    def __init__(self):
        pass

    def get_e3fp_ecfp_colors(self, n=2, reverse=False):
        """Get colors for E3FP...N-2 2d-like variants...ECFP."""
        return sns.cubehelix_palette(n, start=1.0, rot=.1, dark=0, light=.75,
                                     hue=2.5, reverse=reverse)

    @property
    def e3fp_color(self):
        """Get color for E3FP."""
        return self.get_e3fp_ecfp_colors(n=1)

    @property
    def ecfp_color(self):
        """Get color for ECFP."""
        return self.get_e3fp_ecfp_colors(n=1, reverse=True)

    @property
    def mol_colors(self):
        """Get colors for molecules."""
        return {'Alphaprodine': (0.216, 0.495, 0.720),
                'Anpirtoline': (0.894, 0.102, 0.110),
                'Cypenamine': (0.304, 0.683, 0.293)}


class DefaultFonts(object):

    """Default fonts/sizes for plotting."""

    def __init__(self):
        self.panel_label_fontsize = 16
        self.title_fontsize = 14
        self.ax_label_fontsize = 12
        self.ax_ticks_fontsize = 10
        self.legend_fontsize = 10
