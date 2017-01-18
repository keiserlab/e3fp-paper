"""Defaults used for plotting.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import seaborn as sns
sns.set_style("white")


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
        return sns.color_palette()


class DefaultFonts(object):

    """Default fonts/sizes for plotting."""

    def __init__(self):
        self.panel_label_fontsize = 16
        self.title_fontsize = 14
        self.ax_label_fontsize = 12
        self.ax_ticks_fontsize = 10
        self.legend_fontsize = 10

