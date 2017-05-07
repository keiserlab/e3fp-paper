"""Plotting utilities

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import string
from .defaults import DefaultFonts

fonts = DefaultFonts()


def add_panel_label(label, ax, xoffset=-.2, yoffset=1.12):
    """Add a panel label to an axis."""
    ax.text(xoffset, yoffset, label, transform=ax.transAxes,
            fontsize=fonts.panel_label_fontsize, fontweight='bold', va='top',
            ha='left')


def add_panel_labels(fig=None, axes=None, xoffset=.17, yoffset=.03,
                     label_offset=0):
    """Add labels to all axes."""
    panel_labels = string.ascii_uppercase[label_offset:]

    if axes is None or len(axes) == 0:
        axes = fig.get_axes()

    try:
        x = [-v for v in xoffset]
    except TypeError:
        x = [-xoffset for v in axes]

    try:
        y = [1 + v for v in yoffset]
    except TypeError:
        y = [1 + yoffset for v in axes]

    for i, ax in enumerate(axes):
        label = panel_labels[i]
        ax.text(x[i], y[i], s=label, fontsize=fonts.panel_label_fontsize,
                fontweight='bold', va='bottom', ha='right',
                transform=ax.transAxes)
