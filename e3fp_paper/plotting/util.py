"""Plotting utilities

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from .defaults import DefaultFonts

fonts = DefaultFonts()


def add_panel_label(label, ax, xoffset=-.2, yoffset=1.12):
    """Add a panel label to an axis."""
    ax.text(xoffset, yoffset, label, transform=ax.transAxes,
            fontsize=fonts.panel_label_fontsize, fontweight='bold', va='top',
            ha='left')
