"""Plotting utilities

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
from .defaults import DefaultFonts

fonts = DefaultFonts()

def add_panel_label(label, ax):
    """Add a panel label to an axis."""
    ax.text(-.2, 1.1, label, transform=ax.transAxes,
            fontsize=fonts.panel_label_fontsize, fontweight='bold', va='top',
            ha='right')
