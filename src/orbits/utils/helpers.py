"""Utility functions for orbital mechanics simulations."""

import itertools
import matplotlib.pyplot as plt

def cm_color():
    """Create a circular generator of colors from plt.cm.tab10"""
    return itertools.cycle(plt.cm.tab10.colors)

# Global color generator instance
_colors = cm_color()

def get_next_color():
    """Get the next color from the global color cycle"""
    return next(_colors)