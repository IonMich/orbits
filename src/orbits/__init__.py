"""Orbits: A Python package for accurate gravitational orbit simulations."""

from .core.systems import StarSystem
from .core.objects import AstroObject
from .core.constants import G, M_sun, M_jupiter

# Import export functionality to add methods to StarSystem
from . import export

# Clean API exports
__all__ = [
    'StarSystem',
    'AstroObject', 
    'G',
    'M_sun',
    'M_jupiter',
]

# Version info
__version__ = "0.1.0"
