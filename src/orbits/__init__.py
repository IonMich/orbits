"""Orbits: A Python package for accurate gravitational orbit simulations."""

from .core.systems import StarSystem
from .core.objects import AstroObject
from .core.constants import G, M_sun, M_jupiter
from .utils.nasa_horizons import get_planet_vectors

# Import export functionality to add methods to StarSystem
from . import export

# Clean API exports
__all__ = [
    'StarSystem',
    'AstroObject', 
    'G',
    'M_sun',
    'M_jupiter',
    'get_planet_vectors',
]

# Version info
__version__ = "0.1.0"
