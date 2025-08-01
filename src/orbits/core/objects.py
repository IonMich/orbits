"""Astrophysical objects for orbital mechanics simulations."""

import numpy as np
from ..utils.helpers import get_next_color


class AstroObject:    
    """
    Create an astrophysical object with defined mass 2-D position and 2-D velocity
    """
    def __init__(self, name, mass, radius=1, color=None, star_system=None, pos=None, vel=None):
        self.mass = mass
        self.name = name
        self.radius = radius
        if color is None:
            self.color = get_next_color()
        else:
            self.color = color
        if star_system != None:
            if pos == None:
                pos = np.zeros(star_system.n_dim)
            if vel == None:
                vel = np.zeros(star_system.n_dim)
            self.star_system = star_system
            self.star_system.add_astro_object(self, pos, vel)
    
    def pos(self):
        """
        Return the position of the object as a numpy array of length solar_system.n_dim
        """
        if self.star_system == None:
            raise ValueError("The object is not in a star system")
        else:
            return self.star_system.get_position(self)

    def vel(self):
        """
        Return the velocity of the object as a numpy array of length solar_system.n_dim
        """
        if self.star_system == None:
            raise ValueError("The object is not in a star system")
        else:
            return self.star_system.get_velocity(self)
        
    def distFromObj(self,otherObj):
        """
        Calculate the distance of this object  from another object
        
        Return the distance as a float 
        """
        if self.star_system != otherObj.star_system:
            raise ValueError("The two objects are not in the same star system")
        else:
            # return np.linalg.norm(self.pos() - otherObj.pos())
            raise NotImplementedError("This method is not implemented, because it is better to calculate "
                                        "the distances of all objects from each other at once")
    
    def copy(self, configuration=None, star_system=None):
        """Copy the object to a new AstroObject and assign to it new position and velocity"""
        if configuration == None:
            configuration = self.pos(), self.vel()
        if star_system == None:
            star_system = self.star_system
        return AstroObject(self.M, star_system, configuration[0], configuration[1])

    def update(self, configuration):
        """Update the position and velocity of the object"""
        self.star_system.set_position(self, configuration[0])
        self.star_system.set_velocity(self, configuration[1])