"""Astrophysical objects for orbital mechanics simulations."""

from typing import Optional
import numpy as np
from ..utils.helpers import get_next_color


class AstroObject:
    """
    Class to simulate an astronomical object with a mass, radius, color, star system, position and velocity
    """
    def __init__(
        self,
        name: str,
        mass: float,
        radius: float = 0.1,
        color: Optional[str] = None,
        star_system: Optional["StarSystem"] = None,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
    ):
        """
        Create an AstroObject instance with a name, mass, radius, color, star_system, position and velocity

        Parameters
        ----------
        name : str
            The name of the object
        mass : float
            The mass of the object in solar masses
        radius : float, optional
            The radius of the object in AU, by default 0.1
        color : str, optional
            The color of the object, by default None
        star_system : StarSystem, optional
            The star system the object belongs to, by default None
        pos : np.ndarray, optional
            The position of the object in the star system, by default None
            The length of the array must be equal to the dimension of the star system
        vel : np.ndarray, optional
            The velocity of the object in the star system, by default None
            The length of the array must be equal to the dimension of the star system
        """
        self.mass = mass
        self.name = name
        self.radius = radius
        if color is None:
            self.color = get_next_color()
        else:
            self.color = color
        if star_system is not None:
            if pos is None:
                pos = np.zeros(star_system.n_dim)
            if vel is None:
                vel = np.zeros(star_system.n_dim)
            self.star_system = star_system
            self.star_system.add_astro_object(self, pos, vel)
    
    def pos(self) -> np.ndarray:
        """
        Return the position of the object as a numpy array of length solar_system.n_dim
        """
        if self.star_system is None:
            raise ValueError("The object is not in a star system")
        else:
            return self.star_system.get_position(self)

    def vel(self) -> np.ndarray:
        """
        Return the velocity of the object as a numpy array of length solar_system.n_dim
        """
        if self.star_system is None:
            raise ValueError("The object is not in a star system")
        else:
            return self.star_system.get_velocity(self)
        
    def distFromObj(self, otherObj: "AstroObject") -> Exception:
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
    
    def copy(self, configuration: Optional[tuple] = None, star_system: Optional["StarSystem"] = None) -> "AstroObject":
        """Copy the object to a new AstroObject and assign to it new position and velocity"""
        if configuration is None:
            configuration = self.pos(), self.vel()
        if star_system is None:
            star_system = self.star_system
        return AstroObject(
            name=self.name,
            mass=self.mass,
            radius=self.radius,
            color=self.color,
            star_system=star_system,
            pos=configuration[0],
            vel=configuration[1],
        )

    def update(self, configuration):
        """Update the position and velocity of the object"""
        self.star_system.set_position(self, configuration[0])
        self.star_system.set_velocity(self, configuration[1])