"""Factory methods for creating predefined star systems."""

from typing import Optional
import numpy as np
from ..core.constants import G, M_sun, M_jupiter
from ..core.objects import AstroObject
from ..utils.nasa_horizons import get_planet_vectors


class SystemFactory:
    """Factory class for creating predefined star systems."""
    
    @classmethod
    def star_and_planet(cls, StarSystemClass, star_mass, planet_mass, planet_period, step_size=1E-3):
        """
        Create a SolarSystem with a star and a planet
        """
        planet_x = star_mass * (G * planet_period**2 / (2*np.pi * (star_mass + planet_mass))**2)**(1/3)
        star_x = - planet_x * planet_mass / star_mass
        planet_vy = 2 * np.pi * planet_x / planet_period
        star_vy = - planet_vy * planet_mass / star_mass

        phase_space = np.array([star_x, 0, planet_x, 0, 0, star_vy, 0, planet_vy])

        solar_system = StarSystemClass(
            name="Star and planet",
            astro_objects=[
                AstroObject(mass=star_mass, name="Star"),
                AstroObject(mass=planet_mass, name="Planet"),
            ],
            phase_space=phase_space,
            step_size=step_size,
        )

        return solar_system

    @classmethod
    def our_solar_system(cls, StarSystemClass, t0: Optional[str] = None, step_size: float = 1E-3):
        """
        Create the solar system using real NASA Horizons data.
        
        Parameters
        ----------
        StarSystemClass : class
            The StarSystem class to instantiate
        t0 : str, optional
            Start date in 'YYYY-MM-DD' format, by default "1945-01-01"
        step_size : float, optional
            Integration step size in days, by default 1E-3
            
        Returns
        -------
        StarSystemClass
            Configured solar system with real planetary data
        """
        n_dim = 3
        names_p = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        colors_p = ["#808080", "#FFA500", "#0000FF", "#FF0000", "#D0B49F", "#FFA500", "#00FFFF", "#0000FF"]
        masses_p = np.array([3.285E23, 4.867E24, 5.972E24, 6.39E23, 1.898E27, 5.683E26, 8.681E25, 1.024E26])
        masses_p = masses_p / 1.989E30  # Convert to solar masses
        masses = np.concatenate(([1], masses_p))  # Add Sun mass
        n_objects = len(masses)
        radius_p = np.array([2439.7, 6051.8, 6371, 3389.5, 69911, 58232, 25362, 24622])
        R_sun = 695700
        radius_p = radius_p / R_sun  # Convert to solar radii

        # Use NASA Horizons data for 1945-01-01 by default
        if t0 is None:
            t0 = "1945-01-01"
        planet_vectors = get_planet_vectors(t0)

        # Create initial positions as array [x1, y1, z1, x2, y2, z2, ...]
        initial_positions = planet_vectors[:, :n_dim].flatten()
        # Create initial velocities as array [vx1, vy1, vz1, vx2, vy2, vz2, ...]
        initial_velocities = planet_vectors[:, n_dim:].flatten()
        phase_space = np.concatenate((initial_positions, initial_velocities))

        # Rescale positions and velocities to center of mass frame
        for i in range(n_dim):
            # Calculate center of mass for this dimension
            center_of_mass_i = np.sum(masses * phase_space[i:n_dim*n_objects:n_dim]) / np.sum(masses)
            # Calculate total momentum for this dimension
            total_momentum_i = np.sum(masses * phase_space[n_dim*n_objects+i::n_dim])
            # Subtract center of mass from positions
            phase_space[i:n_dim*n_objects:n_dim] -= center_of_mass_i
            # Subtract total momentum from velocities
            phase_space[n_dim*n_objects+i::n_dim] -= total_momentum_i / np.sum(masses)

        astro_objects = [
            AstroObject(
                mass=M_sun, 
                name="Sun", 
                radius=1, 
                color="#FFFF00"
            )
        ]
        for i in range(len(masses_p)):
            astro_objects.append(AstroObject(
                mass=masses_p[i], 
                name=names_p[i],
                radius=radius_p[i],
                color=colors_p[i],
            ))

        solar_system = StarSystemClass(
            name="Our solar system",
            astro_objects=astro_objects,
            n_dim=n_dim,
            phase_space=phase_space,
            step_size=step_size,
        )

        return solar_system

    @classmethod        
    def random_solar_system(cls, StarSystemClass, n_objects, step_size=1E-3):
        """
        Create a random solar system with n_objects
        """
        n_dim = 2
        astro_objects = []
        # generate a random phase space. The first n_objects are the positions, the second n_objects are the velocities
        # the positions are in the range [-1,1] AU and the velocities are in the range [-0.1,0.1] AU/day
        phase_space = np.random.rand(2*n_dim*n_objects) * 2 - 1
        phase_space[n_objects:] = phase_space[n_objects:] * 0.06 - 0.03

        # devide the objects in two groups, one with stars and one with planets
        n_stars = np.random.randint(1, n_objects)
        n_planets = n_objects - n_stars

        # generate the masses of the stars and planets
        star_masses = np.random.rand(n_stars) * 2 + 1
        planet_masses = np.random.rand(n_planets) * 0.1 + 0.1
        masses = np.concatenate((star_masses, planet_masses))

        # generate the names of the stars and planets
        star_names = [f"Star {i}" for i in range(n_stars)]
        planet_names = [f"Planet {i}" for i in range(n_planets)]

        # rescale the positions and velocities so that the center of mass is at the origin
        # and the total momentum is zero

        # calculate the center of mass
        center_of_mass_x = np.sum(masses * phase_space[:2*n_objects:2]) / np.sum(masses)
        center_of_mass_y = np.sum(masses * phase_space[1:2*n_objects:2]) / np.sum(masses)

        # calculate the total momentum
        total_momentum_x = np.sum(masses * phase_space[2*n_objects::2])
        total_momentum_y = np.sum(masses * phase_space[2*n_objects+1::2])

        # rescale the positions and velocities
        phase_space[:2*n_objects:2] -= center_of_mass_x
        phase_space[1:2*n_objects:2] -= center_of_mass_y
        phase_space[2*n_objects::2] -= total_momentum_x / np.sum(masses)
        phase_space[2*n_objects+1::2] -= total_momentum_y / np.sum(masses)

        # create the astro objects
        for i in range(n_stars):
            astro_objects.append(AstroObject(mass=star_masses[i], name=star_names[i]))
        for i in range(n_planets):
            astro_objects.append(AstroObject(mass=planet_masses[i], name=planet_names[i]))

        solar_system = StarSystemClass(
            name="Random solar system",
            astro_objects=astro_objects,
            phase_space=phase_space,
            step_size=step_size,
        )

        return solar_system