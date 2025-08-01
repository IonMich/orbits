#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:24:33 2018

@author: yannis
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# all masses are in solar masses
# all distances are in AU
# all times are in days
M_sun = 1.0
G = 0.000295912208
M_jupiter = 1 / 1047.93 * M_sun

def cm_color():
    import itertools
    # create a cyrcular generator of colors from plt.cm.tab10
    return itertools.cycle(plt.cm.tab10.colors)
    
colors = cm_color()

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
            self.color = next(colors)
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


class StarSystem:
    """
    Class to simulate a solar system with stars and planets
    """
    def __init__(
        self,
        name: str,
        astro_objects: list[AstroObject] | int = [],
        phase_space: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None,
        n_dim: int = 2,
        evolve_method: str = "S4",
        step_size: float = 1E-3,
    ):
        self.name = str(name)
        self.n_dim = n_dim
        self.astro_objects: list[AstroObject] = []
        self.step_size = step_size
        
        try:
            self.evolve = {
                "RK4": self.rk4,
                "ME": self.modified_Euler,
                "S2": self.symplectic2,
                "S4": self.symplectic4,
            }[evolve_method]
        except KeyError:
            raise ValueError("The evolve_method must be one of 'RK4', 'ME', 'S2' or 'S4'")
        # check that astro_objects is a list
        if not isinstance(astro_objects, list):
            # check that astro_objects is a positive integer
            if not isinstance(astro_objects, int) or astro_objects <= 0:
                raise ValueError(
                    "The astro_objects must be a list or a positive integer. "
                    "Leave it an empty list if you want it to be inferred "
                    "from the phase_space and masses")
            _astro_objects = []
            create = int(astro_objects)
        else:
            # check that astro_objects is a list of AstroObject
            _masses = []
            for astro_object in astro_objects:
                if not isinstance(astro_object, AstroObject):
                    raise TypeError("The astro_objects must be a list of AstroObject")
                # check that the astro_objects are not already in a star system
                try:
                    if astro_object.star_system is not None:
                        raise ValueError("The astro_objects must not be in a star system")
                except AttributeError:
                    pass
                _masses.append(astro_object.mass)
            _astro_objects = astro_objects.copy()
            create = 0
        
        num_astro_objects = len(_astro_objects) + create

        if phase_space is not None and not isinstance(phase_space, np.ndarray):
            raise TypeError("The phase_space must be a numpy array")
        if masses is not None and not isinstance(masses, (np.ndarray, list)):
            raise TypeError("The masses must be a numpy array or a list")
        if (phase_space is not None) and (masses is not None) and phase_space.size != 2 * self.n_dim * len(masses):
            raise ValueError("The phase_space divided by 2*n_dim must have the same length as the list of masses")
        if (phase_space is not None) and num_astro_objects > 0 and len(phase_space) != 2 * self.n_dim * num_astro_objects:
            raise ValueError("The phase_space divided by 2*n_dim must have the same length as the (list of) astro_objects")
        if (masses is not None) and create > 0 and len(masses) != create:
            raise ValueError("The list of masses must have the same length as the number of astro_objects to create")
        if (masses is not None) and _astro_objects:
            raise ValueError("The masses must be None if the list of astro_objects is not empty")

        # Generate default phase_space and masses if not given
        if phase_space is None:
            # create default positions and velocities if possible
            if (masses is not None):
                phase_space = np.zeros(2 * self.n_dim * len(masses))
                create = len(masses)
            elif num_astro_objects > 0:
                phase_space = np.zeros(2 * self.n_dim * num_astro_objects)
            else:
                raise ValueError("If the phase_space is not defined, the (list of) astro_objects or the list of masses must be defined")
        if (masses is not None) is None and num_astro_objects == 0:
            if (phase_space is not None):
                masses = np.ones(phase_space.size // (2 * self.n_dim))
                create = len(masses)
            else:
                raise ValueError("If the masses are not defined, the (list of) astro_objects or the phase_space must be defined")

        # Create the astro_objects if needed
        for i in range(create):
            astro_object = AstroObject(
                name="AstroObject {}".format(i),
                mass=masses[i],
            )
            _astro_objects.append(astro_object)

        # Add the _astro_objects to the star system
        for i, astro_object in enumerate(_astro_objects):
            self.add_astro_object(astro_object)
        
        self.phase_space = phase_space
        try:
            self.masses = _masses
        except NameError:
            self.masses = masses

        self.masses = np.array(self.masses)

    def add_astro_object(
        self,
        astro_object: AstroObject,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
    ):
        """
        Add an AstroObject instance to the star system
        """
        self.astro_objects.append(astro_object)
        astro_object.star_system = self
        try:
            self.masses = np.append(self.masses, astro_object.mass)
        except AttributeError:
            self.masses = np.array([astro_object.mass])
        if pos is not None and vel is not None:
            try:
                self.phase_space = np.append(self.phase_space, np.zeros(2*self.n_dim))
            except AttributeError:
                self.phase_space = np.zeros(2*self.n_dim)
            self.set_position(astro_object, pos)
            self.set_velocity(astro_object, vel)
    
    def get_position(self, astro_object: AstroObject) -> np.ndarray:
        """
        Return the position of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        return self.phase_space[index*self.n_dim:(index+1)*self.n_dim]

    def set_position(self, astro_object: AstroObject, pos: np.ndarray) -> None:
        """
        Set the position of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        self.phase_space[index*self.n_dim:(index+1)*self.n_dim] = pos

    def set_positions(self, pos: np.ndarray) -> None:
        """
        Set the position of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        self.phase_space[:self.phase_space.size//2] = pos

    def get_velocity(self, astro_object: AstroObject) -> np.ndarray:
        """
        Return the velocity of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        return self.phase_space[(index+1)*self.n_dim:(index+2)*self.n_dim]

    def set_velocity(self, astro_object: AstroObject, vel: np.ndarray) -> None:
        """
        Set the velocity of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        self.phase_space[(index+1)*self.n_dim:(index+2)*self.n_dim] = vel

    def get_pairwise_separations(self) -> np.ndarray:
        """
        Return the array of the pairwise separations of the list of AstroObject instances
        in the same star system. Each separation is an array of length equal to self.n_dim
        """
        num_objs = len(self.masses)
        positions = self.phase_space[:self.phase_space.size//2].reshape(num_objs, self.n_dim)
        # separations[i,j] is the separation vector pointing from the jth object to the ith object,
        # i.e. separations[i,j] = positions[i] - positions[j]
        separations = positions[:,np.newaxis,:] - positions[np.newaxis,:,:] # shape (num_objs, num_objs, n_dim)
        distances = np.linalg.norm(separations,axis=2) # shape (num_objs, num_objs)
        return separations, distances

    def get_accelerations(self) -> np.ndarray:
        """
        Return the array of the accelerations of the list of AstroObject instances
        in the same star system
        """
        separations, distances = self.get_pairwise_separations()
        
        G_over_r3 = G* np.divide(1, distances**3, out=np.zeros_like(distances), where=distances!=0)
        acceleration_components = - G_over_r3[:,:,np.newaxis] * separations * self.masses[np.newaxis,:,np.newaxis]
        
        accelerations = np.nansum(acceleration_components, axis=1) 
        
        # flatten the array of accelerations
        accelerations = accelerations.flatten()
        # now the shape of accelerations is (num_objs * self.n_dim,)
        return accelerations

    def get_phase_space_derivatives(self) -> np.ndarray:
        """
        Return the array of the derivatives of the phase space of the list of AstroObject instances
        in the same star system
        """
        velocities = self.phase_space[self.phase_space.size//2:]
        accelerations = self.get_accelerations()

        return np.concatenate((velocities, accelerations))

    def set_phase_space(self, phase_space: np.ndarray) -> None:
        """
        Set the phase space of the star system
        """
        self.phase_space = phase_space

    def get_kinetic_energy(self) -> float:
        """
        Return the kinetic energy of the star system
        """
        velocities = self.phase_space[self.phase_space.size//2:].reshape(len(self.masses), self.n_dim)
        kinetic_energy = 0.5 * np.sum(self.masses * np.sum(velocities**2, axis=1))
        return kinetic_energy

    def get_potential_energy(self) -> float:
        """
        Return the potential energy of the star system
        """
        _, distances = self.get_pairwise_separations()
        # note that distances includes the distance from each object to itself, which is zero.
        # so we should sum over only the upper triangular part of the matrix, excluding the diagonal
        energy = 0
        for i in range(len(self.masses)):
            for j in range(i+1,len(self.masses)):
                energy += - G * self.masses[i] * self.masses[j] / distances[i,j]
        return energy

    def get_total_energy(self) -> float:
        """
        Return the total energy of the star system
        """
        return self.get_kinetic_energy() + self.get_potential_energy()

    def rk4(self, inplace=True) -> np.ndarray:
        """
        Implement RK4 to evolve the phase space by dt=self.step_size
        """
        h = self.step_size
        original_configurations = self.phase_space.copy()
        
        k1 = h * self.get_phase_space_derivatives()
        self.set_phase_space(original_configurations+0.5*k1)
        k2 = h * self.get_phase_space_derivatives()
        self.set_phase_space(original_configurations+0.5*k2)
        k3 = h * self.get_phase_space_derivatives()
        self.set_phase_space(original_configurations+k3)
        k4 = h * self.get_phase_space_derivatives()

        new_phase_space = original_configurations + (k1 + 2*k2 + 2*k3 + k4)/6

        if inplace:
            self.set_phase_space(new_phase_space)
        else:
            self.set_phase_space(original_configurations)

        return new_phase_space

    def modified_Euler(self, inplace=True) -> np.ndarray:
        """
        Implement the modified Euler method, which is a symplectic integrator

        Also known as the leapfrog integrator, which is a first-order symplectic integrator

        We apply it in its kick-drift form:
        p' = p + h * a(q)
        q' = q + h * p'

        The terms drift and kick are associated with approximating the potential as a h-delta comb
        i.e. H(q,p) = 1/2 p^2 + Φ(q)* h * Σ_{j=-inf}^{inf} δ(t - jh)
        and evolving at each step from t-ε to t + h - ε, where 0 < ε << h
        Then there is a "kick" from t-ε to t + ε and a "drift" from t + ε to t + h - ε.

        CAUTION: Symplectic integrators under fixed time step are no longer symplectic
        if the time step is varied depending on the potitional and velocity of the system.
        """
        h = self.step_size
        original_configurations = self.phase_space.copy()

        positions, velocities = self.phase_space.copy()[:self.phase_space.size//2], self.phase_space.copy()[self.phase_space.size//2:]
        # kick step
        accelerations = self.get_accelerations()
        new_velocities = velocities + h * accelerations
        # drift step
        new_positions = positions + h * new_velocities
        new_phase_space = np.concatenate((new_positions, new_velocities))
        if inplace:
            self.set_phase_space(new_phase_space)
        else:
            self.set_phase_space(original_configurations)

        return new_phase_space

    def symplectic2(self, inplace=True) -> np.ndarray:
        """
        Implement the Verlet method, which is a symplectic integrator

        Also known as the leapfrog integrator, which is a second-order symplectic integrator

        We apply it in its drift-kick-drift form:
        q_1/2 = q + h/2 * p
        p' = p + h * a(q_1/2)
        q' = q_1/2 + h/2 * p'

        We see here that essentially we break the drift step into two parts, and apply the kick in between

        This integrator is actually time-reversible, and it does not require additional memory 
        for the intermediate steps (so good for N-body simulations)
        """
        h = self.step_size
        original_configurations = self.phase_space.copy()

        positions, velocities = self.phase_space.copy()[:self.phase_space.size//2], self.phase_space.copy()[self.phase_space.size//2:]
        
        # first drift step
        new_positions = positions + h/2 * velocities
        # kick step
        self.set_phase_space(np.concatenate((new_positions, velocities)))
        accelerations = self.get_accelerations()
        new_velocities = velocities + h * accelerations
        # second drift step
        new_positions = new_positions + h/2 * new_velocities
        new_phase_space = np.concatenate((new_positions, new_velocities))
        if inplace:
            self.set_phase_space(new_phase_space)
        else:
            self.set_phase_space(original_configurations)

        return new_phase_space

    def symplectic4(self, inplace=True) -> np.ndarray:
        """
        Implement the symplectic integrator of 4th order to evolve myRi by h
        Coefficients found in: 
        http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-5071.pdf
        """
        h = self.step_size
        x = (2**(1/3) + 2**(-1/3) - 1)/6
        
        ## TODO: Find what is wrong with this choice
    #    c1 = c4 = x + 1/2
    #    c2 = c3 = -x
    #    d1 = d3 = 2*x + 1
    #    d2 = - 4*x - 1
        
        c1 = 0
        c3 = - 4*x - 1
        c2 = c4 = 2*x + 1
        d2 = d3 = -x
        d1 = d4 = x + 1/2

        c = np.array([c1,c2,c3,c4])
        d = np.array([d1,d2,d3,d4])

        original_configurations = self.phase_space.copy()

        positions, velocities = (
            self.phase_space.copy()[:self.phase_space.size//2], 
            self.phase_space.copy()[self.phase_space.size//2:]
            )

        for i in range(4):
            accelerations = self.get_accelerations()
            velocities += c[i] * accelerations * h
            positions += d[i] * velocities * h
            new_phase_space = np.concatenate((positions, velocities))
            self.set_phase_space(new_phase_space)

        if not inplace:
            # Restore the original configuration
            self.set_phase_space(original_configurations)

        return new_phase_space
    
    def adapt_step(self, relative_error=1E-5, inplace=True):
        """
        Adapt the step size to reduce local error in positions to some specified relative error
        """
        h = self.step_size
        ## Evolve with two steps of h
        original_configurations = self.phase_space.copy()
        self.evolve(inplace=True)
        self.evolve(inplace=True)
        positions = (
            self.phase_space.copy()[:self.phase_space.size//2]
            .reshape(len(self.astro_objects), self.n_dim)
            )

        self.set_phase_space(original_configurations)
            
        ## Evolve with one step of 2*h
        self.step_size = 2*h
        self.evolve(inplace=True)
        positions2 = self.phase_space.copy()[:self.phase_space.size//2].reshape(len(self.astro_objects), self.n_dim)
        self.set_phase_space(original_configurations)
        self.step_size = h

        
        
        ## Calculate the maximum relative distance, defined (in 2D) as the maximum of sqrt((x1-x2)^2 + (y1-y2)^2) / sqrt(x1^2 + y1^2)
        # where (x1,y1) and (x2,y2) are the positions achieved with two steps of h and one step of 2*h
        position_differences = positions - positions2
        norm_positions = np.linalg.norm(positions, axis=1)
        norm_position_differences = np.linalg.norm(position_differences, axis=1)

        maxRelDist = np.max(norm_position_differences / norm_positions)

        # Using error estimates. Multiply by 0.85 for extra safety.   
        h = 0.85 * h * (relative_error / maxRelDist )**(1/5)

        if inplace:
            self.step_size = h
        
        return h

    @classmethod
    def star_and_planet(cls, star_mass, planet_mass, planet_period, step_size=1E-3):
        """
        Create a SolarSystem with a star and a planet
        """
        planet_x = star_mass * (G * planet_period**2 / (2*np.pi * (star_mass + planet_mass))**2)**(1/3)
        star_x = - planet_x * planet_mass / star_mass
        planet_vy = 2 * np.pi * planet_x / planet_period
        star_vy = - planet_vy * planet_mass / star_mass

        phase_space = np.array([star_x, 0, planet_x, 0, 0, star_vy, 0, planet_vy])
        star = AstroObject(
            mass=star_mass,
            name="Star",
        )
        planet = AstroObject(
            mass=planet_mass,
            name="Planet",
        )
        astro_objects = [star, planet]
        solar_system = cls(
            name="Star and planet",
            astro_objects=astro_objects,
            phase_space=phase_space,
            step_size=step_size,
        )

        return solar_system

    def get_planet_vectors(start_time_str):
        """Example usage: get_planet_vectors('2022-12-20')"""
        import requests
        import datetime

        API_URL = 'https://ssd.jpl.nasa.gov/api/horizons.api'

        command_codes = ['10', '199', '299', '399', '499', '599', '699', '799', '899']
        options = {
            "format": 'json',
            "MAKE_EPHEM": 'YES',
            "COMMAND": None,
            "EPHEM_TYPE": 'VECTORS',
            "CENTER": '500@0',
            "START_TIME": None,
            "STOP_TIME": None,
            "STEP_SIZE": '2d',
            "VEC_TABLE": '2',
            "REF_SYSTEM": "ICRF",
            "REF_PLANE": "ECLIPTIC",
            "VEC_CORR": "NONE",
            "OUT_UNITS": 'au-d',
            "VEC_LABELS": "YES",
            "VEC_DELTA_T": "NO",
            "CSV_FORMAT": "YES",
            "OBJ_DATA": "YES",
        }

        start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d')
        stop_time = start_time + datetime.timedelta(days=1)
        
        options['START_TIME'] = start_time.strftime('%Y-%m-%d')
        options['STOP_TIME'] = stop_time.strftime('%Y-%m-%d')
        planet_vectors = []
        for code in command_codes:
            options['COMMAND'] = code
            response = requests.get(API_URL, params=options)
            data = response.json()['result']
            # get the output csv data. It starts with $$SOE and ends with $$EOE
            csv_data = data[data.find('$$SOE')+5:data.find('$$EOE')-1]
            # strip any final commas and split the data into a list
            csv_data = csv_data.strip(',').split(',')
            # remove the first 2 elements, which are the time in two different formats
            csv_data = csv_data[2:]
            # convert the strings to floats
            csv_data = [float(x) for x in csv_data]
            planet_vectors.append(csv_data)
        return np.array(planet_vectors)

    @classmethod
    def our_solar_system(cls, t0=None, step_size=1E-3):
        """
        Create the inner solar system
        """
        n_dim = 3
        names_p = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        colors_p = ["#808080", "#FFA500", "#0000FF", "#FF0000", "#D0B49F", "#FFA500", "#00FFFF", "#0000FF"]
        masses_p = np.array([3.285E23, 4.867E24, 5.972E24, 6.39E23, 1.898E27, 5.683E26, 8.681E25, 1.024E26])
        masses_p = masses_p / 1.989E30
        masses = np.concatenate(([1], masses_p))
        n_objects = len(masses)
        radius_p = np.array([2439.7, 6051.8, 6371, 3389.5, 69911, 58232, 25362, 24622])
        R_sun = 695700
        radius_p = radius_p / R_sun        

        # use the data from NASA for 1945-01-01
        # https://ssd.jpl.nasa.gov/horizons.api
        if t0 is None:
            t0 = "1945-01-01"
        planet_vectors = cls.get_planet_vectors(t0)

        # create the initial positions as an array [x1, y1, z1, x2, y2, z2, ...]
        initial_positions = planet_vectors[:, :n_dim].flatten()
        # create the initial velocities as an array [vx1, vy1, vz1, vx2, vy2, vz2, ...]
        initial_velocities = planet_vectors[:, n_dim:].flatten()
        phase_space = np.concatenate(
            (initial_positions,
            initial_velocities)
        )

        # rescale the positions and velocities of x, y and possibly z
        for i in range(n_dim):
            # calculate the center of mass
            center_of_mass_i = np.sum(masses * phase_space[:n_dim*n_objects:n_dim]) / np.sum(masses)
            # calculate the total momentum
            total_momentum_i = np.sum(masses * phase_space[n_dim*n_objects::n_dim])
            # subtract the center of mass from the positions
            phase_space[i:n_dim*n_objects:n_dim] -= center_of_mass_i
            # subtract the total momentum from the velocities
            phase_space[n_dim*n_objects+i::n_dim] -= total_momentum_i / np.sum(masses)

        astro_objects = [
            AstroObject(
                mass=M_sun, 
                name="Sun", 
                radius=1, 
                color="#FFFF00")
                ]
        for i in range(len(masses_p)):
            astro_objects.append(AstroObject(
                mass=masses_p[i], 
                name=names_p[i],
                radius=radius_p[i],
                color=colors_p[i],
            ))


        solar_system = cls(
            name="Our solar system",
            astro_objects=astro_objects,
            n_dim=n_dim,
            phase_space=phase_space,
            step_size=step_size,
        )

        return solar_system


    @classmethod
    def random_solar_system(cls, n_objects, step_size=1E-3):
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

        solar_system = cls(
            name="Random solar system",
            astro_objects=astro_objects,
            phase_space=phase_space,
            step_size=step_size,
        )

        return solar_system


    def plot_orbits(self, t_0, t_end, indices=None, keep_points=None, real_time=False, resample_every_dt=1, mark_every_dt=None, animated=False, save=False, adaptive=True, relative_error=1E-5):
        """
        Plot the animation of the solar system
        Add a pause/continue button to the animation
        # add stick immediately below bottom of the x-axis another plot with the evolution of the difference between the total energy and the initial total energy
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # calculate the sizes of the axes
        ax_dE_height_proportion = 0.2
        x_margin = 0.15
        y_margin_bottom = 0.1
        ax_height_proportion = 1 - (ax_dE_height_proportion + y_margin_bottom)
        figure_x_size = 4
        figure_y_size = figure_x_size * (1 - 2*x_margin) / (1 - y_margin_bottom - ax_dE_height_proportion)

        fig = plt.figure(figsize=(figure_x_size, figure_y_size))
        ax_dE = fig.add_axes([x_margin, y_margin_bottom, 1-2*x_margin, ax_dE_height_proportion])

        projection = None
        aspect = 'equal'

        ax = fig.add_axes(
            [x_margin, 
            1-ax_height_proportion, 
            1-2*x_margin, 
            ax_height_proportion],
            projection=projection
        )

        # set the limits of the animation axis as the max of the distance of the objects from the origin
        # restrict the indices of the objects to plot to `indices`
        print(self.phase_space)
        if (not real_time) and (indices is None):
            max_dist = np.max(np.linalg.norm(self.phase_space[:self.phase_space.size//2].reshape(-1,self.n_dim), axis=1))
        else:
            max_dist = np.max(
                np.linalg.norm(
                    self.phase_space[:self.phase_space.size//2]
                    .reshape(-1,self.n_dim), axis=1),
                    where=np.isin(np.arange(self.masses.size), indices),
                    initial=0
                )
        max_coord = 2 * max_dist
        ax.set_xlim(-max_coord, max_coord)
        ax.set_ylim(-max_coord, max_coord)
        ax.set_aspect(aspect, adjustable='box', anchor='C')
        # remove the ticks and labels of the animation axis
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.tick_params(
            axis='x', 
            which='both', 
            bottom=True, 
            top=False,
            labelbottom=True,
            labeltop=False,
            direction='in',
            pad=-20,
        )

        # the default plt marker size is rcParams['lines.markersize'] ** 2. 
        # We want to scale the marker size with the radius of the object
        marker_size = np.array([max(obj.radius,0.3) for obj in self.astro_objects]) ** 2 * 5

        ## Create the scatter plot
        positions = self.phase_space[:self.phase_space.size//2].reshape((len(self.masses), self.n_dim))
        
        lines = []
        for i in range(len(self.astro_objects)):
            data = [positions[i, 0], positions[i, 1]]
            line, = ax.plot(
                *data,
                "o",
                alpha=0.5,
                markersize=marker_size[i],
                label=self.astro_objects[i].name,
                color=self.astro_objects[i].color,
            )
            lines.append(line)

        special_points = plt.scatter(
            [],
            [],
            s=10,
            color="black",
            marker="o",
            zorder=50000,
        )


        #  since G is in units of AU^3 / (Msun * day^2), the energy ([Msun * AU^2 / day^2]) is in units of Msun * AU^2 / day^2
        print(f"Total energy: {self.get_total_energy()} [Msun * AU^2 / day^2]")

        # the initial point is at x=t_0 and y=0
        initial_energy = self.get_total_energy()
        x = [t_0]
        y = [0]

        # create an artist for the energy evolution plot
        line, = ax_dE.plot(
            x, 
            y, 
            color="black", 
            label=r"$\frac{\Delta E}{E_0}$",
        )

        # set the scale of the y-axis to be logarithmic
        ax_dE.set_yscale("log")
        ax_dE.set_xscale("log")
        ax_dE.set_ylim(1E-20, 1E3)
        ax_dE.set_xlim(t_0, t_end)
        ax_dE.legend()

        ## Create the pause/continue button
        pause = False
        def onClick(event):
            nonlocal pause
            pause ^= True
        fig.canvas.mpl_connect("button_press_event", onClick)
        

        ## Create the progress bar
        from tqdm import tqdm
        pbar = tqdm(total=t_end-t_0, unit="day", desc="Time", position=0, leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        if not real_time:
            if indices is None:
                indices = np.arange(len(self.astro_objects))
            time = t_0
            positions = []
            times = []
            energies = []
            while time < t_end:
                self.evolve(inplace=True)
                time += self.step_size
                positions.append(self.phase_space[:self.phase_space.size//2].reshape((len(self.masses), self.n_dim)))
                times.append(time)
                energies.append(self.get_total_energy())
                pbar.update(self.step_size)
                pbar.set_postfix({"Energy": self.get_total_energy(), "Step size": float(self.step_size)})
                self.adapt_step(relative_error=relative_error,inplace=True)
            pbar.close()

            positions = np.array(positions)
            times = np.array(times)
            energies = np.array(energies)

            # keep only the positions of the objects in `indices`
            positions = positions[:, indices, :]
            x_data = positions[:, :, 0]
            y_data = positions[:, :, 1]

            
            if resample_every_dt is not None:
                # use np.interp to resample the data at regular intervals of resample_every_dt
                # the new times are the multiples of resample_every_dt that are smaller than the last time
                new_times = np.arange(times[0], times[-1], resample_every_dt)
                new_x_data = np.zeros((len(new_times), len(indices)))
                new_y_data = np.zeros((len(new_times), len(indices)))
                for i, line_idx in enumerate(indices):
                    new_x_data[:, i] = np.interp(new_times, times, x_data[:, i])
                    new_y_data[:, i] = np.interp(new_times, times, y_data[:, i])
                energies = np.interp(new_times, times, energies)
                times = new_times
                x_data = new_x_data
                y_data = new_y_data
                print(times.shape)
                print(x_data.shape)
                print(y_data.shape)
                # store x_data, y_data, times and energies in a csv file
                np.savetxt(f"{self.name}_positions.csv", np.hstack((times[:, np.newaxis], x_data)), delimiter=",")


            animated = True
            if not animated:
                for i, line_idx in enumerate(indices):
                    lines[line_idx].set_data(x_data[:, i], y_data[:, i])
                
            else:
                def animate(i):
                    if pause:
                        return *lines, line
                    for j, line_idx in enumerate(indices):
                        lines[line_idx].set_data(x_data[i, j], y_data[i, j])
                    line.set_data(times[:i], abs(energies[:i] - initial_energy)/abs(initial_energy))
                    return *lines, line

                anim = animation.FuncAnimation(fig, animate, frames=len(times), interval=40, blit=True)
                if save:
                    from matplotlib.animation import PillowWriter
                    anim.save(f"{self.name}.gif", dpi=150, writer=PillowWriter(fps=25))
                plt.show()
                return anim

            
            if mark_every_dt is not None:
                # find the indices of times that are just above multiples of mark_every_dt
                # to do that, calculate the remainder of the division of times by mark_every_dt
                # and find the indices of the elements that are at a local minimum:
                special_indices_dt = np.where(np.diff(np.mod(times, mark_every_dt)) < 0)[0] + 1
                x_special = x_data[special_indices_dt, :].flatten()
                y_special = y_data[special_indices_dt, :].flatten()
                special_points.set_offsets(np.array([x_special, y_special]).T)

            
            
            line.set_data(times, abs(energies - initial_energy)/abs(initial_energy))
            plt.show()

            return

        title = ax.text(0.5, 0.9, 'Initializing...', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

        def animate(i, adaptive):
            # If pbar.n is greater than t_end, stop the animation
            if pbar.n >= t_end:
                pbar.close()
                # stop the animation
                return *lines, title, line, special_points
            if pause:
                return *lines, title, line, special_points

            self.evolve(inplace=True)
            positions = self.phase_space[:self.phase_space.size//2].reshape((len(self.astro_objects), self.n_dim))
            # positions = [[x1, y1 (, z1)], [x2, y2 (, z2)], ...]
            
            # # get center of mass
            # center_of_mass_x = np.sum(self.phase_space[:self.phase_space.size//2:self.n_dim] * self.masses) / np.sum(self.masses)
            # center_of_mass_y = np.sum(self.phase_space[1:self.phase_space.size//2:self.n_dim] * self.masses) / np.sum(self.masses)
            # # print(center_of_mass_x, center_of_mass_y)
            # # get total momentum
            # total_momentum_x = np.sum(self.phase_space[self.phase_space.size//2::self.n_dim] * self.masses)
            # total_momentum_y = np.sum(self.phase_space[self.phase_space.size//2+1::self.n_dim] * self.masses)
            # # print(total_momentum_x, total_momentum_y)

            for i in range(len(self.astro_objects)):
                if i not in indices or keep_points in [None,0]:
                    x = np.array([])
                    y = np.array([])
                else:
                    x, y = lines[i].get_data()
                    if isinstance(keep_points, str) and keep_points == "all":
                        pass
                    elif isinstance(keep_points, int) and len(x) > keep_points:
                        x = x[-keep_points+1:]
                        y = y[-keep_points+1:]
                x = np.append(x[:], positions[i, 0])
                y = np.append(y[:], positions[i, 1])
                
                lines[i].set_data(x, y)
            pbar.update(self.step_size)

            ## Update the title. Step size is in scientific notation if it is smaller than 1E-3 or larger than 1E3
            if self.step_size < 1E-3 or self.step_size > 1E3:
                title.set_text(f"t = {pbar.n/365.2:.2f} years, dt = {self.step_size:.3E} days")
            else:
                title.set_text(f"t = {pbar.n/365.2:.2f} years, dt = {self.step_size:.3f} days")
            ## Update the normalized energy bars
            total_energy = self.get_total_energy()

            ## Update the energy plot
            total_energy = self.get_total_energy()

            x_data = line.get_xdata()
            y_data = line.get_ydata()

            new_time = x_data[-1] + self.step_size
            # if new_time is greater than an integer multiple of 365.2, but x_data[-1] is not, 
            # add a new black point in ax for the AstroObject with i=0 (the Sun)
            # make this black point always visible by setting the zorder to something high
            if mark_every_dt is not None and mark_every_dt > 0:
                if new_time % mark_every_dt < x_data[-1] % mark_every_dt:
                    for i in range(len(self.astro_objects)):
                        if indices is None or i in indices:
                            old_offsets = special_points.get_offsets()
                            new_offsets = np.concatenate((old_offsets, [[positions[i, 0], positions[i, 1]]]))
                            special_points.set_offsets(new_offsets)

            x_data = np.append(x_data, new_time)
            y_data = np.append(y_data, abs(total_energy - initial_energy)/abs(initial_energy))
            line.set_data(x_data, y_data)

            if adaptive:
                ## Adapt the step size
                self.adapt_step(relative_error=relative_error, inplace=True)

            return *lines, title, line, special_points

        anim = FuncAnimation(
            fig, 
            animate, 
            frames=None,
            init_func=None,
            fargs=(adaptive,),
            interval=1,
            repeat=False,
            blit=True,
        )

        plt.show()

        if save:
            anim.save(
                f"animation_{self.name}.gif", 
                writer="imagemagick", 
                fps=30,
            )

        pbar.close()


if __name__ == "__main__":

    # ###### Random solar system ######
    # solar_system = StarSystem.star_and_planet(
    #     star_mass=M_sun, 
    #     planet_mass=M_jupiter, 
    #     planet_period=11.862*365.25,
    #     step_size=50,
    # )

    #### bounded solar system ######
    # while True:
    #     solar_system = StarSystem.random_solar_system(
    #         n_objects=3,
    #         step_size=1E-6,
    #     )
    #     if solar_system.get_total_energy() < 0:
    #         break

    ###### Our solar system ######
    solar_system = StarSystem.our_solar_system(
        t0='2023-01-04',
        step_size=1E-3)



    ## Evolve the solar system for a number of days
    t = 0
    duration =  10 # years
    t_end = t + duration * 365.25

    # indices of the objects to plot trails
    # if None and real_time is False, plot trails for all objects
    # if None and real_time is True, don't plot any trails
    indices = [0,1,2,3,4]

    # keep_points is the number of points to keep in the plot for `indices`. If None, keep all. Ignored if real_time is False 
    keep_points = "all"

    # resample_every_dt is the number of days between each point when real_time is False
    # Default is 1 day. Ignored if real_time is True
    resample_every_dt = 0.5 # in days

    # animated is whether to animate the plot when real_time is False    
    animated=True

    # mark_every_dt is the number of days between each black point in the energy plot
    mark_every_dt = 4 * 365.25

    # real_time is whether to plot as you evolve or precompute the entire evolution and then plot
    # It is usually much faster to precompute the evolution and then plot
    # Note also that the size of the plot affects the speed of the animation if real_time is True
    real_time = True

    # relative_error is the relative error to use for adaptive step size
    # if you evolve quasi-steady systems, a value of ~1E-5 should be fine
    # if you evolve collisional systems, this should be much smaller, e.g. 1E-10
    relative_error = 1E-6

    ## Animate the orbits
    solar_system.plot_orbits(
        t, 
        t_end, 
        indices=indices, 
        keep_points=keep_points, 
        resample_every_dt=resample_every_dt,
        animated=animated,
        mark_every_dt=mark_every_dt, 
        real_time=real_time,
        save=True, 
        adaptive=True, 
        relative_error=relative_error)