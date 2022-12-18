#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:24:33 2018

@author: yannis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
import random

M_sun = 1.0             ## Mass of Earth's Sun in Solar Masses
G = 0.000296005155      ## Gravitational Constant in AU^3 per Solar Masses per Days^2
M_jupiter = 1/1047.93 * M_sun

def cm_color():
    import itertools
    # create a cyrcular generator of colors from plt.cm.tab10
    return itertools.cycle(plt.cm.tab10.colors)
    
colors = cm_color()

class AstroObject:    
    """
    Create an astrophysical object with defined mass 2-D position and 2-D velocity
    """
    def __init__(self, name, mass, radius=1, color=None, star_system=None, pos=None, vel=None):
        self.mass = mass
        self.name = name
        self.radius = radius
        if color is None:
            self.color = next(colors)
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


class StarSystem:
    """
    Class to simulate a solar system with stars and planets
    """
    def __init__(self, name, astro_objects=[], phase_space=None, masses=None, evolve_method="S4", step_size=0.001):
        self.name = str(name)
        self.n_dim = 2
        self.astro_objects = []
        self.step_size = step_size
        
        if evolve_method == "RK4":
            self.evolve = self.rk4
        elif evolve_method == "ME":
            self.evolve = self.modified_Euler
        elif evolve_method == "S2":
            self.evolve = self.symplectic2
        elif evolve_method == "S4":
            self.evolve = self.symplectic4
        else:
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
                    if astro_object.star_system != None:
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


    def add_astro_object(self, astro_object, pos=None, vel=None):
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
    
    def get_position(self, astro_object):
        """
        Return the position of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        return self.phase_space[index*self.n_dim:(index+1)*self.n_dim]

    def set_position(self, astro_object, pos):
        """
        Set the position of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        self.phase_space[index*self.n_dim:(index+1)*self.n_dim] = pos

    def set_positions(self, pos):
        """
        Set the position of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        self.phase_space[:self.phase_space.size//2] = pos

    def get_velocity(self, astro_object):
        """
        Return the velocity of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        return self.phase_space[(index+1)*self.n_dim:(index+2)*self.n_dim]

    def set_velocity(self, astro_object, vel):
        """
        Set the velocity of an AstroObject instance in the same star system as a numpy array of length self.n_dim
        """
        index = self.astro_objects.index(astro_object)
        self.phase_space[(index+1)*self.n_dim:(index+2)*self.n_dim] = vel

    def get_pairwise_separations(self):
        """
        Return the array of the pairwise separations of the list of AstroObject instances
        in the same star system. Each separation is an array of length equal to self.n_dim
        """
        num_objs = len(self.astro_objects)
        positions = self.phase_space[:self.phase_space.size//2].reshape(num_objs, self.n_dim)
        # separations[i,j] is the separation vector pointing from the jth object to the ith object,
        # i.e. separations[i,j] = positions[i] - positions[j]
        separations = positions[:,np.newaxis,:] - positions[np.newaxis,:,:] # shape (num_objs, num_objs, n_dim)
        distances = np.linalg.norm(separations,axis=2) # shape (num_objs, num_objs)
        return separations, distances

    def get_accelerations(self):
        """
        Return the array of the accelerations of the list of AstroObject instances
        in the same star system
        """
        separations, distances = self.get_pairwise_separations()
        
        G_over_r3 = G/distances**3
        acceleration_components = - G_over_r3[:,:,np.newaxis] * separations * self.masses[np.newaxis,:,np.newaxis]
        
        accelerations = np.nansum(acceleration_components,axis=1) 
        
        # flatten the array of accelerations
        accelerations = accelerations.flatten()
        # now the shape of accelerations is (num_objs * self.n_dim,)
        return accelerations

    def get_phase_space_derivatives(self):
        """
        Return the array of the derivatives of the phase space of the list of AstroObject instances
        in the same star system
        """
        velocities = self.phase_space[self.phase_space.size//2:]
        accelerations = self.get_accelerations()

        return np.concatenate((velocities, accelerations))

    def set_phase_space(self, phase_space):
        """
        Set the phase space of the star system
        """
        self.phase_space = phase_space

    def get_kinetic_energy(self):
        """
        Return the kinetic energy of the star system
        """
        velocities = self.phase_space[self.phase_space.size//2:].reshape(len(self.astro_objects), self.n_dim)
        kinetic_energy = 0.5 * np.sum(self.masses * np.sum(velocities**2, axis=1))
        return kinetic_energy

    def get_potential_energy(self):
        """
        Return the potential energy of the star system
        """
        _, distances = self.get_pairwise_separations()
        # note that distances includes the distance from each object to itself, which is zero.
        # so we should sum over only the upper triangular part of the matrix, excluding the diagonal
        energy = 0
        for i in range(len(self.astro_objects)):
            for j in range(i+1,len(self.astro_objects)):
                energy += - G * self.masses[i] * self.masses[j] / distances[i,j]
        return energy

    def get_total_energy(self):
        """
        Return the total energy of the star system
        """
        return self.get_kinetic_energy() + self.get_potential_energy()

    def rk4(self, inplace=True):
        """
        Implement RK4 to evolve myRi by h
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

    def modified_Euler(self, inplace=True):
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

    def symplectic2(self, inplace=True):
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

    def symplectic4(self, inplace=True):
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

        positions, velocities = self.phase_space.copy()[:self.phase_space.size//2], self.phase_space.copy()[self.phase_space.size//2:]

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
    
    def adaptStep(self, relError=1E-6, inplace=True):
        """
        Adapt the step size to reduce local error to some specified relative error (in our case 1E-3)
        The relative error is specifically calculated on the total energy of the system
        """
        h = self.step_size
        ## Evolve with two steps of h
        original_configurations = self.phase_space.copy()
        self.evolve(inplace=True)
        self.evolve(inplace=True)
        positions = self.phase_space.copy()[:self.phase_space.size//2].reshape(len(self.astro_objects), self.n_dim)
        energy1 = self.get_total_energy()

        self.set_phase_space(original_configurations)
            
        ## Evolve with one step of 2*h
        self.step_size = 2*h
        self.evolve(inplace=True)
        energy2 = self.get_total_energy()
        positions2 = self.phase_space.copy()[:self.phase_space.size//2].reshape(len(self.astro_objects), self.n_dim)
        self.set_phase_space(original_configurations)
        self.step_size = h

        
        
        ## Calculate the maximum relative distance, defined as the maximum of sqrt((x1-x2)^2 + (y1-y2)^2) / sqrt(x1^2 + y1^2)
        # where (x1,y1) and (x2,y2) are the positions achieved with two steps of h and one step of 2*h
        position_differences = positions - positions2
        norm_positions = np.linalg.norm(positions, axis=1)
        norm_position_differences = np.linalg.norm(position_differences, axis=1)

        maxRelDist = np.max(norm_position_differences / norm_positions)
                
        h = 0.85 * h * (relError / maxRelDist )**(1/5)

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

        solar_system = cls(
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
    def our_solar_system(cls, step_size=1E-3):
        """
        Create the inner solar system
        """
        masses_p = np.array([3.285E23, 4.867E24, 5.972E24, 6.39E23, 1.898E27, 5.683E26, 8.681E25, 1.024E26])
        radius_p = np.array([2439.7, 6051.8, 6371, 3389.5, 69911, 58232, 25362, 24622])
        R_sun = 695700
        radiuses = np.concatenate(([R_sun], radius_p))/R_sun
        radius_p = radius_p / max(radius_p)
        masses_p = masses_p / 1.989E30
        masses = np.concatenate(([1], masses_p))
        print(masses)
        n_objects = len(masses)
        periods_p = np.array([88, 224.7, 365.2, 686.98, 4332, 10759, 30685, 60190])
        names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        colors = ["#808080", "#FFA500", "#0000FF", "#FF0000", "#D0B49F", "#FFA500", "#00FFFF", "#0000FF"]

        semimajor = (G * M_sun * periods_p**2 / (2*np.pi)**2)**(1/3)
        initial_phases = np.random.rand(len(masses_p)) * 2 * np.pi
        initial_positions_x = semimajor * np.cos(initial_phases)
        initial_positions_y = semimajor * np.sin(initial_phases)
        # create the initial positions as an array [x1, y1, x2, y2, ...]
        initial_positions = np.concatenate((initial_positions_x[None,:], initial_positions_y[None,:])).T.flatten()
        speeds = 2 * np.pi * semimajor / periods_p
        initial_velocities_x = - speeds * np.sin(initial_phases)
        initial_velocities_y = speeds * np.cos(initial_phases)
        # create the initial velocities as an array [vx1, vy1, vx2, vy2, ...]
        initial_velocities = np.concatenate((initial_velocities_x[None,:], initial_velocities_y[None,:])).T.flatten()
        print(initial_velocities)
        # raise
        phase_space = np.concatenate(
            (np.array([0, 0]),
            initial_positions,
            np.array([0, 0]),
            initial_velocities)
        )
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

        astro_objects = [AstroObject(mass=M_sun, name="Sun", radius=1, color="#FFFF00")]
        for i in range(len(masses_p)):
            astro_objects.append(AstroObject(
                mass=masses_p[i], 
                name=names[i],
                radius=radius_p[i],
                color=colors[i],
            ))


        solar_system = cls(
            name="Our solar system",
            astro_objects=astro_objects,
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


    def plot_orbits_animation(self, t_0, t_end, save=False, adaptive=True):
        """
        Plot the animation of the solar system
        Add a pause/continue button to the animation
        Add a progress bar in the animation subtitle using tqdm
        # add three bars to the animation subtitle, one for the total energy, total kinetic energy and total potential energy
        # the bars should be colored according to the energy:
        #   - total energy: red
        #   - total kinetic energy: green
        #   - total potential energy: blue
        # the bars should be normalized to the total energy
        # the bars should be updated every step
        # they are created via ax.bar([0, 1, 2], [total_energy, total_kinetic_energy, total_potential_energy], color=["red", "green", "blue"])
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig = plt.figure(figsize=(3,3))
        # add two axes to the figure, one for the animation and one for the bars
        # make the animation axis fill the figure, while the bars should be placed in the lower left corner
        ax = fig.add_axes([0, 0, 1, 1])
        # set the limits of the animation axis
        max_coord = 13*np.max(np.abs(self.phase_space[:self.phase_space.size//2]))
        ax.set_xlim(-max_coord, max_coord)
        ax.set_ylim(-max_coord, max_coord)
        ax.set_aspect("auto")
        ax.grid()

        ax_bars = fig.add_axes([0.3, 0.1, 0.5, 0.2])
        ax_bars.set_ylim(-1.3, 1.3)

        # the default plt marker size is rcParams['lines.markersize'] ** 2. 
        # We want to scale the marker size with the radius of the object
        marker_size = np.array([max(obj.radius,0.5) for obj in self.astro_objects]) ** 2 * 5

        ## Create the scatter plot
        positions = self.phase_space[:self.phase_space.size//2].reshape((len(self.astro_objects), self.n_dim))
        lines = []
        for i in range(len(self.astro_objects)):
            line, = ax.plot(
                positions[i, 0],
                positions[i, 1],
                "o",
                markersize=marker_size[i],
                label=self.astro_objects[i].name,
                color=self.astro_objects[i].color,
            )
            lines.append(line)

        # add the bars to the axes with x-labels T, K, P
        bars = ax_bars.bar(
            [0, 1, 2],
            [self.get_total_energy(), self.get_kinetic_energy(), self.get_potential_energy()],
            color=["red", "green", "blue"],
        )
        ax_bars.set_xticks([0, 1, 2])
        ax_bars.set_xticklabels(["T", "K", "P"])
        # place the bars in the lower left corner of the axes
        y_bar_limits = max(abs(self.get_total_energy()), abs(self.get_kinetic_energy()), abs(self.get_potential_energy()))
        ax_bars.set_ylim(-1.3*y_bar_limits, 1.3*y_bar_limits)
        #  since G is in units of AU^3 / (Msun * day^2), the energy ([Msun * AU^2 / day^2]) is in units of Msun * AU^2 / day^2
        print(f"Total energy: {self.get_total_energy()} [Msun * AU^2 / day^2]")

        ## Create the legend
        ax.legend()

        ## Create the pause/continue button
        pause = False
        def onClick(event):
            nonlocal pause
            pause ^= True
        fig.canvas.mpl_connect("button_press_event", onClick)

        ## Create the progress bar
        from tqdm import tqdm
        pbar = tqdm(total=t_end-t_0)

        # add an artist to the axes to update the title
        # set the location of the artist to the upper center of the axes

        title = ax.text(0.5, 0.9, 'matplotlib', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

        def animate(i, adaptive):
            # If pbar.n is greater than t_end, stop the animation
            if pbar.n >= t_end:
                pbar.close()
                # stop the animation
                return *lines, title, *bars
            if pause:
                return *lines, title, *bars

            self.evolve(inplace=True)
            positions = self.phase_space[:self.phase_space.size//2].reshape((len(self.astro_objects), self.n_dim))
            # positions = [[x1, y1], [x2, y2], ...]
            
            # get center of mass
            center_of_mass_x = np.sum(self.phase_space[:self.phase_space.size//2:2] * self.masses) / np.sum(self.masses)
            center_of_mass_y = np.sum(self.phase_space[1:self.phase_space.size//2:2] * self.masses) / np.sum(self.masses)
            # print(center_of_mass_x, center_of_mass_y)
            # get total momentum
            total_momentum_x = np.sum(self.phase_space[self.phase_space.size//2::2] * self.masses)
            total_momentum_y = np.sum(self.phase_space[self.phase_space.size//2+1::2] * self.masses)
            # print(total_momentum_x, total_momentum_y)

            for i in range(len(self.astro_objects)):
                # TODO: add artist for the tail with a varying alpha
                lines[i].set_data(positions[i, 0], positions[i, 1])

            pbar.update(self.step_size)

            ## Update the title. Step size is in scientific notation if it is smaller than 1E-3 or larger than 1E3
            if self.step_size < 1E-3 or self.step_size > 1E3:
                title.set_text(f"t = {pbar.n:.2f} days, step_size = {self.step_size:.3E} days")
            else:
                title.set_text(f"t = {pbar.n:.2f} days, step_size = {self.step_size:.3f} days")
            ## Update the normalized energy bars
            total_energy = self.get_total_energy()
            total_kinetic_energy = self.get_kinetic_energy()
            total_potential_energy = self.get_potential_energy()

            print(total_energy, total_kinetic_energy, total_potential_energy)
            
            bars[0].set_height(total_energy)
            bars[1].set_height(total_kinetic_energy)
            bars[2].set_height(total_potential_energy)



            if adaptive:
                ## Adapt the step size
                self.adaptStep(relError=1E-5, inplace=True)

            return *lines, title, *bars

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

    def plot_energy_evolution(self, t_0, t_end, save=False, adaptive=True):
        """Plots the relative energy error of the system as a function of time.

        Parameters
        ----------
        t_0 : float
            The initial time.
        t_end : float
            The final time.
        save : bool, optional
            Whether to save the animation as a gif, by default False
        """

        ## Create the wide and short figure
        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(1, 1, 1)

        # the initial point is at x=t_0 and y=0
        initial_energy = self.get_total_energy()
        x = [t_0]
        y = [0]

        # create an artist to update the line
        line, = ax.plot(
            x, 
            y, 
            "k-",
            color="black", 
            label="Total energy")

        # set the scale of the y-axis to be logarithmic
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1E-20, 1E3)
        ax.set_xlim(t_0, t_end)

        # ax.axhline(y=1, color="red", linestyle="--")
        ## Create the pause/continue button
        pause = False
        def onClick(event):
            nonlocal pause
            pause ^= True
        fig.canvas.mpl_connect("button_press_event", onClick)

        ## Create the progress bar
        from tqdm import tqdm
        pbar = tqdm(total=t_end-t_0)

        # add an artist to the axes to update the title
        # set the location of the artist to the upper center of the axes

        title = ax.text(0.5, 0.9, 'matplotlib', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

        def animate(i,adaptive):
            # If pbar.n is greater than t_end, stop the animation
            if pbar.n >= t_end:
                pbar.close()
                # stop the animation
                return line, title
            if pause:
                return line, title

            self.evolve(inplace=True)

            pbar.update(self.step_size)

            ## Update the title. Step size is in scientific notation if it is smaller than 1E-3 or larger than 1E3
            if self.step_size < 1E-3 or self.step_size > 1E3:
                title.set_text(f"t = {pbar.n:.2f} days, step_size = {self.step_size:.3E} days")
            else:
                title.set_text(f"t = {pbar.n:.2f} days, step_size = {self.step_size:.3f} days")
            ## Update the normalized energy bars
            total_energy = self.get_total_energy()

            x_data = line.get_xdata()
            y_data = line.get_ydata()

            new_time = x_data[-1] + self.step_size
            x_data = np.append(x_data, new_time)
            y_data = np.append(y_data, abs(total_energy - initial_energy)/abs(initial_energy))
            line.set_data(x_data, y_data)

            if adaptive:
                ## Adapt the step size
                self.adaptStep(relError=1E-5, inplace=True)

            return line, title

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=None,
            init_func=None,
            interval=1,
            fargs=(adaptive,),
            repeat=False,
            blit=True,
        )

        plt.show()

        if save:
            anim.save(
                f"energy_evolution_{self.name}.gif", 
                writer="imagemagick", 
                fps=30,
            )

        pbar.close()

        


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ###### Random solar system ######
    # solar_system = StarSystem.star_and_planet(
    #     star_mass=M_sun, 
    #     planet_mass=M_jupiter, 
    #     planet_period=11.862*365.25,
    #     step_size=50,
    # )

    ###### bounded solar system ######
    # while True:
    #     solar_system = StarSystem.random_solar_system(
    #         n_objects=5,
    #         step_size=1E-6,
    #     )
    #     if solar_system.get_total_energy() < 0:
    #         break

    ###### Our solar system ######
    solar_system = StarSystem.our_solar_system(step_size=1E-3)



    ## Evolve the solar system for a number of days
    t = 0
    t_end = 10000

    ## Animate the orbits
    solar_system.plot_orbits_animation(t, t_end, save=False, adaptive=True)

    ## Plot the energy evolution
    # solar_system.plot_energy_evolution(t, t_end, save=False, adaptive=True)

