"""Core StarSystem class for orbital mechanics simulations."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .constants import G
from .objects import AstroObject
from .integrators import IntegratorMixin
from ..physics.energy import EnergyMixin
from ..presets.factories import SystemFactory


class StarSystem(IntegratorMixin, EnergyMixin):
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

        fig = plt.figure(figsize=(8,8))
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

    # Factory methods as class methods
    @classmethod
    def star_and_planet(cls, star_mass, planet_mass, planet_period, step_size=1E-3):
        return SystemFactory.star_and_planet(cls, star_mass, planet_mass, planet_period, step_size)

    @classmethod
    def our_solar_system(cls, step_size=1E-3):
        return SystemFactory.our_solar_system(cls, step_size)

    @classmethod        
    def random_solar_system(cls, n_objects, step_size=1E-3):
        return SystemFactory.random_solar_system(cls, n_objects, step_size)