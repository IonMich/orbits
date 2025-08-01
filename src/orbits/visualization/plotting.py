"""Advanced plotting capabilities for orbital mechanics simulations."""

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


class PlottingMixin:
    """Mixin class providing advanced plotting capabilities for StarSystem objects."""
    
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