"""Energy calculations for orbital mechanics simulations."""

import numpy as np
from ..core.constants import G


class EnergyMixin:
    """Mixin class providing energy calculation methods for StarSystem."""
    
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