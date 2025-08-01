"""Numerical integration methods for orbital mechanics."""

import numpy as np


class IntegratorMixin:
    """Mixin class providing numerical integration methods for StarSystem."""
    
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