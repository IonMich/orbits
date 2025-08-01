"""Test that refactored code works identically to original."""

import pytest
import numpy as np
from orbits import StarSystem, AstroObject, G, M_sun, M_jupiter


class TestRefactoredCode:
    """Test suite to verify refactored code maintains original functionality."""
    
    def test_imports(self):
        """Test that all main components can be imported."""
        assert StarSystem is not None
        assert AstroObject is not None
        assert G > 0
        assert M_sun > 0
        assert M_jupiter > 0
    
    def test_astro_object_creation(self):
        """Test AstroObject creation."""
        obj = AstroObject(name="Test", mass=1.0, radius=2.0)
        assert obj.name == "Test"
        assert obj.mass == 1.0
        assert obj.radius == 2.0
        assert obj.color is not None
    
    def test_star_system_creation(self):
        """Test StarSystem creation with objects."""
        star = AstroObject(name="Star", mass=M_sun)
        planet = AstroObject(name="Planet", mass=M_jupiter)
        
        system = StarSystem(
            name="Test System",
            astro_objects=[star, planet],
            phase_space=np.zeros(8),  # 2D, 2 objects
            step_size=0.01
        )
        
        assert system.name == "Test System"
        assert len(system.astro_objects) == 2
        assert system.step_size == 0.01
    
    def test_star_and_planet_factory(self):
        """Test the star_and_planet factory method."""
        system = StarSystem.star_and_planet(
            star_mass=M_sun,
            planet_mass=M_jupiter,
            planet_period=365.25,
            step_size=0.1
        )
        
        assert system.name == "Star and planet"
        assert len(system.astro_objects) == 2
        assert system.astro_objects[0].name == "Star"
        assert system.astro_objects[1].name == "Planet"
    
    def test_energy_conservation(self):
        """Test that energy is conserved during evolution."""
        system = StarSystem.star_and_planet(
            star_mass=M_sun,
            planet_mass=M_jupiter,
            planet_period=365.25,
            step_size=0.1
        )
        
        initial_energy = system.get_total_energy()
        
        # Evolve for several steps
        for _ in range(10):
            system.evolve()
        
        final_energy = system.get_total_energy()
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        
        # Energy should be conserved to high precision
        assert relative_error < 1e-6
    
    def test_integrator_methods_exist(self):
        """Test that all integrator methods are available."""
        system = StarSystem.star_and_planet(M_sun, M_jupiter, 365.25)
        
        # Test that all integration methods exist
        assert hasattr(system, 'rk4')
        assert hasattr(system, 'modified_Euler')
        assert hasattr(system, 'symplectic2')
        assert hasattr(system, 'symplectic4')
        assert hasattr(system, 'adaptStep')
    
    def test_energy_methods_exist(self):
        """Test that energy calculation methods are available."""
        system = StarSystem.star_and_planet(M_sun, M_jupiter, 365.25)
        
        # Test that energy methods exist and return values
        kinetic = system.get_kinetic_energy()
        potential = system.get_potential_energy()
        total = system.get_total_energy()
        
        assert isinstance(kinetic, (int, float))
        assert isinstance(potential, (int, float))
        assert isinstance(total, (int, float))
        assert abs(total - (kinetic + potential)) < 1e-10
    
    def test_phase_space_operations(self):
        """Test phase space manipulation methods."""
        system = StarSystem.star_and_planet(M_sun, M_jupiter, 365.25)
        
        # Test getting and setting positions/velocities
        star = system.astro_objects[0]
        original_pos = system.get_position(star).copy()
        original_vel = system.get_velocity(star).copy()
        
        # Modify position
        new_pos = original_pos + np.array([0.1, 0.1])
        system.set_position(star, new_pos)
        
        # Verify the change
        retrieved_pos = system.get_position(star)
        np.testing.assert_array_almost_equal(retrieved_pos, new_pos)
    
    def test_pairwise_separations(self):
        """Test pairwise separation calculations."""
        system = StarSystem.star_and_planet(M_sun, M_jupiter, 365.25)
        
        separations, distances = system.get_pairwise_separations()
        
        # Check shapes
        n_objects = len(system.astro_objects)
        assert separations.shape == (n_objects, n_objects, system.n_dim)
        assert distances.shape == (n_objects, n_objects)
        
        # Diagonal distances should be zero
        np.testing.assert_array_almost_equal(np.diag(distances), 0)
        
        # Distance matrix should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)