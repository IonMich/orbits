#!/usr/bin/env python3
"""Demo of the refactored orbits package with clean API."""

import orbits
from orbits.core.constants import M_sun, M_jupiter

def main():
    print("🌟 Orbits Package Demo")
    print("=" * 50)
    
    # Create a Jupiter-Sun system using the clean API
    print("Creating a star-planet system...")
    system = orbits.StarSystem.star_and_planet(
        star_mass=M_sun,
        planet_mass=M_jupiter,
        planet_period=11.862 * 365.25,  # Jupiter's orbital period in days
        step_size=10.0
    )
    
    print(f"✓ System created: {system.name}")
    print(f"✓ Objects: {[obj.name for obj in system.astro_objects]}")
    print(f"✓ Initial total energy: {system.get_total_energy():.6e} [M☉ AU²/day²]")
    
    # Simulate for a short period
    print("\nSimulating orbital evolution...")
    initial_energy = system.get_total_energy()
    
    for day in range(0, 100, 10):
        system.evolve()
        energy = system.get_total_energy()
        energy_error = abs(energy - initial_energy) / abs(initial_energy)
        print(f"Day {day:3d}: Energy = {energy:.6e}, Error = {energy_error:.2e}")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    main()