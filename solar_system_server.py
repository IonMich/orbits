#!/usr/bin/env python3
"""Unified server providing both pre-computed simulations and streaming from Python physics."""

import orbits
from orbits.core.constants import M_sun
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

def create_simulation_endpoints(server):
    """Add custom simulation endpoints to the existing OrbitServer."""
    
    @server.app.get("/api/simulation/earth-sun")
    async def get_earth_sun_simulation():
        """Get Earth-Sun system simulation computed with Python physics."""
        try:
            print("🌍 Computing Earth-Sun system...")
            system = orbits.StarSystem.star_and_planet(
                star_mass=M_sun,
                planet_mass=3e-6 * M_sun,  # Earth mass in solar masses
                planet_period=365.25,
                step_size=1.0
            )
            
            # Export trajectory data
            from tempfile import NamedTemporaryFile
            import os
            
            with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                data = system.export_json_compressed(
                    tmp.name,
                    duration_days=730,  # 2 years
                    max_frames=1000
                )
            
            os.unlink(tmp.name)  # Clean up temp file
            print("✓ Earth-Sun simulation computed")
            return data
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Earth-Sun simulation failed: {error_details}")
            raise HTTPException(status_code=500, detail=f"Failed to compute Earth-Sun simulation: {str(e)}")

    @server.app.get("/api/simulation/solar-system")
    async def get_solar_system_simulation():
        """Get our solar system simulation using real NASA data computed with Python physics."""
        try:
            print("☀️ Computing solar system with NASA data...")
            system = orbits.StarSystem.our_solar_system(step_size=1.0)
            
            # Export trajectory data
            from tempfile import NamedTemporaryFile
            import os
            
            with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                data = system.export_json_compressed(
                    tmp.name,
                    duration_days=1095,  # 3 years
                    max_frames=2000
                )
            
            os.unlink(tmp.name)  # Clean up temp file
            print("✓ Solar system simulation computed")
            return data
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Solar system simulation failed: {error_details}")
            raise HTTPException(status_code=500, detail=f"Failed to compute solar system simulation: {str(e)}")

    @server.app.get("/api/simulation/random")
    async def get_random_system_simulation():
        """Get a random planetary system simulation computed with Python physics."""
        try:
            print("🎲 Computing random planetary system...")
            import random
            n_objects = random.randint(4, 8)  # 4-8 objects including star
            
            system = orbits.StarSystem.random_solar_system(
                n_objects=n_objects,
                step_size=0.5
            )
            
            # Export trajectory data
            from tempfile import NamedTemporaryFile
            import os
            
            with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                data = system.export_json_compressed(
                    tmp.name,
                    duration_days=1000,  # ~2.7 years
                    max_frames=1500
                )
            
            os.unlink(tmp.name)  # Clean up temp file
            print(f"✓ Random system simulation computed ({n_objects} objects)")
            return data
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Random system simulation failed: {error_details}")
            raise HTTPException(status_code=500, detail=f"Failed to compute random system simulation: {str(e)}")

def main():
    print("🚀 Starting Unified Orbits Server")
    print("=" * 50)
    
    # Create a default solar system for streaming mode
    print("Creating default solar system for streaming...")
    default_system = orbits.StarSystem.our_solar_system(step_size=0.05)
    
    # Create the OrbitServer (includes built-in streaming WebSocket support)
    server = default_system.serve(host="localhost", port=8000)
    
    # Add CORS middleware for frontend access
    server.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add our custom simulation endpoints
    create_simulation_endpoints(server)
    
    print("📡 Available endpoints:")
    print("    Pre-computed simulations:")
    print("      http://localhost:8000/api/simulation/earth-sun")
    print("      http://localhost:8000/api/simulation/solar-system")
    print("      http://localhost:8000/api/simulation/random")
    print("    Built-in endpoints:")
    print("      http://localhost:8000/api/metadata")
    print("      http://localhost:8000/api/trajectory")
    print("    WebSocket streaming:")
    print("      ws://localhost:8000/ws")
    print("\nPress Ctrl+C to stop server")
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")

if __name__ == "__main__":
    main()