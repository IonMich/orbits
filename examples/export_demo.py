#!/usr/bin/env python3
"""Demo of export functionality - both static and streaming."""

import asyncio
import orbits
from orbits.core.constants import M_sun, M_jupiter


def demo_static_export():
    """Demonstrate static trajectory export."""
    print("🌟 Static Export Demo")
    print("=" * 50)
    
    # Create a Jupiter-Sun system
    system = orbits.StarSystem.star_and_planet(
        star_mass=M_sun,
        planet_mass=M_jupiter,
        planet_period=11.862 * 365.25,  # Jupiter's period
        step_size=10.0
    )
    
    print(f"Created system: {system.name}")
    
    # Export to JSON
    print("\n📄 Exporting trajectory to JSON...")
    data = system.export_json(
        "jupiter_orbit.json",
        duration_days=365.25,  # 1 year
        time_step=5.0
    )
    
    print(f"✓ Exported {len(data['trajectory'])} frames")
    print(f"✓ File size: ~{len(str(data)) / 1024:.1f} KB")
    
    # Export compressed version for longer simulation
    print("\n🗜️  Exporting compressed trajectory...")
    compressed_data = system.export_json_compressed(
        "jupiter_orbit_long.json",
        duration_days=11.862 * 365.25,  # Full Jupiter orbit
        max_frames=1000
    )
    
    print(f"✓ Compressed {compressed_data['simulation']['original_frames']} frames to {len(compressed_data['trajectory'])}")
    print(f"✓ Compression ratio: {compressed_data['simulation']['compression_ratio']:.1f}x")


def demo_streaming():
    """Demonstrate WebSocket streaming."""
    print("\n\n🌐 Streaming Demo")
    print("=" * 50)
    
    # Create an Earth-Sun system for faster motion
    system = orbits.StarSystem.star_and_planet(
        star_mass=M_sun,
        planet_mass=3e-6 * M_sun,  # Earth mass
        planet_period=365.25,
        step_size=0.1  # Small step for smooth animation
    )
    
    print(f"Created system: {system.name}")
    print("Starting WebSocket streamer...")
    
    # Create streamer
    streamer = system.start_stream(port=8765, fps=60)
    
    print(f"🚀 Streamer ready on port {streamer.port}")
    print("Connect a WebSocket client to ws://localhost:8765")
    print("Press Ctrl+C to stop streaming")
    
    try:
        # Run the streaming server
        streamer.run()
    except KeyboardInterrupt:
        print("\n🛑 Streaming stopped by user")


def demo_server():
    """Demonstrate full server with both static and streaming."""
    print("\n\n🖥️  Full Server Demo")
    print("=" * 50)
    
    # Create our solar system (simplified)
    system = orbits.StarSystem.our_solar_system(step_size=0.5)
    
    print(f"Created system: {system.name}")
    print("Starting full server with both static API and WebSocket streaming...")
    print("\nEndpoints available:")
    print("  📄 Static API:")
    print("    http://localhost:8000/api/metadata")
    print("    http://localhost:8000/api/trajectory?duration=365&max_frames=1000")
    print("  🌐 WebSocket streaming:")
    print("    ws://localhost:8000/ws")
    print("\nPress Ctrl+C to stop server")
    
    try:
        # Start the server (blocking call)
        system.start_server(host="localhost", port=8000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")


def main():
    """Run all demos."""
    print("🚀 Orbits Export Functionality Demo")
    print("=" * 60)
    
    # Static export demo
    demo_static_export()
    
    # Choose demo mode
    print("\n\nChoose demo mode:")
    print("1. WebSocket Streaming")
    print("2. Full Server (API + WebSocket)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            demo_streaming()
        elif choice == "2":
            demo_server()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")


if __name__ == "__main__":
    main()