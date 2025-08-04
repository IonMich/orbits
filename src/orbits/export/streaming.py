"""Real-time streaming functionality via WebSocket."""

import asyncio
import json
import time
from typing import Set, Dict, Any, Optional, Union
import websockets
from websockets.server import WebSocketServerProtocol

# FastAPI WebSocket support
try:
    from fastapi import WebSocket as FastAPIWebSocket
except ImportError:
    FastAPIWebSocket = None

from .base import StreamingExporter


class WebSocketStreamer(StreamingExporter):
    """Stream real-time simulation data via WebSocket."""
    
    def __init__(self, star_system, target_fps: int = 60, port: int = 8765):
        super().__init__(star_system, target_fps)
        self.port = port
        self.clients: Set[Union[WebSocketServerProtocol, "FastAPIWebSocket"]] = set()
        self.server = None
        self.streaming_task = None
        self.is_playing = False
        self.playback_speed = 1.0
        self._current_simulation_type = 'solar-system'  # Track current simulation
        self._initial_phase_space = None  # Store initial state for fast reset
        
    async def register_client(self, websocket: Union[WebSocketServerProtocol, "FastAPIWebSocket"]):
        """Register a new client connection."""
        self.clients.add(websocket)
        print(f"✓ Client connected. Total clients: {len(self.clients)}")
        
        # Send initial metadata to new client
        initial_data = {
            "type": "metadata",
            "objects": self.get_object_metadata(),
            "units": self.get_units_info(),
            "fps": self.target_fps
        }
        await self._send_to_client(websocket, json.dumps(initial_data))
    
    async def _send_to_client(self, websocket: Union[WebSocketServerProtocol, "FastAPIWebSocket"], message: str):
        """Send message to client, handling both websocket types."""
        if FastAPIWebSocket and isinstance(websocket, FastAPIWebSocket):
            # FastAPI WebSocket
            await websocket.send_text(message)
        else:
            # websockets library WebSocket
            await websocket.send(message)
    
    async def unregister_client(self, websocket: Union[WebSocketServerProtocol, "FastAPIWebSocket"]):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        print(f"✗ Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_frame(self, frame_data: Dict[str, Any]):
        """Broadcast frame data to all connected clients."""
        if not self.clients:
            return
            
        frame_message = {
            "type": "frame",
            **frame_data
        }
        message = json.dumps(frame_message)
        
        # Send to all clients (remove disconnected ones)
        disconnected = set()
        for client in self.clients:
            try:
                await self._send_to_client(client, message)
            except Exception as e:
                # Handle both websockets.exceptions.ConnectionClosed and FastAPI disconnections
                print(f"Client disconnected: {e}")
                disconnected.add(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            await self.unregister_client(client)
    
    async def handle_client_message(self, websocket: Union[WebSocketServerProtocol, "FastAPIWebSocket"], message: str):
        """Handle messages from clients."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "ping":
                await self._send_to_client(websocket, json.dumps({"type": "pong"}))
            elif msg_type == "request_metadata":
                metadata = {
                    "type": "metadata",
                    "objects": self.get_object_metadata(),
                    "units": self.get_units_info(),
                    "fps": self.target_fps
                }
                await self._send_to_client(websocket, json.dumps(metadata))
            elif msg_type == "change_simulation":
                simulation_type = data.get("simulation_type")
                if simulation_type in ["earth-sun", "solar-system", "random"]:
                    await self.change_simulation_type(simulation_type)
                    # Send new metadata after simulation change
                    metadata = {
                        "type": "metadata",
                        "objects": self.get_object_metadata(),
                        "units": self.get_units_info(),
                        "fps": self.target_fps
                    }
                    await self._send_to_client(websocket, json.dumps(metadata))
                else:
                    print(f"Invalid simulation type: {simulation_type}")
            elif msg_type == "play":
                self.is_playing = True
                print("▶️ Simulation playing")
            elif msg_type == "pause":
                self.is_playing = False
                print("⏸️ Simulation paused")
            elif msg_type == "reset":
                await self.reset_simulation()
                print("🔄 Simulation reset")
            elif msg_type == "set_speed":
                speed = data.get("speed", 1.0)
                self.playback_speed = float(speed)
                print(f"⚡ Speed set to {speed}x")
            else:
                print(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON message: {message}")
    
    async def change_simulation_type(self, simulation_type: str):
        """Change the simulation type dynamically."""
        print(f"🔄 Changing simulation to: {simulation_type}")
        
        # Import here to avoid circular imports
        import orbits
        
        # Create new star system based on type
        if simulation_type == "earth-sun":
            from orbits.core.constants import M_sun
            M_earth = M_sun / 333000  # Earth's mass is about 1/333,000 of the Sun's mass
            new_system = orbits.StarSystem.star_and_planet(
                star_mass=M_sun, 
                planet_mass=M_earth, 
                planet_period=365.25,  # Earth's orbital period in days
                step_size=0.05
            )
        elif simulation_type == "solar-system":
            new_system = orbits.StarSystem.our_solar_system(step_size=0.05)
        elif simulation_type == "random":
            import random
            n_objects = random.randint(4, 8)
            new_system = orbits.StarSystem.random_solar_system(n_objects=n_objects, step_size=0.05)
        else:
            print(f"❌ Unknown simulation type: {simulation_type}")
            return
        
        # Replace the current star system
        self.star_system = new_system
        self._current_simulation_type = simulation_type  # Track current type
        
        # Store initial state for fast reset
        self._initial_phase_space = new_system.phase_space.copy()
        
        print(f"✅ Successfully changed to {simulation_type} simulation")
    
    async def reset_simulation(self):
        """Reset the simulation to initial state."""
        if self._initial_phase_space is not None:
            # Fast reset: restore initial phase space
            self.star_system.phase_space = self._initial_phase_space.copy()
        else:
            # Fallback: recreate system (slower)
            current_system_type = getattr(self, '_current_simulation_type', 'solar-system')
            await self.change_simulation_type(current_system_type)
        
        # Reset streaming state and auto-pause
        self.current_frame = 0
        self.current_time = 0.0
        self.is_playing = False
        print("🔄 Simulation reset to initial state")
    
    def get_current_frame(self) -> Dict[str, Any]:
        """Get current frame data without advancing simulation."""
        return self.get_frame_data(self.current_frame, self.current_time)
    
    def get_next_frame(self) -> Dict[str, Any]:
        """Override to respect is_playing state."""
        if not self.is_streaming:
            raise RuntimeError("Streaming not started. Call start_streaming() first.")
        
        # Get current frame data
        frame_data = self.get_frame_data(self.current_frame, self.current_time)
        
        # Only advance simulation if playing
        if self.is_playing:
            # Advance simulation
            self.star_system.evolve()
            self.current_time += self.star_system.step_size
            self.current_frame += 1
        
        return frame_data
    
    async def client_handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle individual client connections."""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def streaming_loop(self):
        """Main streaming loop - advances simulation and broadcasts frames."""
        self.start_streaming()
        
        frame_duration = 1.0 / self.target_fps  # seconds per frame
        last_time = time.time()
        skip_counter = 0  # For handling speeds < 1.0
        
        print(f"🚀 Starting streaming loop at {self.target_fps} FPS")
        
        try:
            while self.is_streaming:
                loop_start = time.time()
                
                try:
                    # Handle different playback speeds
                    should_advance = False
                    
                    if self.is_playing:
                        if self.playback_speed >= 1.0:
                            # Speed >= 1.0: advance normally or multiple times
                            should_advance = True
                        else:
                            # Speed < 1.0: skip some frames
                            skip_interval = int(1.0 / self.playback_speed)
                            skip_counter += 1
                            if skip_counter >= skip_interval:
                                should_advance = True
                                skip_counter = 0
                    
                    # Get frame data (advances if should_advance)
                    if should_advance:
                        frame_data = self.get_next_frame()
                        
                        # Apply speed multiplier for speeds > 1.0
                        if self.playback_speed > 1.0:
                            extra_steps = int(self.playback_speed) - 1
                            for _ in range(extra_steps):
                                # Advance simulation extra times for speed
                                self.star_system.evolve()
                                self.current_time += self.star_system.step_size
                                self.current_frame += 1
                            # Update frame data with final position
                            frame_data = self.get_frame_data(self.current_frame, self.current_time)
                    else:
                        # Just get current frame without advancing
                        frame_data = self.get_current_frame()
                    
                    # Always broadcast current frame
                    await self.broadcast_frame(frame_data)
                    
                except Exception as e:
                    import traceback
                    print(f"❌ Error getting/broadcasting frame: {e}")
                    print(f"Full traceback: {traceback.format_exc()}")
                    break
                
                # Calculate how long to sleep to maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_duration - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    # Running behind - could print warning for debugging
                    pass
                    
        except Exception as e:
            import traceback
            print(f"❌ Error in streaming loop: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
        finally:
            self.stop_streaming()
            print("🛑 Streaming loop stopped")
    
    async def start_server(self):
        """Start the WebSocket server."""
        print(f"🌐 Starting WebSocket server on port {self.port}")
        
        self.server = await websockets.serve(
            self.client_handler,
            "localhost", 
            self.port
        )
        
        # Start streaming loop
        self.streaming_task = asyncio.create_task(self.streaming_loop())
        
        print(f"✅ Server ready! Connect to ws://localhost:{self.port}")
        
        # Keep server running
        await self.server.wait_closed()
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        print("🛑 Stopping server...")
        
        self.stop_streaming()
        
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        print("✅ Server stopped")
    
    def run(self):
        """Run the streaming server (blocking call)."""
        try:
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            print("\n🛑 Server interrupted by user")


def add_streaming_methods_to_system():
    """Add streaming methods to StarSystem class."""
    from ..core.systems import StarSystem
    
    def start_stream(self, port: int = 8765, fps: int = 60):
        """Start real-time WebSocket streaming."""
        streamer = WebSocketStreamer(self, target_fps=fps, port=port)
        return streamer
    
    # Add method to StarSystem class
    StarSystem.start_stream = start_stream