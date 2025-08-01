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
            else:
                print(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON message: {message}")
    
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
        
        print(f"🚀 Starting streaming loop at {self.target_fps} FPS")
        
        try:
            while self.is_streaming:
                loop_start = time.time()
                
                try:
                    # Get next frame data
                    frame_data = self.get_next_frame()
                    print(f"🎬 Generated frame {frame_data.get('frame', '?')} at time {frame_data.get('time', '?')}")
                    
                    # Broadcast to all clients
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