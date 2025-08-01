"""FastAPI server for serving both static data and WebSocket streaming."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .static import JSONExporter
from .streaming import WebSocketStreamer


class OrbitServer:
    """FastAPI server for orbital data visualization."""
    
    def __init__(self, star_system, host: str = "localhost", port: int = 8000):
        self.star_system = star_system
        self.host = host
        self.port = port
        self.app = FastAPI(title="Orbits Visualization Server", version="0.1.0")
        self.streamer: Optional[WebSocketStreamer] = None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Orbits Visualization Server",
                "endpoints": {
                    "metadata": "/api/metadata",
                    "trajectory": "/api/trajectory",
                    "websocket": "/ws"
                }
            }
        
        @self.app.get("/api/metadata")
        async def get_metadata():
            """Get system metadata."""
            exporter = JSONExporter(self.star_system)
            return {
                "objects": exporter.get_object_metadata(),
                "units": exporter.get_units_info(),
                "system_name": self.star_system.name
            }
        
        @self.app.get("/api/trajectory")
        async def get_trajectory(
            duration: float = 365.0,
            time_step: Optional[float] = None,
            max_frames: Optional[int] = None
        ):
            """Get complete trajectory data."""
            try:
                exporter = JSONExporter(self.star_system)
                
                if max_frames:
                    # Use compressed export
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        data = exporter.export_compressed(
                            tmp.name, 
                            duration, 
                            max_frames, 
                            time_step
                        )
                    Path(tmp.name).unlink()  # Clean up temp file
                else:
                    # Use regular export
                    data = {
                        "objects": exporter.get_object_metadata(),
                        "units": exporter.get_units_info(),
                        "trajectory": exporter.simulate_trajectory(duration, time_step)
                    }
                
                return data
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time streaming."""
            await websocket.accept()
            
            try:
                # Initialize streamer if not exists
                if not self.streamer:
                    self.streamer = WebSocketStreamer(self.star_system, target_fps=60)
                    # Start streaming loop if this is the first client
                    if not self.streamer.is_streaming:
                        self.streamer.start_streaming()
                        asyncio.create_task(self.streamer.streaming_loop())
                
                # Register client
                await self.streamer.register_client(websocket)
                
                # Handle client messages
                while True:
                    try:
                        message = await websocket.receive_text()
                        await self.streamer.handle_client_message(websocket, message)
                    except WebSocketDisconnect:
                        break
                        
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                if self.streamer:
                    await self.streamer.unregister_client(websocket)
                    # Stop streaming if no more clients
                    if not self.streamer.clients:
                        self.streamer.stop_streaming()
    
    async def start_async(self):
        """Start server asynchronously."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def start(self):
        """Start server (blocking call)."""
        print(f"🚀 Starting Orbits server on http://{self.host}:{self.port}")
        print(f"📡 WebSocket streaming on ws://{self.host}:{self.port}/ws")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


def add_server_methods_to_system():
    """Add server methods to StarSystem class."""
    from ..core.systems import StarSystem
    
    def serve(self, host: str = "localhost", port: int = 8000):
        """Start FastAPI server for visualization."""
        server = OrbitServer(self, host, port)
        return server
    
    def start_server(self, host: str = "localhost", port: int = 8000):
        """Start server (blocking call)."""
        server = self.serve(host, port)
        server.start()
    
    # Add methods to StarSystem class
    StarSystem.serve = serve
    StarSystem.start_server = start_server