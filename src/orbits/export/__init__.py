"""Export functionality for orbits package."""

from .static import JSONExporter, add_export_methods_to_system
from .streaming import WebSocketStreamer, add_streaming_methods_to_system  
from .server import OrbitServer, add_server_methods_to_system

# Add export methods to StarSystem when module is imported
add_export_methods_to_system()
add_streaming_methods_to_system()
add_server_methods_to_system()

__all__ = [
    'JSONExporter',
    'WebSocketStreamer', 
    'OrbitServer',
]