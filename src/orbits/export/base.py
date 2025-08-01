"""Base classes for data export functionality."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np


class BaseExporter(ABC):
    """Abstract base class for all exporters."""
    
    def __init__(self, star_system):
        self.star_system = star_system
    
    def get_object_metadata(self) -> List[Dict[str, Any]]:
        """Get static metadata for all objects (sent once)."""
        metadata = []
        for i, obj in enumerate(self.star_system.astro_objects):
            # Convert color to list for JSON serialization
            color = obj.color
            if isinstance(color, tuple):
                color = list(color)
            metadata.append({
                "id": i,
                "name": obj.name,
                "mass": float(obj.mass),
                "radius": float(obj.radius),
                "color": color
            })
        return metadata
    
    def get_units_info(self) -> Dict[str, str]:
        """Get unit information."""
        return {
            "length": "AU",
            "time": "days", 
            "mass": "M_sun"
        }
    
    def get_positions_3d(self) -> List[List[float]]:
        """Convert 2D positions to 3D (adding z=0) and return as list."""
        positions_2d = self.star_system.phase_space[:self.star_system.phase_space.size//2]
        positions_2d = positions_2d.reshape(len(self.star_system.astro_objects), 2)
        
        # Convert to 3D by adding z=0
        positions_3d = []
        for pos_2d in positions_2d:
            positions_3d.append([float(pos_2d[0]), float(pos_2d[1]), 0.0])
        
        return positions_3d
    
    def get_frame_data(self, frame_number: int, current_time: float) -> Dict[str, Any]:
        """Get current frame data."""
        return {
            "frame": frame_number,
            "time": float(current_time),
            "positions": self.get_positions_3d()
        }
    
    def export(self, *args, **kwargs):
        """Export data. Implementation depends on the specific exporter."""
        raise NotImplementedError("Subclass must implement export() method")


class TrajectoryExporter(BaseExporter):
    """Base class for trajectory-based exporters (static export)."""
    
    def simulate_trajectory(self, duration_days: float, time_step: Optional[float] = None) -> List[Dict[str, Any]]:
        """Simulate trajectory and collect frame data."""
        if time_step is None:
            time_step = self.star_system.step_size
        
        # Store original state
        original_phase_space = self.star_system.phase_space.copy()
        original_step_size = self.star_system.step_size
        
        # Set desired time step
        self.star_system.step_size = time_step
        
        trajectory = []
        current_time = 0.0
        frame_number = 0
        
        # Add initial frame
        trajectory.append(self.get_frame_data(frame_number, current_time))
        
        # Simulate trajectory
        while current_time < duration_days:
            self.star_system.evolve()
            current_time += time_step
            frame_number += 1
            
            trajectory.append(self.get_frame_data(frame_number, current_time))
        
        # Restore original state
        self.star_system.phase_space = original_phase_space
        self.star_system.step_size = original_step_size
        
        return trajectory


class StreamingExporter(BaseExporter):
    """Base class for real-time streaming exporters."""
    
    def __init__(self, star_system, target_fps: int = 60):
        super().__init__(star_system)
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps  # seconds per frame
        self.current_frame = 0
        self.current_time = 0.0
        self.is_streaming = False
    
    def start_streaming(self):
        """Start streaming mode."""
        self.is_streaming = True
        self.current_frame = 0
        self.current_time = 0.0
    
    def stop_streaming(self):
        """Stop streaming mode."""
        self.is_streaming = False
    
    def get_next_frame(self) -> Dict[str, Any]:
        """Get next frame data and advance simulation."""
        if not self.is_streaming:
            raise RuntimeError("Streaming not started. Call start_streaming() first.")
        
        # Get current frame data
        frame_data = self.get_frame_data(self.current_frame, self.current_time)
        
        # Advance simulation
        self.star_system.evolve()
        self.current_time += self.star_system.step_size
        self.current_frame += 1
        
        return frame_data