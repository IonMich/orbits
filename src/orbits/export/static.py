"""Static trajectory export functionality."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .base import TrajectoryExporter


class JSONExporter(TrajectoryExporter):
    """Export complete trajectories to JSON format."""
    
    def export(self, 
               filepath: str, 
               duration_days: float, 
               time_step: Optional[float] = None,
               include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export trajectory to JSON file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        duration_days : float
            Duration to simulate in days
        time_step : float, optional
            Time step for simulation. If None, uses system default
        include_metadata : bool
            Whether to include object metadata and units
            
        Returns
        -------
        dict
            The exported data structure
        """
        # Simulate trajectory
        trajectory = self.simulate_trajectory(duration_days, time_step)
        
        # Build export data structure
        export_data = {
            "trajectory": trajectory,
            "simulation": {
                "duration_days": duration_days,
                "time_step": time_step or self.star_system.step_size,
                "num_frames": len(trajectory),
                "integrator": getattr(self.star_system, 'evolve', None).__name__ if hasattr(self.star_system, 'evolve') else "unknown"
            }
        }
        
        if include_metadata:
            export_data["objects"] = self.get_object_metadata()
            export_data["units"] = self.get_units_info()
        
        # Write to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Exported {len(trajectory)} frames to {filepath}")
        return export_data
    
    def export_compressed(self, 
                         filepath: str, 
                         duration_days: float,
                         max_frames: int = 10000,
                         time_step: Optional[float] = None) -> Dict[str, Any]:
        """
        Export trajectory with frame compression for large datasets.
        
        Parameters
        ----------
        filepath : str
            Output file path
        duration_days : float
            Duration to simulate in days
        max_frames : int
            Maximum number of frames to export (will skip frames if needed)
        time_step : float, optional
            Time step for simulation
            
        Returns
        -------
        dict
            The exported data structure
        """
        # Simulate full trajectory
        full_trajectory = self.simulate_trajectory(duration_days, time_step)
        
        # Compress by selecting frames
        if len(full_trajectory) <= max_frames:
            compressed_trajectory = full_trajectory
        else:
            # Select evenly spaced frames
            step = len(full_trajectory) / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            compressed_trajectory = [full_trajectory[i] for i in indices]
        
        # Build export data
        export_data = {
            "trajectory": compressed_trajectory,
            "simulation": {
                "duration_days": duration_days,
                "time_step": time_step or self.star_system.step_size,
                "num_frames": len(compressed_trajectory),
                "original_frames": len(full_trajectory),
                "compression_ratio": len(full_trajectory) / len(compressed_trajectory)
            },
            "objects": self.get_object_metadata(),
            "units": self.get_units_info()
        }
        
        # Write to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Exported {len(compressed_trajectory)} frames (compressed from {len(full_trajectory)}) to {filepath}")
        return export_data


def add_export_methods_to_system():
    """Add export methods to StarSystem class."""
    from ..core.systems import StarSystem
    
    def export_json(self, filepath: str, duration_days: float, time_step: Optional[float] = None, **kwargs):
        """Export trajectory to JSON file."""
        exporter = JSONExporter(self)
        return exporter.export(filepath, duration_days, time_step, **kwargs)
    
    def export_json_compressed(self, filepath: str, duration_days: float, max_frames: int = 10000, **kwargs):
        """Export compressed trajectory to JSON file."""
        exporter = JSONExporter(self)
        return exporter.export_compressed(filepath, duration_days, max_frames, **kwargs)
    
    # Add methods to StarSystem class
    StarSystem.export_json = export_json
    StarSystem.export_json_compressed = export_json_compressed