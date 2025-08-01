"""Test export functionality."""

import pytest
import json
import tempfile
from pathlib import Path
import asyncio

from orbits import StarSystem, M_sun, M_jupiter
from orbits.export.static import JSONExporter
from orbits.export.streaming import WebSocketStreamer


class TestStaticExport:
    """Test static export functionality."""
    
    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        return StarSystem.star_and_planet(
            star_mass=M_sun,
            planet_mass=M_jupiter,
            planet_period=365.25,
            step_size=1.0
        )
    
    def test_json_exporter_creation(self, simple_system):
        """Test JSON exporter creation."""
        exporter = JSONExporter(simple_system)
        assert exporter.star_system == simple_system
    
    def test_metadata_extraction(self, simple_system):
        """Test metadata extraction."""
        exporter = JSONExporter(simple_system)
        
        metadata = exporter.get_object_metadata()
        assert len(metadata) == 2
        assert metadata[0]["name"] == "Star"
        assert metadata[1]["name"] == "Planet"
        assert all("mass" in obj for obj in metadata)
        assert all("radius" in obj for obj in metadata)
        assert all("color" in obj for obj in metadata)
        
        units = exporter.get_units_info()
        assert "length" in units
        assert "time" in units
        assert "mass" in units
    
    def test_3d_position_conversion(self, simple_system):
        """Test 2D to 3D position conversion."""
        exporter = JSONExporter(simple_system)
        positions_3d = exporter.get_positions_3d()
        
        assert len(positions_3d) == 2  # 2 objects
        for pos in positions_3d:
            assert len(pos) == 3  # x, y, z
            assert pos[2] == 0.0  # z should be 0
    
    def test_trajectory_simulation(self, simple_system):
        """Test trajectory simulation."""
        exporter = JSONExporter(simple_system)
        trajectory = exporter.simulate_trajectory(duration_days=10.0, time_step=1.0)
        
        assert len(trajectory) == 11  # 0 to 10 days inclusive
        
        # Check frame structure
        frame = trajectory[0]
        assert "frame" in frame
        assert "time" in frame
        assert "positions" in frame
        assert len(frame["positions"]) == 2  # 2 objects
        assert len(frame["positions"][0]) == 3  # 3D coordinates
    
    def test_json_export_to_file(self, simple_system):
        """Test JSON export to file."""
        exporter = JSONExporter(simple_system)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            data = exporter.export(tmp.name, duration_days=5.0, time_step=1.0)
            
            # Check returned data structure
            assert "trajectory" in data
            assert "simulation" in data
            assert "objects" in data
            assert "units" in data
            
            # Check file was created and contains valid JSON
            assert Path(tmp.name).exists()
            with open(tmp.name, 'r') as f:
                file_data = json.load(f)
                assert file_data == data
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_compressed_export(self, simple_system):
        """Test compressed export for large datasets."""
        exporter = JSONExporter(simple_system)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            data = exporter.export_compressed(
                tmp.name, 
                duration_days=100.0, 
                max_frames=50, 
                time_step=1.0
            )
            
            assert len(data["trajectory"]) == 50  # Compressed to max_frames
            assert data["simulation"]["original_frames"] == 101  # Original frame count
            assert data["simulation"]["compression_ratio"] > 1
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_system_export_methods(self, simple_system):
        """Test export methods added to StarSystem."""
        # Test that methods exist
        assert hasattr(simple_system, 'export_json')
        assert hasattr(simple_system, 'export_json_compressed')
        
        # Test method usage
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            data = simple_system.export_json(tmp.name, duration_days=5.0)
            assert "trajectory" in data
            Path(tmp.name).unlink()


class TestStreamingExport:
    """Test streaming export functionality."""
    
    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        return StarSystem.star_and_planet(
            star_mass=M_sun,
            planet_mass=M_jupiter,
            planet_period=365.25,
            step_size=1.0
        )
    
    def test_websocket_streamer_creation(self, simple_system):
        """Test WebSocket streamer creation."""
        streamer = WebSocketStreamer(simple_system, target_fps=30)
        assert streamer.star_system == simple_system
        assert streamer.target_fps == 30
        assert not streamer.is_streaming
        assert len(streamer.clients) == 0
    
    def test_streaming_state_management(self, simple_system):
        """Test streaming state management."""
        streamer = WebSocketStreamer(simple_system)
        
        # Initially not streaming
        assert not streamer.is_streaming
        
        # Start streaming
        streamer.start_streaming()
        assert streamer.is_streaming
        assert streamer.current_frame == 0
        assert streamer.current_time == 0.0
        
        # Stop streaming
        streamer.stop_streaming()
        assert not streamer.is_streaming
    
    def test_frame_generation(self, simple_system):
        """Test frame data generation."""
        streamer = WebSocketStreamer(simple_system)
        streamer.start_streaming()
        
        # Get first frame
        frame1 = streamer.get_next_frame()
        assert frame1["frame"] == 0
        assert frame1["time"] == 0.0
        assert len(frame1["positions"]) == 2
        
        # Get second frame - should advance simulation
        frame2 = streamer.get_next_frame()
        assert frame2["frame"] == 1
        assert frame2["time"] == simple_system.step_size
        
        # Positions should be different (system evolved)
        assert frame1["positions"] != frame2["positions"]
    
    def test_system_streaming_methods(self, simple_system):
        """Test streaming methods added to StarSystem."""
        # Test that method exists
        assert hasattr(simple_system, 'start_stream')
        
        # Test method usage
        streamer = simple_system.start_stream(port=8766, fps=30)
        assert isinstance(streamer, WebSocketStreamer)
        assert streamer.target_fps == 30
        assert streamer.port == 8766


class TestServerIntegration:
    """Test server integration."""
    
    @pytest.fixture
    def simple_system(self):
        """Create a simple test system."""
        return StarSystem.star_and_planet(
            star_mass=M_sun,
            planet_mass=M_jupiter,
            planet_period=365.25,
            step_size=1.0
        )
    
    def test_server_methods_exist(self, simple_system):
        """Test that server methods are added to StarSystem."""
        assert hasattr(simple_system, 'serve')
        assert hasattr(simple_system, 'start_server')
    
    def test_server_creation(self, simple_system):
        """Test server creation."""
        from orbits.export.server import OrbitServer
        
        server = simple_system.serve(host="localhost", port=8001)
        assert isinstance(server, OrbitServer)
        assert server.host == "localhost"
        assert server.port == 8001
        assert server.star_system == simple_system