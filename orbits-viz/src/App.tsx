import { useState, useCallback, useEffect } from 'react';
import { SidebarProvider, SidebarTrigger } from './components/ui/sidebar';
import { AppSidebar } from './components/AppSidebar';
import { useWebSocket } from './hooks/useWebSocket';
import * as OrbitTypes from './types/orbit';
import Scene from './components/OrbitViewer/Scene';

type TrajectoryData = OrbitTypes.TrajectoryData;
type SimulationControls = OrbitTypes.SimulationControls;

function App() {
  const [mode, setMode] = useState<'non-streaming' | 'streaming' | 'file'>('non-streaming');
  const [simulationType, setSimulationType] = useState<'earth-sun' | 'solar-system' | 'random'>('earth-sun');
  const [precomputedData, setPrecomputedData] = useState<TrajectoryData | null>(null);
  const [showTrails, setShowTrails] = useState(true);
  const [scale, setScale] = useState(1);
  const [trails, setTrails] = useState<Map<number, [number, number, number][]>>(new Map());
  
  // Backend connection state
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isLoadingSimulation, setIsLoadingSimulation] = useState(false);
  
  // Simulation state
  const [controls, setControls] = useState<SimulationControls>({
    isPlaying: false,
    speed: 1,
    currentFrame: 0,
    totalFrames: 0,
    currentTime: 0
  });

  // WebSocket connection for streaming mode
  const {
    isConnected,
    isConnecting,
    objects: streamingObjects,
    latestFrame: streamingFrame,
    error: streamingError,
    connect,
    disconnect
  } = useWebSocket({ 
    url: 'ws://localhost:8000/ws', 
    autoConnect: mode === 'streaming' 
  });

  // Fetch pre-computed simulation data from Python backend
  const fetchSimulationData = async (type: 'earth-sun' | 'solar-system' | 'random'): Promise<TrajectoryData> => {
    const apiEndpoints = {
      'earth-sun': 'http://localhost:8000/api/simulation/earth-sun',
      'solar-system': 'http://localhost:8000/api/simulation/solar-system', 
      'random': 'http://localhost:8000/api/simulation/random'
    };

    setIsLoadingSimulation(true);
    setBackendError(null);

    try {
      const response = await fetch(apiEndpoints[type]);
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setBackendError(null); // Clear any previous errors
      return data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setBackendError(`Backend unavailable: ${errorMessage}`);
      throw error;
    } finally {
      setIsLoadingSimulation(false);
    }
  };

  // Determine data source based on mode
  const objects = (() => {
    if (mode === 'streaming' && streamingObjects) {
      return streamingObjects;
    }
    return precomputedData?.objects || [];
  })();

  const currentFrame = (() => {
    if (mode === 'streaming' && streamingFrame) {
      return streamingFrame;
    }
    if (precomputedData?.trajectory && controls.currentFrame < precomputedData.trajectory.length) {
      return precomputedData.trajectory[controls.currentFrame];
    }
    return null;
  })();

  // Handle mode changes
  const handleModeChange = useCallback(async (newMode: 'non-streaming' | 'streaming' | 'file') => {
    setMode(newMode);
    setTrails(new Map()); // Clear trails when switching modes
    
    if (newMode === 'non-streaming') {
      try {
        console.log(`Fetching ${simulationType} simulation data from backend...`);
        const data = await fetchSimulationData(simulationType);
        console.log(`✓ Loaded ${simulationType} simulation from backend`);
        setPrecomputedData(data);
      } catch (error) {
        console.error(`❌ Failed to load ${simulationType} simulation:`, error);
        setPrecomputedData(null);
      }
    } else {
      // Clear backend error when switching to streaming mode
      setBackendError(null);
    }
  }, [simulationType]);

  // Handle simulation type changes
  const handleSimulationTypeChange = useCallback(async (newType: 'earth-sun' | 'solar-system' | 'random') => {
    setSimulationType(newType);
    
    if (mode === 'non-streaming') {
      try {
        console.log(`Fetching ${newType} simulation data from backend...`);
        const data = await fetchSimulationData(newType);
        console.log(`✓ Loaded ${newType} simulation from backend`);
        setPrecomputedData(data);
        
        setTrails(new Map()); // Clear trails when switching simulation types
        setControls(prev => ({
          ...prev,
          isPlaying: false,
          currentFrame: 0,
          currentTime: 0
        }));
      } catch (error) {
        console.error(`❌ Failed to load ${newType} simulation:`, error);
        setPrecomputedData(null);
      }
    }
  }, [mode]);

  // Initialize precomputed data on first load
  useEffect(() => {
    if (mode === 'non-streaming' && !precomputedData) {
      handleSimulationTypeChange(simulationType);
    }
  }, [mode, precomputedData, simulationType, handleSimulationTypeChange]);

  // Initialize total frames for non-streaming mode
  useEffect(() => {
    if (mode === 'non-streaming' && precomputedData?.trajectory) {
      setControls(prev => ({
        ...prev,
        totalFrames: precomputedData.trajectory.length,
        currentTime: precomputedData.trajectory[0]?.time || 0
      }));
    }
  }, [mode, precomputedData?.trajectory]);

  // Handle streaming frame updates
  useEffect(() => {
    if (mode === 'streaming' && streamingFrame) {
      setControls(prev => ({
        ...prev,
        currentFrame: streamingFrame.frame,
        currentTime: streamingFrame.time
      }));
    }
  }, [mode, streamingFrame]);

  // Update trails when frame changes
  useEffect(() => {
    if (currentFrame && objects.length > 0) {
      setTrails(prevTrails => {
        const newTrails = new Map(prevTrails);
        
        objects.forEach((object, index) => {
          const position = currentFrame.positions[index];
          if (position) {
            const existingTrail = newTrails.get(object.id) || [];
            const updatedTrail = [...existingTrail, position];
            
            // Limit trail length for performance
            const maxTrailLength = 1000;
            if (updatedTrail.length > maxTrailLength) {
              updatedTrail.splice(0, updatedTrail.length - maxTrailLength);
            }
            
            newTrails.set(object.id, updatedTrail);
          }
        });
        
        return newTrails;
      });
    }
  }, [currentFrame, objects]);

  // Animation loop for non-streaming mode
  useEffect(() => {
    if (mode !== 'non-streaming' || !controls.isPlaying || !precomputedData?.trajectory) {
      return;
    }

    const interval = setInterval(() => {
      setControls(prev => {
        const nextFrame = prev.currentFrame + prev.speed;
        if (nextFrame >= prev.totalFrames) {
          return { ...prev, isPlaying: false, currentFrame: prev.totalFrames - 1 };
        }
        
        const frameData = precomputedData.trajectory[Math.floor(nextFrame)];
        return {
          ...prev,
          currentFrame: Math.floor(nextFrame),
          currentTime: frameData?.time || prev.currentTime
        };
      });
    }, 50);

    return () => clearInterval(interval);
  }, [controls.isPlaying, controls.speed, mode, precomputedData?.trajectory]);

  // Control handlers
  const handlePlayPause = useCallback(() => {
    setControls(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
  }, []);

  const handleReset = useCallback(() => {
    setControls(prev => ({
      ...prev,
      isPlaying: false,
      currentFrame: 0,
      currentTime: precomputedData?.trajectory?.[0]?.time || 0
    }));
    setTrails(new Map());
  }, [precomputedData?.trajectory]);

  const handleFrameChange = useCallback((newFrame: number) => {
    if (mode === 'non-streaming') {
      const frameData = precomputedData?.trajectory?.[newFrame];
      setControls(prev => ({
        ...prev,
        currentFrame: newFrame,
        currentTime: frameData?.time || prev.currentTime,
        isPlaying: false
      }));
    }
  }, [mode, precomputedData?.trajectory]);

  const handleSpeedChange = useCallback((newSpeed: number) => {
    setControls(prev => ({ ...prev, speed: newSpeed }));
  }, []);

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <AppSidebar
          mode={mode}
          onModeChange={handleModeChange}
          simulationType={simulationType}
          onSimulationTypeChange={handleSimulationTypeChange}
          controls={controls}
          onControlsChange={(partial) => setControls(prev => ({ ...prev, ...partial }))}
          showTrails={showTrails}
          onShowTrailsChange={setShowTrails}
          scale={scale}
          onScaleChange={setScale}
          onPlayPause={handlePlayPause}
          onReset={handleReset}
          onFrameChange={handleFrameChange}
          onSpeedChange={handleSpeedChange}
          isConnected={isConnected}
          isConnecting={isConnecting}
          streamingError={streamingError}
          onConnect={connect}
          onDisconnect={disconnect}
          backendError={backendError}
          isLoadingSimulation={isLoadingSimulation}
        />
        
        <main className="flex-1 relative">
          {/* Sidebar toggle button */}
          <div className="absolute top-4 left-4 z-10">
            <SidebarTrigger className="bg-background/80 backdrop-blur-sm" />
          </div>
          
          {/* Full-screen Three.js visualization */}
          <div className="w-full h-full">
            {objects.length > 0 && currentFrame ? (
              <Scene
                objects={objects}
                currentFrame={currentFrame}
                trails={trails}
                showTrails={showTrails}
                scale={scale}
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
                <div className="text-center space-y-4 max-w-lg mx-auto px-4">
                  {/* Loading State */}
                  {isLoadingSimulation && (
                    <>
                      <div className="text-6xl mb-4">⏳</div>
                      <h2 className="text-2xl font-bold text-white">Loading Simulation</h2>
                      <p className="text-slate-300">
                        Computing orbital mechanics in Python backend...
                      </p>
                      <div className="flex justify-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                      </div>
                    </>
                  )}
                  
                  {/* Backend Error State */}
                  {!isLoadingSimulation && backendError && mode === 'non-streaming' && (
                    <>
                      <div className="text-6xl mb-4">🔌</div>
                      <h2 className="text-2xl font-bold text-red-400">Backend Unavailable</h2>
                      <p className="text-slate-300">
                        The Python simulation server is not running. Both Pre-computed and Streaming modes require the backend server.
                      </p>
                      <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-left">
                        <p className="text-red-300 text-sm font-mono">
                          {backendError}
                        </p>
                      </div>
                      <div className="text-slate-400 text-sm space-y-2">
                        <p>To start the backend server, run:</p>
                        <code className="bg-slate-800 px-3 py-1 rounded text-green-400 text-xs">
                          uv run python solar_system_server.py
                        </code>
                        <p>The same server handles both Pre-computed and Streaming modes.</p>
                      </div>
                      <button
                        onClick={() => handleSimulationTypeChange(simulationType)}
                        className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors"
                      >
                        🔄 Retry Connection
                      </button>
                    </>
                  )}
                  
                  {/* Welcome State */}
                  {!isLoadingSimulation && !backendError && (
                    <>
                      <div className="text-6xl mb-4">🌌</div>
                      <h2 className="text-2xl font-bold text-white">Welcome to Orbits</h2>
                      <p className="text-slate-300">
                        Select a simulation mode from the sidebar: choose Pre-computed for complete simulations or Streaming for real-time physics
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </SidebarProvider>
  );
}

export default App;
