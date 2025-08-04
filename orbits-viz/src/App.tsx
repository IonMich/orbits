import { useState, useCallback, useEffect } from 'react';
import { SidebarProvider, SidebarTrigger } from './components/ui/sidebar';
import { AppSidebar } from './components/AppSidebar';
import { useWebSocket } from './hooks/useWebSocket';
import * as OrbitTypes from './types/orbit';
import Scene from './components/OrbitViewer/Scene';

type TrajectoryData = OrbitTypes.TrajectoryData;
type SimulationControls = OrbitTypes.SimulationControls;

function App() {
  const [mode, setMode] = useState<'streaming' | 'file'>('streaming');
  const [simulationType, setSimulationType] = useState<'earth-sun' | 'solar-system' | 'random'>('solar-system');
  const [showTrails, setShowTrails] = useState(true);
  const [scale, setScale] = useState(1);
  const [trails, setTrails] = useState<Map<number, [number, number, number][]>>(new Map());
  
  // Backend connection state
  const [backendError, setBackendError] = useState<string | null>(null);
  const [isLoadingSimulation, setIsLoadingSimulation] = useState(false);
  const [isChangingSimulation, setIsChangingSimulation] = useState(false);
  
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
    disconnect,
    requestSimulation,
    sendMessage
  } = useWebSocket({ 
    url: 'ws://localhost:8000/ws', 
    autoConnect: mode === 'streaming',
    simulationType: simulationType
  });


  // Use streaming objects as data source
  const objects = streamingObjects || [];

  // Use streaming frame as current frame
  const currentFrame = streamingFrame;

  // Handle mode changes
  const handleModeChange = useCallback(async (newMode: 'streaming' | 'file') => {
    setMode(newMode);
    setTrails(new Map()); // Clear trails when switching modes
    
    // Clear backend error when switching modes
    setBackendError(null);
  }, []);

  // Handle simulation type changes
  const handleSimulationTypeChange = useCallback(async (newType: 'earth-sun' | 'solar-system' | 'random') => {
    setSimulationType(newType);
    
    // Show loading state (especially important for solar-system with Horizons API)
    setIsChangingSimulation(true);
    
    // Clear trails immediately when starting to switch
    setTrails(new Map());
    
    // Request new simulation type via WebSocket if connected
    if (isConnected) {
      requestSimulation(newType);
    }
  }, [isConnected, requestSimulation]);


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

  // Stop loading when new objects arrive (simulation changed successfully)
  useEffect(() => {
    if (streamingObjects && streamingObjects.length > 0 && isChangingSimulation) {
      setIsChangingSimulation(false);
      // Clear trails again to ensure clean start with new simulation
      setTrails(new Map());
    }
  }, [streamingObjects, isChangingSimulation]);

  // Update trails when frame changes AND simulation is playing
  useEffect(() => {
    if (currentFrame && objects.length > 0 && controls.isPlaying) {
      setTrails(prevTrails => {
        const newTrails = new Map(prevTrails);
        
        objects.forEach((object, index) => {
          const position = currentFrame.positions[index];
          if (position) {
            const existingTrail = newTrails.get(object.id) || [];
            const updatedTrail = [...existingTrail, position];
            
            // No trail length limit - let trails grow indefinitely
            newTrails.set(object.id, updatedTrail);
          }
        });
        
        return newTrails;
      });
    }
  }, [currentFrame, objects, controls.isPlaying]);

  // Don't auto-start playback - let user control when to play

  // Control handlers
  const handlePlayPause = useCallback(() => {
    const newIsPlaying = !controls.isPlaying;
    setControls(prev => ({ ...prev, isPlaying: newIsPlaying }));
    
    // Send play/pause command to WebSocket
    if (isConnected) {
      sendMessage({ type: newIsPlaying ? 'play' : 'pause' });
    }
  }, [controls.isPlaying, isConnected, sendMessage]);

  const handleReset = useCallback(() => {
    // Clear trails immediately
    setTrails(new Map());
    
    // Send reset command to WebSocket (backend will auto-pause)
    if (isConnected) {
      sendMessage({ type: 'reset' });
    }
    
    // Update frontend state to match backend (paused, reset time/frame)
    setControls(prev => ({
      ...prev,
      isPlaying: false,
      currentFrame: 0,
      currentTime: 0
    }));
  }, [isConnected, sendMessage]);

  const handleFrameChange = useCallback((newFrame: number) => {
    // Frame changes are handled by streaming data
    setControls(prev => ({
      ...prev,
      currentFrame: newFrame,
      isPlaying: false
    }));
  }, []);

  const handleSpeedChange = useCallback((newSpeed: number) => {
    setControls(prev => ({ ...prev, speed: newSpeed }));
    
    // Send speed change command to WebSocket
    if (isConnected) {
      sendMessage({ type: 'set_speed', speed: newSpeed });
    }
  }, [isConnected, sendMessage]);

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <AppSidebar
          mode={mode}
          onModeChange={handleModeChange}
          simulationType={simulationType}
          onSimulationTypeChange={handleSimulationTypeChange}
          isChangingSimulation={isChangingSimulation}
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
            <Scene
              objects={objects}
              currentFrame={currentFrame}
              trails={trails}
              showTrails={showTrails}
              scale={scale}
            />
          </div>
        </main>
      </div>
    </SidebarProvider>
  );
}

export default App;
