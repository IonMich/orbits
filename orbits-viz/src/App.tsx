import { useState, useCallback, useEffect } from 'react';
import { SidebarProvider, SidebarTrigger } from './components/ui/sidebar';
import { AppSidebar } from './components/AppSidebar';
import { useWebSocket } from './hooks/useWebSocket';
import * as OrbitTypes from './types/orbit';
import Scene from './components/OrbitViewer/Scene';

type TrajectoryData = OrbitTypes.TrajectoryData;
type SimulationControls = OrbitTypes.SimulationControls;
type FrameData = OrbitTypes.FrameData;
type AstroObject = OrbitTypes.AstroObject;

function App() {
  const [mode, setMode] = useState<'demo' | 'streaming' | 'file'>('demo');
  const [demoData, setDemoData] = useState<TrajectoryData | null>(null);
  const [showTrails, setShowTrails] = useState(true);
  const [scale, setScale] = useState(1);
  const [trails, setTrails] = useState<Map<number, [number, number, number][]>>(new Map());
  
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
    units: streamingUnits,
    latestFrame: streamingFrame,
    error: streamingError,
    connect,
    disconnect
  } = useWebSocket({ 
    url: 'ws://localhost:8000/ws', 
    autoConnect: mode === 'streaming' 
  });

  // Create demo data
  const createDemoData = (): TrajectoryData => {
    const objects = [
      {
        id: 0,
        name: 'Sun',
        mass: 1.0,
        radius: 0.1,
        color: '#FFFF00'
      },
      {
        id: 1,
        name: 'Earth',
        mass: 3e-6,
        radius: 0.05,
        color: '#0000FF'
      }
    ];

    const trajectory = [];
    for (let frame = 0; frame < 365; frame++) {
      const time = frame;
      const angle = (frame / 365) * 2 * Math.PI;
      trajectory.push({
        frame,
        time,
        positions: [
          [0, 0, 0] as [number, number, number], // Sun at center
          [Math.cos(angle) * 3, 0, Math.sin(angle) * 3] as [number, number, number] // Earth orbit
        ]
      });
    }

    return {
      objects,
      units: { length: 'AU', time: 'days', mass: 'M_sun' },
      trajectory
    };
  };

  // Determine data source based on mode
  const objects = (() => {
    if (mode === 'streaming' && streamingObjects) {
      return streamingObjects;
    }
    return demoData?.objects || [];
  })();

  const currentFrame = (() => {
    if (mode === 'streaming' && streamingFrame) {
      return streamingFrame;
    }
    if (demoData?.trajectory && controls.currentFrame < demoData.trajectory.length) {
      return demoData.trajectory[controls.currentFrame];
    }
    return null;
  })();

  // Handle mode changes
  const handleModeChange = useCallback((newMode: 'demo' | 'streaming' | 'file') => {
    if (newMode === 'demo' && !demoData) {
      setDemoData(createDemoData());
    }
    setMode(newMode);
    setTrails(new Map()); // Clear trails when switching modes
  }, [demoData]);

  // Initialize demo data on first load
  useEffect(() => {
    if (mode === 'demo' && !demoData) {
      setDemoData(createDemoData());
    }
  }, [mode, demoData]);

  // Initialize total frames for static mode
  useEffect(() => {
    if (mode === 'demo' && demoData?.trajectory) {
      setControls(prev => ({
        ...prev,
        totalFrames: demoData.trajectory.length,
        currentTime: demoData.trajectory[0]?.time || 0
      }));
    }
  }, [mode, demoData?.trajectory]);

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

  // Animation loop for static mode
  useEffect(() => {
    if (mode !== 'demo' || !controls.isPlaying || !demoData?.trajectory) {
      return;
    }

    const interval = setInterval(() => {
      setControls(prev => {
        const nextFrame = prev.currentFrame + prev.speed;
        if (nextFrame >= prev.totalFrames) {
          return { ...prev, isPlaying: false, currentFrame: prev.totalFrames - 1 };
        }
        
        const frameData = demoData.trajectory[Math.floor(nextFrame)];
        return {
          ...prev,
          currentFrame: Math.floor(nextFrame),
          currentTime: frameData?.time || prev.currentTime
        };
      });
    }, 50);

    return () => clearInterval(interval);
  }, [controls.isPlaying, controls.speed, mode, demoData?.trajectory]);

  // Control handlers
  const handlePlayPause = useCallback(() => {
    setControls(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
  }, []);

  const handleReset = useCallback(() => {
    setControls(prev => ({
      ...prev,
      isPlaying: false,
      currentFrame: 0,
      currentTime: demoData?.trajectory?.[0]?.time || 0
    }));
    setTrails(new Map());
  }, [demoData?.trajectory]);

  const handleFrameChange = useCallback((newFrame: number) => {
    if (mode === 'demo') {
      const frameData = demoData?.trajectory?.[newFrame];
      setControls(prev => ({
        ...prev,
        currentFrame: newFrame,
        currentTime: frameData?.time || prev.currentTime,
        isPlaying: false
      }));
    }
  }, [mode, demoData?.trajectory]);

  const handleSpeedChange = useCallback((newSpeed: number) => {
    setControls(prev => ({ ...prev, speed: newSpeed }));
  }, []);

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <AppSidebar
          mode={mode}
          onModeChange={handleModeChange}
          controls={controls}
          onControlsChange={setControls}
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
                <div className="text-center space-y-4">
                  <div className="text-6xl mb-4">🌌</div>
                  <h2 className="text-2xl font-bold text-white">Welcome to Orbits</h2>
                  <p className="text-slate-300 max-w-md">
                    Select a simulation mode from the sidebar to begin exploring orbital mechanics in 3D
                  </p>
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
