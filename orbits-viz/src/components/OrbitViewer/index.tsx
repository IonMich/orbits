import { useState, useEffect, useCallback, useMemo } from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Slider } from '../ui/slider';
import { Switch } from '../ui/switch';
import { Badge } from '../ui/badge';
import { useWebSocket } from '../../hooks/useWebSocket';
import * as OrbitTypes from '../../types/orbit';

type TrajectoryData = OrbitTypes.TrajectoryData;
type FrameData = OrbitTypes.FrameData;
type AstroObject = OrbitTypes.AstroObject;
type SimulationControls = OrbitTypes.SimulationControls;
import Scene from './Scene';
import { Play, Pause, RotateCcw, Wifi, WifiOff } from 'lucide-react';

interface OrbitViewerProps {
  data?: TrajectoryData;
  streamingUrl?: string;
  mode?: 'static' | 'streaming' | 'hybrid';
}

export default function OrbitViewer({ 
  data, 
  streamingUrl = 'ws://localhost:8000/ws',
  mode = 'static' 
}: OrbitViewerProps) {
  // Simulation state
  const [controls, setControls] = useState<SimulationControls>({
    isPlaying: false,
    speed: 1,
    currentFrame: 0,
    totalFrames: 0,
    currentTime: 0
  });
  
  // Display options
  const [showTrails, setShowTrails] = useState(true);
  const [scale, setScale] = useState(1);
  
  // Trail storage for orbital paths
  const [trails, setTrails] = useState<Map<number, [number, number, number][]>>(new Map());
  
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
    url: streamingUrl, 
    autoConnect: mode === 'streaming' 
  });

  // Determine data source based on mode
  const objects = useMemo(() => {
    if (mode === 'streaming' && streamingObjects) {
      return streamingObjects;
    }
    return data?.objects || [];
  }, [mode, streamingObjects, data?.objects]);

  const currentFrame = useMemo(() => {
    if (mode === 'streaming' && streamingFrame) {
      return streamingFrame;
    }
    if (data?.trajectory && controls.currentFrame < data.trajectory.length) {
      return data.trajectory[controls.currentFrame];
    }
    return null;
  }, [mode, streamingFrame, data?.trajectory, controls.currentFrame]);

  // Initialize total frames for static mode
  useEffect(() => {
    if (mode === 'static' && data?.trajectory) {
      setControls(prev => ({
        ...prev,
        totalFrames: data.trajectory.length,
        currentTime: data.trajectory[0]?.time || 0
      }));
    }
  }, [mode, data?.trajectory]);

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
    if (mode !== 'static' || !controls.isPlaying || !data?.trajectory) {
      return;
    }

    const interval = setInterval(() => {
      setControls(prev => {
        const nextFrame = prev.currentFrame + prev.speed;
        if (nextFrame >= prev.totalFrames) {
          return { ...prev, isPlaying: false, currentFrame: prev.totalFrames - 1 };
        }
        
        const frameData = data.trajectory[Math.floor(nextFrame)];
        return {
          ...prev,
          currentFrame: Math.floor(nextFrame),
          currentTime: frameData?.time || prev.currentTime
        };
      });
    }, 50); // ~20 FPS

    return () => clearInterval(interval);
  }, [controls.isPlaying, controls.speed, mode, data?.trajectory]);

  // Control handlers
  const handlePlayPause = useCallback(() => {
    if (mode === 'streaming') {
      // For streaming, this would toggle the server's streaming state
      // For now, just toggle local state
      setControls(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
    } else {
      setControls(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
    }
  }, [mode]);

  const handleReset = useCallback(() => {
    setControls(prev => ({
      ...prev,
      isPlaying: false,
      currentFrame: 0,
      currentTime: data?.trajectory?.[0]?.time || 0
    }));
    setTrails(new Map());
  }, [data?.trajectory]);

  const handleFrameChange = useCallback((newFrame: number) => {
    if (mode === 'static') {
      const frameData = data?.trajectory?.[newFrame];
      setControls(prev => ({
        ...prev,
        currentFrame: newFrame,
        currentTime: frameData?.time || prev.currentTime,
        isPlaying: false
      }));
    }
  }, [mode, data?.trajectory]);

  const handleSpeedChange = useCallback((newSpeed: number) => {
    setControls(prev => ({ ...prev, speed: newSpeed }));
  }, []);

  if (objects.length === 0) {
    return (
      <Card className="w-full h-96 flex items-center justify-center">
        <div className="text-center">
          <p className="text-muted-foreground mb-4">
            {mode === 'streaming' ? 'Connecting to simulation...' : 'No orbit data loaded'}
          </p>
          {mode === 'streaming' && (
            <Button onClick={connect} disabled={isConnecting}>
              {isConnecting ? 'Connecting...' : 'Connect'}
            </Button>
          )}
        </div>
      </Card>
    );
  }

  return (
    <div className="w-full space-y-4">
      {/* 3D Scene */}
      <Card className="overflow-hidden">
        <div className="h-96 relative">
          <Scene
            objects={objects}
            currentFrame={currentFrame}
            trails={trails}
            showTrails={showTrails}
            scale={scale}
          />
          
          {/* Connection status for streaming mode */}
          {mode === 'streaming' && (
            <div className="absolute top-4 right-4">
              <Badge variant={isConnected ? 'default' : 'destructive'} className="flex items-center gap-1">
                {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                {isConnected ? 'Connected' : 'Disconnected'}
              </Badge>
            </div>
          )}
          
          {/* Current time display */}
          <div className="absolute top-4 left-4">
            <Badge variant="outline">
              Time: {controls.currentTime.toFixed(1)} days
            </Badge>
          </div>
        </div>
      </Card>

      {/* Controls Panel */}
      <Card className="p-4">
        <div className="space-y-4">
          {/* Playback Controls */}
          <div className="flex items-center gap-4">
            <Button
              onClick={handlePlayPause}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              {controls.isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {controls.isPlaying ? 'Pause' : 'Play'}
            </Button>
            
            <Button
              onClick={handleReset}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </Button>

            {mode === 'streaming' && (
              <Button
                onClick={isConnected ? disconnect : connect}
                variant="outline"
                size="sm"
              >
                {isConnected ? 'Disconnect' : 'Connect'}
              </Button>
            )}
          </div>

          {/* Timeline Scrubber (Static mode only) */}
          {mode === 'static' && controls.totalFrames > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium">Timeline</label>
              <Slider
                value={[controls.currentFrame]}
                onValueChange={([value]) => handleFrameChange(value)}
                max={controls.totalFrames - 1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Frame: {controls.currentFrame}</span>
                <span>Total: {controls.totalFrames}</span>
              </div>
            </div>
          )}

          {/* Speed Control */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Speed: {controls.speed}x</label>
            <Slider
              value={[controls.speed]}
              onValueChange={([value]) => handleSpeedChange(value)}
              min={0.1}
              max={5}
              step={0.1}
              className="w-full"
            />
          </div>

          {/* Display Options */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Switch
                id="trails"
                checked={showTrails}
                onCheckedChange={setShowTrails}
              />
              <label htmlFor="trails" className="text-sm font-medium">
                Show Trails
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium">Scale:</label>
              <Slider
                value={[scale]}
                onValueChange={([value]) => setScale(value)}
                min={0.1}
                max={5}
                step={0.1}
                className="w-24"
              />
              <span className="text-sm text-muted-foreground w-8">{scale}x</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Error Display */}
      {streamingError && (
        <Card className="p-4 border-destructive">
          <p className="text-destructive text-sm">{streamingError}</p>
        </Card>
      )}
    </div>
  );
}