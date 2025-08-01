import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from './ui/sidebar';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { Switch } from './ui/switch';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Wifi, 
  WifiOff, 
  Settings, 
  Eye,
  EyeOff,
  Gauge,
  Database,
  Radio
} from 'lucide-react';
import * as OrbitTypes from '../types/orbit';

type SimulationControls = OrbitTypes.SimulationControls;

interface AppSidebarProps {
  mode: 'non-streaming' | 'streaming' | 'file';
  onModeChange: (mode: 'non-streaming' | 'streaming' | 'file') => void;
  simulationType?: 'earth-sun' | 'solar-system' | 'random';
  onSimulationTypeChange?: (type: 'earth-sun' | 'solar-system' | 'random') => void;
  controls: SimulationControls;
  onControlsChange: (controls: Partial<SimulationControls>) => void;
  showTrails: boolean;
  onShowTrailsChange: (show: boolean) => void;
  scale: number;
  onScaleChange: (scale: number) => void;
  onPlayPause: () => void;
  onReset: () => void;
  onFrameChange: (frame: number) => void;
  onSpeedChange: (speed: number) => void;
  // Streaming props
  isConnected?: boolean;
  isConnecting?: boolean;
  streamingError?: string | null;
  onConnect?: () => void;
  onDisconnect?: () => void;
  // Backend props
  backendError?: string | null;
  isLoadingSimulation?: boolean;
}

export function AppSidebar({
  mode,
  onModeChange,
  simulationType = 'earth-sun',
  onSimulationTypeChange,
  controls,
  showTrails,
  onShowTrailsChange,
  scale,
  onScaleChange,
  onPlayPause,
  onReset,
  onFrameChange,
  onSpeedChange,
  isConnected = false,
  isConnecting = false,
  streamingError = null,
  onConnect,
  onDisconnect,
  backendError = null,
  isLoadingSimulation = false
}: AppSidebarProps) {

  return (
    <Sidebar className="border-r">
      <SidebarHeader>
        <div className="px-4 py-4">
          <h2 className="text-lg font-semibold">Orbits Control</h2>
          <p className="text-sm text-muted-foreground">
            Physics-based orbital simulations
          </p>
        </div>
      </SidebarHeader>

      <SidebarContent>
        {/* Simulation Mode */}
        <SidebarGroup>
          <SidebarGroupLabel>Simulation Mode</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton 
                  onClick={() => onModeChange('non-streaming')}
                  isActive={mode === 'non-streaming'}
                  className="flex items-center gap-2"
                >
                  <Database className="w-4 h-4" />
                  <span>Pre-computed</span>
                  {mode === 'non-streaming' && <Badge variant="secondary" className="ml-auto">Active</Badge>}
                </SidebarMenuButton>
              </SidebarMenuItem>
              
              <SidebarMenuItem>
                <SidebarMenuButton 
                  onClick={() => onModeChange('streaming')}
                  isActive={mode === 'streaming'}
                  className="flex items-center gap-2"
                >
                  <Radio className="w-4 h-4" />
                  <span>Real-time Stream</span>
                  {mode === 'streaming' && <Badge variant="secondary" className="ml-auto">Active</Badge>}
                </SidebarMenuButton>
              </SidebarMenuItem>

              <SidebarMenuItem>
                <SidebarMenuButton 
                  onClick={() => onModeChange('file')}
                  disabled
                  className="flex items-center gap-2"
                >
                  <Settings className="w-4 h-4" />
                  <span>Load File</span>
                  <Badge variant="outline" className="ml-auto">Soon</Badge>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <Separator />

        {/* Simulation Type (Non-streaming mode only) */}
        {mode === 'non-streaming' && (
          <>
            <SidebarGroup>
              <SidebarGroupLabel>Simulation Type</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton 
                      onClick={() => onSimulationTypeChange?.('earth-sun')}
                      isActive={simulationType === 'earth-sun'}
                      className="flex items-center gap-2"
                    >
                      <span className="text-blue-500">🌍</span>
                      <span>Earth-Sun System</span>
                      {simulationType === 'earth-sun' && <Badge variant="secondary" className="ml-auto">Active</Badge>}
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  
                  <SidebarMenuItem>
                    <SidebarMenuButton 
                      onClick={() => onSimulationTypeChange?.('solar-system')}
                      isActive={simulationType === 'solar-system'}
                      className="flex items-center gap-2"
                    >
                      <span className="text-yellow-500">☀️</span>
                      <span>Our Solar System</span>
                      {simulationType === 'solar-system' && <Badge variant="secondary" className="ml-auto">Active</Badge>}
                    </SidebarMenuButton>
                  </SidebarMenuItem>

                  <SidebarMenuItem>
                    <SidebarMenuButton 
                      onClick={() => onSimulationTypeChange?.('random')}
                      isActive={simulationType === 'random'}
                      className="flex items-center gap-2"
                    >
                      <span className="text-purple-500">🎲</span>
                      <span>Random System</span>
                      {simulationType === 'random' && <Badge variant="secondary" className="ml-auto">Active</Badge>}
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
            <Separator />
          </>
        )}

        {/* Backend Status (Non-streaming mode only) */}
        {mode === 'non-streaming' && (
          <>
            <SidebarGroup>
              <SidebarGroupLabel>Backend Status</SidebarGroupLabel>
              <SidebarGroupContent className="px-4 space-y-3">
                <div className="flex items-center justify-between">
                  <Badge variant={backendError ? 'destructive' : 'default'} className="flex items-center gap-1">
                    {backendError ? <WifiOff className="w-3 h-3" /> : <Wifi className="w-3 h-3" />}
                    {backendError ? 'Unavailable' : 'Ready'}
                  </Badge>
                  {isLoadingSimulation && (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <div className="animate-spin rounded-full h-3 w-3 border border-muted-foreground border-t-transparent"></div>
                      Computing...
                    </div>
                  )}
                </div>
                {backendError && (
                  <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
                    Python server not running
                  </div>
                )}
              </SidebarGroupContent>
            </SidebarGroup>
            <Separator />
          </>
        )}

        {/* Connection Status (Streaming mode only) */}
        {mode === 'streaming' && (
          <>
            <SidebarGroup>
              <SidebarGroupLabel>Connection</SidebarGroupLabel>
              <SidebarGroupContent className="px-4 space-y-3">
                <div className="flex items-center justify-between">
                  <Badge variant={isConnected ? 'default' : 'destructive'} className="flex items-center gap-1">
                    {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </Badge>
                  <Button
                    onClick={isConnected ? onDisconnect : onConnect}
                    variant="outline"
                    size="sm"
                    disabled={isConnecting}
                  >
                    {isConnecting ? 'Connecting...' : isConnected ? 'Disconnect' : 'Connect'}
                  </Button>
                </div>
                {streamingError && (
                  <div className="text-xs text-destructive bg-destructive/10 p-2 rounded">
                    {streamingError}
                  </div>
                )}
              </SidebarGroupContent>
            </SidebarGroup>
            <Separator />
          </>
        )}

        {/* Current Status */}
        <SidebarGroup>
          <SidebarGroupLabel>Status</SidebarGroupLabel>
          <SidebarGroupContent className="px-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Time:</span>
              <span className="font-mono">{controls.currentTime.toFixed(1)} days</span>
            </div>
            {mode === 'non-streaming' && (
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Frame:</span>
                <span className="font-mono">{controls.currentFrame} / {controls.totalFrames}</span>
              </div>
            )}
          </SidebarGroupContent>
        </SidebarGroup>

        <Separator />

        {/* Playback Controls */}
        <SidebarGroup>
          <SidebarGroupLabel>Playback</SidebarGroupLabel>
          <SidebarGroupContent className="px-4 space-y-4">
            <div className="flex gap-2">
              <Button
                onClick={onPlayPause}
                variant="outline"
                size="sm"
                className="flex items-center gap-2 flex-1"
              >
                {controls.isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {controls.isPlaying ? 'Pause' : 'Play'}
              </Button>
              
              <Button
                onClick={onReset}
                variant="outline"
                size="sm"
                className="flex items-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>

            {/* Timeline Scrubber (Non-streaming mode only) */}
            {mode === 'non-streaming' && controls.totalFrames > 0 && (
              <div className="space-y-2">
                <label className="text-sm font-medium">Timeline</label>
                <Slider
                  value={[controls.currentFrame]}
                  onValueChange={([value]) => onFrameChange(value)}
                  max={controls.totalFrames - 1}
                  step={1}
                  className="w-full"
                />
              </div>
            )}

            {/* Speed Control */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Speed</label>
                <span className="text-sm text-muted-foreground">{controls.speed}x</span>
              </div>
              <Slider
                value={[controls.speed]}
                onValueChange={([value]) => onSpeedChange(value)}
                min={0.1}
                max={5}
                step={0.1}
                className="w-full"
              />
            </div>
          </SidebarGroupContent>
        </SidebarGroup>

        <Separator />

        {/* Display Options */}
        <SidebarGroup>
          <SidebarGroupLabel>Display</SidebarGroupLabel>
          <SidebarGroupContent className="px-4 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Switch
                  id="trails"
                  checked={showTrails}
                  onCheckedChange={onShowTrailsChange}
                />
                <label htmlFor="trails" className="text-sm font-medium">
                  Orbital Trails
                </label>
              </div>
              {showTrails ? <Eye className="w-4 h-4 text-muted-foreground" /> : <EyeOff className="w-4 h-4 text-muted-foreground" />}
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Object Scale</label>
                <span className="text-sm text-muted-foreground">{scale}x</span>
              </div>
              <Slider
                value={[scale]}
                onValueChange={([value]) => onScaleChange(value)}
                min={0.1}
                max={5}
                step={0.1}
                className="w-full"
              />
            </div>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <div className="px-4 py-2">
          <div className="text-xs text-muted-foreground">
            <div className="flex items-center gap-1">
              <Gauge className="w-3 h-3" />
              Three.js Orbital Visualization
            </div>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}