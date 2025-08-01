export interface AstroObject {
  id: number;
  name: string;
  mass: number;
  radius: number;
  color: string | number[];
}

export interface UnitsInfo {
  length: string;
  time: string;
  mass: string;
}

export interface FrameData {
  frame: number;
  time: number;
  positions: [number, number, number][];
}

export interface TrajectoryData {
  objects: AstroObject[];
  units: UnitsInfo;
  trajectory: FrameData[];
  simulation?: {
    duration_days: number;
    time_step: number;
    num_frames: number;
    integrator?: string;
  };
}

export interface StreamingMessage {
  type: 'metadata' | 'frame' | 'pong';
  objects?: AstroObject[];
  units?: UnitsInfo;
  fps?: number;
  frame?: number;
  time?: number;
  positions?: [number, number, number][];
}

export interface SimulationControls {
  isPlaying: boolean;
  speed: number;
  currentFrame: number;
  totalFrames: number;
  currentTime: number;
}