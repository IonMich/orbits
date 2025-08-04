import { useState, useEffect, useRef, useCallback } from 'react';
import * as OrbitTypes from '../types/orbit';

type StreamingMessage = OrbitTypes.StreamingMessage;
type FrameData = OrbitTypes.FrameData;
type AstroObject = OrbitTypes.AstroObject;
type UnitsInfo = OrbitTypes.UnitsInfo;

interface UseWebSocketOptions {
  url: string;
  autoConnect?: boolean;
  simulationType?: 'earth-sun' | 'solar-system' | 'random';
}

interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  objects: AstroObject[] | null;
  units: UnitsInfo | null;
  latestFrame: FrameData | null;
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (message: any) => void;
  requestSimulation: (type: 'earth-sun' | 'solar-system' | 'random') => void;
}

export function useWebSocket({ url, autoConnect = false, simulationType }: UseWebSocketOptions): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [objects, setObjects] = useState<AstroObject[] | null>(null);
  const [units, setUnits] = useState<UnitsInfo | null>(null);
  const [latestFrame, setLatestFrame] = useState<FrameData | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Clean up any existing connection first
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnecting(true);
    setError(null);

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const message: StreamingMessage = JSON.parse(event.data);
          
          switch (message.type) {
            case 'metadata':
              if (message.objects) setObjects(message.objects);
              if (message.units) setUnits(message.units);
              console.log('📡 Received metadata:', message);
              break;
              
            case 'frame':
              if (message.frame !== undefined && message.time !== undefined && message.positions) {
                console.log(`🎬 Frame ${message.frame} at time ${message.time.toFixed(1)}d`);
                setLatestFrame({
                  frame: message.frame,
                  time: message.time,
                  positions: message.positions
                });
              } else {
                console.warn('⚠️ Invalid frame data:', message);
              }
              break;
              
            case 'pong':
              // Handle ping/pong for connection health
              break;
              
            default:
              console.log('📨 Unknown message type:', message.type, message);
          }
        } catch (err) {
          console.error('❌ Failed to parse WebSocket message:', err, 'Raw data:', event.data);
          setError('Failed to parse server message');
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setIsConnecting(false);
        wsRef.current = null;

        // Auto-reconnect after 3 seconds if it wasn't a manual disconnect
        if (event.code !== 1000) {
          setError('Connection lost. Reconnecting...');
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        console.error('WebSocket readyState:', ws.readyState);
        console.error('WebSocket URL:', ws.url);
        setError('WebSocket connection error');
        setIsConnecting(false);
      };

    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError('Failed to create WebSocket connection');
      setIsConnecting(false);
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setIsConnecting(false);
    setError(null);
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const requestSimulation = useCallback((type: 'earth-sun' | 'solar-system' | 'random') => {
    sendMessage({
      type: 'change_simulation',
      simulation_type: type
    });
  }, [sendMessage]);

  useEffect(() => {
    let mounted = true;
    
    if (autoConnect && mounted) {
      // Small delay to prevent React development mode double-mounting issues
      const timer = setTimeout(() => {
        if (mounted) {
          connect();
        }
      }, 100);
      
      return () => {
        mounted = false;
        clearTimeout(timer);
        disconnect();
      };
    }

    return () => {
      mounted = false;
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Send simulation type request when simulationType changes
  useEffect(() => {
    if (isConnected && simulationType) {
      requestSimulation(simulationType);
    }
  }, [isConnected, simulationType, requestSimulation]);


  return {
    isConnected,
    isConnecting,
    objects,
    units,
    latestFrame,
    error,
    connect,
    disconnect,
    sendMessage,
    requestSimulation,
  };
}