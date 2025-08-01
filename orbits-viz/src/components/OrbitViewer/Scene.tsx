import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text } from '@react-three/drei';
import * as THREE from 'three';
import * as OrbitTypes from '../../types/orbit';

type AstroObject = OrbitTypes.AstroObject;
type FrameData = OrbitTypes.FrameData;

interface AstroBodyProps {
  object: AstroObject;
  position: [number, number, number];
  scale?: number;
}

function AstroBody({ object, position, scale = 1 }: AstroBodyProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const textRef = useRef<THREE.Group>(null);
  
  // Check if this is a star/sun
  const isStar = object.name.toLowerCase().includes('sun') || object.name.toLowerCase().includes('star');
  
  // Convert color from different formats to Three.js color
  const color = useMemo(() => {
    if (typeof object.color === 'string') {
      return object.color;
    } else if (Array.isArray(object.color)) {
      // Convert RGB array [0-1] to hex
      const r = Math.round(object.color[0] * 255);
      const g = Math.round(object.color[1] * 255);
      const b = Math.round(object.color[2] * 255);
      
      // Check if this is a blue planet (like Earth/Jupiter) and make it brighter
      const isBlueish = object.color[2] > object.color[0] && object.color[2] > object.color[1];
      if (isBlueish && !isStar) {
        // Make blue planets much brighter and more vibrant
        return '#00BFFF'; // Even brighter cyan-blue
      }
      
      return `rgb(${r},${g},${b})`;
    }
    return '#ffffff';
  }, [object.color, isStar]);

  // Scale based on object radius, with minimum size for visibility
  const radius = Math.max(object.radius * scale, 0.02);

  useFrame((state) => {
    if (meshRef.current) {
      // Add a subtle rotation for visual interest
      meshRef.current.rotation.y += 0.01;
      
      // Add subtle pulsing effect for stars
      if (isStar) {
        const time = state.clock.getElapsedTime();
        const pulseFactor = 1 + Math.sin(time * 0.5) * 0.02;
        meshRef.current.scale.setScalar(pulseFactor);
      }
    }
    
    // Make text always face the camera (billboard effect)
    if (textRef.current) {
      textRef.current.lookAt(state.camera.position);
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 32]} />
        {isStar ? (
          // Special material for stars - bright and glowing
          <meshBasicMaterial 
            color={color}
            transparent
            opacity={0.9}
          />
        ) : (
          // Standard material for planets
          <meshStandardMaterial 
            color={color}
            metalness={0.1}
            roughness={0.7}
          />
        )}
      </mesh>
      
      {/* Add glow effect for stars */}
      {isStar && (
        <mesh>
          <sphereGeometry args={[radius * 1.2, 16, 16]} />
          <meshBasicMaterial 
            color={color}
            transparent
            opacity={0.05}
          />
        </mesh>
      )}
      
      {/* Object label */}
      <group ref={textRef} position={[0, radius + 0.1, 0]}>
        <Text
          fontSize={0.05}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {object.name}
        </Text>
      </group>
    </group>
  );
}

interface OrbitTrailProps {
  positions: [number, number, number][];
  color: string;
  opacity?: number;
}

function OrbitTrail({ positions, color, opacity = 0.6 }: OrbitTrailProps) {
  const points = useMemo(() => {
    return positions.map(pos => new THREE.Vector3(...pos));
  }, [positions]);

  // Create a curve from the points for tube geometry
  const curve = useMemo(() => {
    if (points.length < 2) return null;
    return new THREE.CatmullRomCurve3(points);
  }, [points]);

  if (points.length < 2 || !curve) return null;

  return (
    <mesh>
      <tubeGeometry args={[curve, Math.min(points.length * 2, 200), 0.008, 8, false]} />
      <meshBasicMaterial 
        color={color} 
        opacity={opacity} 
        transparent 
      />
    </mesh>
  );
}

interface SceneProps {
  objects: AstroObject[];
  currentFrame: FrameData | null;
  trails?: Map<number, [number, number, number][]>;
  showTrails?: boolean;
  scale?: number;
}

export default function Scene({ 
  objects, 
  currentFrame, 
  trails, 
  showTrails = true, 
  scale = 1 
}: SceneProps) {
  // Find the Sun or central star for lighting
  const centralStar = objects.find(obj => 
    obj.name.toLowerCase().includes('sun') || 
    obj.name.toLowerCase().includes('star')
  );

  return (
    <Canvas
      camera={{
        position: [10, 10, 10],
        fov: 50
      }}
      style={{ background: '#000011' }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.1} />
      {centralStar && currentFrame && (
        <pointLight
          position={currentFrame.positions[objects.indexOf(centralStar)] || [0, 0, 0]}
          intensity={2}
          color="#ffffff"
          decay={2}
          distance={100}
        />
      )}

      {/* Background stars */}
      <Stars
        radius={100}
        depth={50}
        count={5000}
        factor={4}
        saturation={0}
        fade
        speed={1}
      />

      {/* Render astronomical objects */}
      {currentFrame && objects.map((object, index) => {
        const position = currentFrame.positions[index];
        if (!position) return null;

        return (
          <AstroBody
            key={object.id}
            object={object}
            position={position}
            scale={scale}
          />
        );
      })}

      {/* Render orbital trails */}
      {showTrails && trails && objects.map((object) => {
        const trail = trails.get(object.id);
        if (!trail || trail.length < 2) return null;

        // Use the same color logic as the AstroBody component
        let color: string;
        if (typeof object.color === 'string') {
          color = object.color;
        } else if (Array.isArray(object.color)) {
          // Check if this is a blue planet and use bright blue for trail too
          const isBlueish = object.color[2] > object.color[0] && object.color[2] > object.color[1];
          const isStar = object.name.toLowerCase().includes('sun') || object.name.toLowerCase().includes('star');
          
          if (isBlueish && !isStar) {
            color = '#00BFFF'; // Even brighter cyan-blue
          } else {
            color = `rgb(${Math.round(object.color[0] * 255)},${Math.round(object.color[1] * 255)},${Math.round(object.color[2] * 255)})`;
          }
        } else {
          color = '#ffffff';
        }

        return (
          <OrbitTrail
            key={`trail-${object.id}`}
            positions={trail}
            color={color}
            opacity={0.8}
          />
        );
      })}

      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        zoomSpeed={0.6}
        panSpeed={0.8}
        rotateSpeed={0.4}
      />
    </Canvas>
  );
}