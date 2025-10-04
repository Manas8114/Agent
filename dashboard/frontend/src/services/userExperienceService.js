/**
 * User Experience Service
 * Handles real-time data integration for gaming and streaming metrics
 */

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const fetchRealTimeData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/real-data`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn('Real-time data not available, using simulation:', error.message);
    return null;
  }
};

export const fetchSystemMetrics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn('System metrics not available, using simulation:', error.message);
    return null;
  }
};

export const fetchHealthData = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn('Health data not available, using simulation:', error.message);
    return null;
  }
};

/**
 * Simulate realistic gaming metrics based on network conditions
 */
export const simulateGamingMetrics = (networkData = null) => {
  const baseFPS = 60;
  const basePing = 45;
  const baseJitter = 2.1;
  const basePacketLoss = 0.1;

  // Use real network data if available
  let latency = basePing;
  let jitter = baseJitter;
  let packetLoss = basePacketLoss;

  if (networkData?.kpis) {
    latency = networkData.kpis.latency_ms || basePing;
    jitter = networkData.kpis.jitter_ms || baseJitter;
    packetLoss = (networkData.kpis.packet_loss_rate * 100) || basePacketLoss;
  }

  // Simulate realistic variations
  const fpsVariation = (Math.random() - 0.5) * 20;
  const pingVariation = (Math.random() - 0.5) * 15;
  const jitterVariation = (Math.random() - 0.5) * 1.5;
  const packetLossVariation = (Math.random() - 0.5) * 0.3;

  return {
    fps: Math.max(30, Math.min(144, baseFPS + fpsVariation)),
    ping: Math.max(20, Math.min(200, latency + pingVariation)),
    jitter: Math.max(0.5, Math.min(15, jitter + jitterVariation)),
    packetLoss: Math.max(0, Math.min(5, packetLoss + packetLossVariation)),
    serverIP: generateServerIP(),
    serverName: generateServerName(),
    isConnected: true,
    timestamp: new Date().toISOString()
  };
};

/**
 * Simulate realistic YouTube streaming metrics
 */
export const simulateStreamingMetrics = (networkData = null) => {
  const baseBuffering = 2.1;
  const baseStartupDelay = 1.2;
  const baseSmoothness = 95.5;

  // Determine resolution based on network quality
  let resolution = '1080p';
  if (networkData?.kpis) {
    const throughput = networkData.kpis.throughput_mbps || 100;
    const latency = networkData.kpis.latency_ms || 50;
    
    if (throughput > 50 && latency < 30) {
      resolution = '4K';
    } else if (throughput > 25 && latency < 50) {
      resolution = '1080p';
    } else if (throughput > 10 && latency < 100) {
      resolution = '720p';
    } else {
      resolution = '480p';
    }
  }

  // Simulate realistic variations
  const bufferingVariation = (Math.random() - 0.5) * 2;
  const startupVariation = (Math.random() - 0.5) * 0.5;
  const smoothnessVariation = (Math.random() - 0.5) * 5;

  return {
    buffering: Math.max(0, Math.min(20, baseBuffering + bufferingVariation)),
    resolution,
    startupDelay: Math.max(0.5, Math.min(10, baseStartupDelay + startupVariation)),
    smoothness: Math.max(70, Math.min(100, baseSmoothness + smoothnessVariation)),
    isPlaying: true,
    timestamp: new Date().toISOString()
  };
};

/**
 * Simulate YouTube-specific streaming metrics with enhanced realism
 */
export const simulateYouTubeMetrics = (networkData = null) => {
  const baseBuffering = 1.8;
  const baseStartupDelay = 0.9;
  const baseSmoothness = 97.2;

  // Determine resolution based on network quality with YouTube-specific logic
  let resolution = '1080p';
  let bufferingMultiplier = 1.0;
  
  if (networkData?.kpis) {
    const throughput = networkData.kpis.throughput_mbps || 100;
    const latency = networkData.kpis.latency_ms || 50;
    const jitter = networkData.kpis.jitter_ms || 2.0;
    
    // YouTube resolution logic based on bandwidth
    if (throughput > 60 && latency < 25 && jitter < 2) {
      resolution = '4K';
      bufferingMultiplier = 0.8;
    } else if (throughput > 35 && latency < 40 && jitter < 3) {
      resolution = '1080p';
      bufferingMultiplier = 0.9;
    } else if (throughput > 15 && latency < 60 && jitter < 5) {
      resolution = '720p';
      bufferingMultiplier = 1.1;
    } else {
      resolution = '480p';
      bufferingMultiplier = 1.3;
    }
  }

  // Simulate realistic variations with YouTube-specific patterns
  const bufferingVariation = (Math.random() - 0.5) * 1.5 * bufferingMultiplier;
  const startupVariation = (Math.random() - 0.5) * 0.3;
  const smoothnessVariation = (Math.random() - 0.5) * 3;

  return {
    buffering: Math.max(0, Math.min(15, baseBuffering + bufferingVariation)),
    resolution,
    startupDelay: Math.max(0.3, Math.min(8, baseStartupDelay + startupVariation)),
    smoothness: Math.max(75, Math.min(100, baseSmoothness + smoothnessVariation)),
    isPlaying: true,
    timestamp: new Date().toISOString()
  };
};

/**
 * Generate realistic server IP addresses
 */
const generateServerIP = () => {
  const regions = [
    { prefix: '192.168.1', name: 'East-1' },
    { prefix: '192.168.2', name: 'West-1' },
    { prefix: '10.0.1', name: 'North-1' },
    { prefix: '10.0.2', name: 'South-1' },
    { prefix: '172.16.1', name: 'Central-1' }
  ];
  
  const region = regions[Math.floor(Math.random() * regions.length)];
  const lastOctet = Math.floor(Math.random() * 254) + 1;
  
  return {
    ip: `${region.prefix}.${lastOctet}`,
    name: `Server-${region.name}`
  };
};

/**
 * Generate realistic server names
 */
const generateServerName = () => {
  const servers = [
    'Server-East-1', 'Server-West-1', 'Server-North-1', 
    'Server-South-1', 'Server-Central-1', 'Server-Europe-1',
    'Server-Asia-1', 'Server-Americas-1'
  ];
  
  return servers[Math.floor(Math.random() * servers.length)];
};

/**
 * Calculate AI improvement metrics
 */
export const calculateAIImprovements = (currentMetrics, beforeMetrics) => {
  const fpsImprovement = ((currentMetrics.fps - beforeMetrics.fps) / beforeMetrics.fps * 100);
  const pingImprovement = ((beforeMetrics.ping - currentMetrics.ping) / beforeMetrics.ping * 100);
  const bufferingImprovement = ((beforeMetrics.buffering - currentMetrics.buffering) / beforeMetrics.buffering * 100);
  
  return {
    fpsImprovement: Math.max(0, fpsImprovement),
    pingImprovement: Math.max(0, pingImprovement),
    bufferingImprovement: Math.max(0, bufferingImprovement),
    overallImprovement: (fpsImprovement + pingImprovement + bufferingImprovement) / 3
  };
};

/**
 * Generate before AI metrics (simulated poor performance)
 */
export const generateBeforeAIMetrics = () => {
  return {
    gaming: {
      fps: 35 + Math.random() * 15, // 35-50 FPS
      ping: 80 + Math.random() * 40, // 80-120ms
      jitter: 5 + Math.random() * 5, // 5-10ms
      packetLoss: 1.5 + Math.random() * 2 // 1.5-3.5%
    },
    streaming: {
      buffering: 6 + Math.random() * 4, // 6-10%
      resolution: '480p',
      startupDelay: 3 + Math.random() * 2, // 3-5s
      smoothness: 70 + Math.random() * 10 // 70-80%
    }
  };
};

const userExperienceService = {
  fetchRealTimeData,
  fetchSystemMetrics,
  fetchHealthData,
  simulateGamingMetrics,
  simulateStreamingMetrics,
  calculateAIImprovements,
  generateBeforeAIMetrics
};

export default userExperienceService;
