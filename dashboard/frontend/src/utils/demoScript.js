/**
 * Demo Script for User Experience Dashboard
 * Simulates realistic gaming and streaming scenarios to showcase AI improvements
 */

export const demoScenarios = {
  // Scenario 1: Gaming Performance Improvement
  gamingImprovement: {
    name: "Gaming Performance Improvement",
    description: "Shows how AI allocation improves gaming performance",
    duration: 30000, // 30 seconds
    steps: [
      {
        time: 0,
        action: "start",
        data: {
          fps: 45,
          ping: 120,
          jitter: 8.5,
          packetLoss: 2.3,
          serverIP: "192.168.1.50",
          serverName: "Server-East-1"
        }
      },
      {
        time: 5000,
        action: "ai_activation",
        data: {
          fps: 55,
          ping: 80,
          jitter: 5.2,
          packetLoss: 1.1,
          serverIP: "192.168.1.100",
          serverName: "Server-East-1-Optimized"
        }
      },
      {
        time: 15000,
        action: "optimization",
        data: {
          fps: 65,
          ping: 45,
          jitter: 2.1,
          packetLoss: 0.3,
          serverIP: "192.168.1.100",
          serverName: "Server-East-1-Optimized"
        }
      },
      {
        time: 25000,
        action: "peak_performance",
        data: {
          fps: 75,
          ping: 35,
          jitter: 1.2,
          packetLoss: 0.1,
          serverIP: "192.168.1.100",
          serverName: "Server-East-1-Optimized"
        }
      }
    ]
  },

  // Scenario 2: YouTube Streaming Improvement
  streamingImprovement: {
    name: "YouTube Streaming Improvement",
    description: "Shows how AI allocation improves streaming quality",
    duration: 30000, // 30 seconds
    steps: [
      {
        time: 0,
        action: "start",
        data: {
          buffering: 8.7,
          resolution: "480p",
          startupDelay: 4.8,
          smoothness: 78.2,
          isPlaying: true
        }
      },
      {
        time: 5000,
        action: "ai_activation",
        data: {
          buffering: 5.2,
          resolution: "720p",
          startupDelay: 2.1,
          smoothness: 85.5,
          isPlaying: true
        }
      },
      {
        time: 15000,
        action: "optimization",
        data: {
          buffering: 2.8,
          resolution: "1080p",
          startupDelay: 1.5,
          smoothness: 92.3,
          isPlaying: true
        }
      },
      {
        time: 25000,
        action: "peak_performance",
        data: {
          buffering: 1.2,
          resolution: "4K",
          startupDelay: 0.8,
          smoothness: 98.7,
          isPlaying: true
        }
      }
    ]
  },

  // Scenario 3: Network Congestion Handling
  networkCongestion: {
    name: "Network Congestion Handling",
    description: "Shows how AI handles network congestion",
    duration: 45000, // 45 seconds
    steps: [
      {
        time: 0,
        action: "normal_operation",
        data: {
          fps: 60,
          ping: 45,
          buffering: 2.1,
          resolution: "1080p",
          serverLoad: 0.65
        }
      },
      {
        time: 10000,
        action: "congestion_detected",
        data: {
          fps: 45,
          ping: 85,
          buffering: 5.2,
          resolution: "720p",
          serverLoad: 0.85
        }
      },
      {
        time: 20000,
        action: "ai_optimization",
        data: {
          fps: 55,
          ping: 60,
          buffering: 3.1,
          resolution: "1080p",
          serverLoad: 0.75
        }
      },
      {
        time: 35000,
        action: "recovery",
        data: {
          fps: 65,
          ping: 40,
          buffering: 1.8,
          resolution: "1080p",
          serverLoad: 0.70
        }
      }
    ]
  }
};

/**
 * Run a demo scenario
 */
export const runDemoScenario = (scenarioName, onUpdate, onComplete) => {
  const scenario = demoScenarios[scenarioName];
  if (!scenario) {
    console.error(`Scenario ${scenarioName} not found`);
    return;
  }

  let currentStep = 0;
  const startTime = Date.now();

  const updateStep = () => {
    const elapsed = Date.now() - startTime;
    const currentStepData = scenario.steps[currentStep];

    if (currentStepData && elapsed >= currentStepData.time) {
      onUpdate(currentStepData.action, currentStepData.data);
      currentStep++;
    }

    if (elapsed < scenario.duration) {
      setTimeout(updateStep, 100); // Check every 100ms
    } else {
      onComplete();
    }
  };

  updateStep();
};

/**
 * Generate realistic server names and IPs
 */
export const generateServerInfo = (region = 'east') => {
  const regions = {
    east: { prefix: '192.168.1', name: 'East' },
    west: { prefix: '192.168.2', name: 'West' },
    north: { prefix: '10.0.1', name: 'North' },
    south: { prefix: '10.0.2', name: 'South' },
    central: { prefix: '172.16.1', name: 'Central' }
  };

  const regionData = regions[region] || regions.east;
  const lastOctet = Math.floor(Math.random() * 254) + 1;
  
  return {
    ip: `${regionData.prefix}.${lastOctet}`,
    name: `Server-${regionData.name}-${Math.floor(Math.random() * 5) + 1}`
  };
};

/**
 * Calculate performance improvements
 */
export const calculateImprovements = (before, after) => {
  const fpsImprovement = ((after.fps - before.fps) / before.fps * 100).toFixed(1);
  const pingImprovement = ((before.ping - after.ping) / before.ping * 100).toFixed(1);
  const bufferingImprovement = ((before.buffering - after.buffering) / before.buffering * 100).toFixed(1);
  
  return {
    fpsImprovement: Math.max(0, fpsImprovement),
    pingImprovement: Math.max(0, pingImprovement),
    bufferingImprovement: Math.max(0, bufferingImprovement),
    overallImprovement: ((parseFloat(fpsImprovement) + parseFloat(pingImprovement) + parseFloat(bufferingImprovement)) / 3).toFixed(1)
  };
};

/**
 * Generate realistic network conditions
 */
export const generateNetworkConditions = (quality = 'good') => {
  const conditions = {
    excellent: {
      latency: { min: 20, max: 35 },
      jitter: { min: 0.5, max: 1.5 },
      packetLoss: { min: 0, max: 0.1 },
      throughput: { min: 100, max: 150 }
    },
    good: {
      latency: { min: 35, max: 60 },
      jitter: { min: 1.0, max: 3.0 },
      packetLoss: { min: 0.1, max: 0.5 },
      throughput: { min: 50, max: 100 }
    },
    fair: {
      latency: { min: 60, max: 100 },
      jitter: { min: 2.0, max: 5.0 },
      packetLoss: { min: 0.5, max: 1.5 },
      throughput: { min: 25, max: 50 }
    },
    poor: {
      latency: { min: 100, max: 200 },
      jitter: { min: 5.0, max: 15.0 },
      packetLoss: { min: 1.5, max: 5.0 },
      throughput: { min: 10, max: 25 }
    }
  };

  const condition = conditions[quality] || conditions.good;
  
  return {
    latency: Math.random() * (condition.latency.max - condition.latency.min) + condition.latency.min,
    jitter: Math.random() * (condition.jitter.max - condition.jitter.min) + condition.jitter.min,
    packetLoss: Math.random() * (condition.packetLoss.max - condition.packetLoss.min) + condition.packetLoss.min,
    throughput: Math.random() * (condition.throughput.max - condition.throughput.min) + condition.throughput.min
  };
};

export default {
  demoScenarios,
  runDemoScenario,
  generateServerInfo,
  calculateImprovements,
  generateNetworkConditions
};
