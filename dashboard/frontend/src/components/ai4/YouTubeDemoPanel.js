import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  RefreshCw, 
  Eye, 
  EyeOff,
  TrendingUp,
  TrendingDown,
  Zap,
  Server,
  Activity
} from 'lucide-react';

// Services
import { 
  fetchRealTimeData, 
  simulateYouTubeMetrics,
  calculateAIImprovements,
  generateBeforeAIMetrics
} from '../../services/userExperienceService';

const YouTubeDemoPanel = ({ data, onRefresh }) => {
  const [showOverlay, setShowOverlay] = useState(true);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [streamingMetrics, setStreamingMetrics] = useState({
    buffering: 2.1,
    resolution: '1080p',
    startupDelay: 1.2,
    smoothness: 95.5,
    serverIP: '192.168.1.100',
    serverName: 'Server-East-1',
    bandwidth: 125.5,
    latency: 45
  });
  const [beforeAIMetrics, setBeforeAIMetrics] = useState({
    buffering: 8.7,
    resolution: '480p',
    startupDelay: 4.8,
    smoothness: 78.2,
    serverIP: '192.168.1.50',
    serverName: 'Server-East-1-Basic'
  });
  const [aiAllocation, setAiAllocation] = useState({
    isActive: true,
    improvement: 0.0,
    serverLoad: 0.65,
    networkOptimization: 0.85,
    lastOptimization: new Date()
  });
  const [realTimeData, setRealTimeData] = useState(null);

  // YouTube video list (trending videos)
  const youtubeVideos = [
    {
      id: 'dQw4w9WgXcQ',
      title: 'Never Gonna Give You Up - Rick Astley',
      thumbnail: 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg'
    },
    {
      id: '9bZkp7q19f0',
      title: 'PSY - GANGNAM STYLE',
      thumbnail: 'https://img.youtube.com/vi/9bZkp7q19f0/maxresdefault.jpg'
    },
    {
      id: 'kJQP7kiw5Fk',
      title: 'Luis Fonsi - Despacito ft. Daddy Yankee',
      thumbnail: 'https://img.youtube.com/vi/kJQP7kiw5Fk/maxresdefault.jpg'
    },
    {
      id: 'YQHsXMglC9A',
      title: 'Adele - Hello',
      thumbnail: 'https://img.youtube.com/vi/YQHsXMglC9A/maxresdefault.jpg'
    },
    {
      id: 'fJ9rUzIMcZQ',
      title: 'Queen - Bohemian Rhapsody',
      thumbnail: 'https://img.youtube.com/vi/fJ9rUzIMcZQ/maxresdefault.jpg'
    }
  ];

  // Select random video
  const selectRandomVideo = useCallback(() => {
    const randomIndex = Math.floor(Math.random() * youtubeVideos.length);
    setCurrentVideo(youtubeVideos[randomIndex]);
  }, [youtubeVideos]);

  // Initialize with random video
  useEffect(() => {
    selectRandomVideo();
  }, [selectRandomVideo]);

  // Simulate real-time streaming metrics
  const updateStreamingMetrics = useCallback(async () => {
    try {
      // Try to fetch real data first
      const realData = await fetchRealTimeData();
      
      // Generate new metrics using the YouTube-specific service
      const newStreamingMetrics = simulateYouTubeMetrics(realData);
      
      // Update streaming metrics
      setStreamingMetrics(prev => ({
        ...prev,
        ...newStreamingMetrics,
        serverIP: generateServerIP(),
        serverName: generateServerName(),
        bandwidth: realData?.kpis?.throughput_mbps || prev.bandwidth,
        latency: realData?.kpis?.latency_ms || prev.latency
      }));

      // Calculate AI improvements
      const improvements = calculateAIImprovements(
        { buffering: newStreamingMetrics.buffering, resolution: newStreamingMetrics.resolution },
        { buffering: beforeAIMetrics.buffering, resolution: beforeAIMetrics.resolution }
      );

      // Update AI allocation
      setAiAllocation(prev => ({
        ...prev,
        improvement: improvements.overallImprovement / 100,
        serverLoad: Math.max(0.3, Math.min(0.9, prev.serverLoad + (Math.random() - 0.5) * 0.05)),
        networkOptimization: Math.max(0.7, Math.min(1.0, prev.networkOptimization + (Math.random() - 0.3) * 0.05)),
        lastOptimization: new Date()
      }));
    } catch (error) {
      console.warn('Failed to update streaming metrics:', error);
    }
  }, [beforeAIMetrics]);

  // Generate server IP and name
  const generateServerIP = () => {
    const regions = [
      { prefix: '192.168.1', name: 'East-1' },
      { prefix: '192.168.2', name: 'West-1' },
      { prefix: '10.0.1', name: 'North-1' },
      { prefix: '10.0.2', name: 'South-1' }
    ];
    const region = regions[Math.floor(Math.random() * regions.length)];
    const lastOctet = Math.floor(Math.random() * 254) + 1;
    return `${region.prefix}.${lastOctet}`;
  };

  const generateServerName = () => {
    const servers = ['Server-East-1', 'Server-West-1', 'Server-North-1', 'Server-South-1'];
    return servers[Math.floor(Math.random() * servers.length)];
  };

  // Auto-refresh every 2 seconds
  useEffect(() => {
    const interval = setInterval(updateStreamingMetrics, 2000);
    return () => clearInterval(interval);
  }, [updateStreamingMetrics]);

  // Initialize before AI metrics
  useEffect(() => {
    const beforeAIMetrics = generateBeforeAIMetrics();
    setBeforeAIMetrics(beforeAIMetrics.streaming);
  }, []);

  // Fetch real data on mount
  useEffect(() => {
    const fetchRealData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/real-data');
        if (response.ok) {
          const realData = await response.json();
          setRealTimeData(realData);
        }
      } catch (error) {
        console.log('Real data not available, using simulation');
      }
    };

    fetchRealData();
  }, []);

  const getBufferingColor = (buffering) => {
    if (buffering <= 2) return 'text-green-500';
    if (buffering <= 5) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getResolutionColor = (resolution) => {
    switch (resolution) {
      case '4K': return 'text-purple-500';
      case '1080p': return 'text-green-500';
      case '720p': return 'text-yellow-500';
      case '480p': return 'text-orange-500';
      default: return 'text-red-500';
    }
  };

  const getStartupDelayColor = (delay) => {
    if (delay <= 2) return 'text-green-500';
    if (delay <= 5) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getResolutionLevel = (resolution) => {
    switch (resolution) {
      case '4K': return 4;
      case '1080p': return 3;
      case '720p': return 2;
      case '480p': return 1;
      default: return 0;
    }
  };

  const bufferingImprovement = ((beforeAIMetrics.buffering - streamingMetrics.buffering) / beforeAIMetrics.buffering * 100).toFixed(1);
  const resolutionImprovement = getResolutionLevel(streamingMetrics.resolution) - getResolutionLevel(beforeAIMetrics.resolution);
  const startupImprovement = ((beforeAIMetrics.startupDelay - streamingMetrics.startupDelay) / beforeAIMetrics.startupDelay * 100).toFixed(1);

  return (
    <div className="dashboard-panel-full dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            YouTube Live Demo
          </h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={selectRandomVideo}
              className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900 hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
            >
              <RefreshCw className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            </button>
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              {showOverlay ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${
              aiAllocation.isActive 
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
            }`}>
              {aiAllocation.isActive ? 'AI Active' : 'AI Inactive'}
            </div>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* YouTube Video Player */}
        <div className="mb-6">
          <div className="relative bg-black rounded-lg overflow-hidden">
            {currentVideo && (
              <iframe
                width="100%"
                height="400"
                src={`https://www.youtube.com/embed/${currentVideo.id}?autoplay=0&controls=1&showinfo=0&rel=0`}
                title={currentVideo.title}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full h-96"
              />
            )}
            
            {/* Metrics Overlay */}
            <AnimatePresence>
              {showOverlay && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className="absolute top-4 right-4 bg-black bg-opacity-80 text-white p-4 rounded-lg backdrop-blur-sm"
                  style={{ minWidth: '280px' }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold">Stats for Nerds</h3>
                    <div className="flex items-center space-x-1">
                      <Activity className="w-3 h-3 text-green-400" />
                      <span className="text-xs text-green-400">Live</span>
                    </div>
                  </div>
                  
                  {/* Real-time Metrics */}
                  <div className="space-y-2">
                    {/* Buffering */}
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-300">Buffering</span>
                      <motion.span 
                        className={`text-sm font-bold ${getBufferingColor(streamingMetrics.buffering)}`}
                        key={streamingMetrics.buffering}
                        initial={{ scale: 1.1 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.2 }}
                      >
                        {streamingMetrics.buffering.toFixed(1)}%
                      </motion.span>
                    </div>
                    
                    {/* Resolution */}
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-300">Resolution</span>
                      <motion.span 
                        className={`text-sm font-bold ${getResolutionColor(streamingMetrics.resolution)}`}
                        key={streamingMetrics.resolution}
                        initial={{ scale: 1.1 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.2 }}
                      >
                        {streamingMetrics.resolution}
                      </motion.span>
                    </div>
                    
                    {/* Startup Delay */}
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-300">Startup</span>
                      <motion.span 
                        className={`text-sm font-bold ${getStartupDelayColor(streamingMetrics.startupDelay)}`}
                        key={streamingMetrics.startupDelay}
                        initial={{ scale: 1.1 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.2 }}
                      >
                        {streamingMetrics.startupDelay.toFixed(1)}s
                      </motion.span>
                    </div>
                    
                    {/* Server Info */}
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-300">Server</span>
                      <div className="text-right">
                        <div className="text-sm font-bold text-blue-400">{streamingMetrics.serverName}</div>
                        <div className="text-xs text-gray-400">{streamingMetrics.serverIP}</div>
                      </div>
                    </div>
                    
                    {/* Bandwidth & Latency */}
                    <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-600">
                      <div className="text-center">
                        <div className="text-xs text-gray-300">Bandwidth</div>
                        <div className="text-sm font-bold text-green-400">{streamingMetrics.bandwidth.toFixed(1)} Mbps</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs text-gray-300">Latency</div>
                        <div className="text-sm font-bold text-blue-400">{streamingMetrics.latency.toFixed(0)}ms</div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* AI Improvement Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Before AI */}
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <TrendingDown className="w-5 h-5 text-red-500 mr-2" />
              <h3 className="text-lg font-semibold text-red-700 dark:text-red-300">Before AI Allocation</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Buffering</span>
                <span className="text-lg font-bold text-red-500">{beforeAIMetrics.buffering.toFixed(1)}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Resolution</span>
                <span className="text-lg font-bold text-orange-500">{beforeAIMetrics.resolution}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Startup Delay</span>
                <span className="text-lg font-bold text-red-500">{beforeAIMetrics.startupDelay.toFixed(1)}s</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Server</span>
                <span className="text-sm font-bold text-gray-600">{beforeAIMetrics.serverName}</span>
              </div>
            </div>
          </div>

          {/* After AI */}
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <TrendingUp className="w-5 h-5 text-green-500 mr-2" />
              <h3 className="text-lg font-semibold text-green-700 dark:text-green-300">After AI Allocation</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Buffering</span>
                <div className="text-right">
                  <span className="text-lg font-bold text-green-500">{streamingMetrics.buffering.toFixed(1)}%</span>
                  <div className="text-xs text-green-600">-{bufferingImprovement}%</div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Resolution</span>
                <div className="text-right">
                  <span className={`text-lg font-bold ${getResolutionColor(streamingMetrics.resolution)}`}>
                    {streamingMetrics.resolution}
                  </span>
                  <div className="text-xs text-green-600">+{resolutionImprovement} levels</div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Startup Delay</span>
                <div className="text-right">
                  <span className="text-lg font-bold text-green-500">{streamingMetrics.startupDelay.toFixed(1)}s</span>
                  <div className="text-xs text-green-600">-{startupImprovement}%</div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Server</span>
                <span className="text-sm font-bold text-blue-600">{streamingMetrics.serverName}</span>
              </div>
            </div>
          </div>
        </div>

        {/* AI Allocation Status */}
        <div className="mt-6 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Zap className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">AI Network Allocation</h3>
            </div>
            <div className="flex items-center space-x-2">
              <Server className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {streamingMetrics.serverName} ({streamingMetrics.serverIP})
              </span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {(aiAllocation.improvement * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Overall Improvement</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {(aiAllocation.networkOptimization * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Network Optimization</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(aiAllocation.serverLoad * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Server Load</div>
            </div>
          </div>
          
          <div className="mt-3 text-sm text-gray-600 dark:text-gray-400">
            Last optimization: {aiAllocation.lastOptimization.toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default YouTubeDemoPanel;
