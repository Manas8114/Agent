import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Gamepad2, 
  Play, 
  Wifi, 
  TrendingUp, 
  TrendingDown,
  Monitor,
  Server,
  Zap,
  Clock,
  Activity
} from 'lucide-react';

// Services
import { 
  fetchRealTimeData, 
  simulateGamingMetrics, 
  simulateStreamingMetrics,
  calculateAIImprovements,
  generateBeforeAIMetrics
} from '../../services/userExperienceService';

const UserExperiencePanel = ({ data, onRefresh }) => {
  const [gamingMetrics, setGamingMetrics] = useState({
    fps: 60,
    ping: 45,
    jitter: 2.1,
    packetLoss: 0.1,
    serverIP: '192.168.1.100',
    serverName: 'Server-East-1',
    isConnected: true,
    beforeAI: {
      fps: 45,
      ping: 120,
      jitter: 8.5,
      packetLoss: 2.3
    }
  });

  const [youtubeMetrics, setYoutubeMetrics] = useState({
    buffering: 2.1,
    resolution: '1080p',
    startupDelay: 1.2,
    smoothness: 95.5,
    isPlaying: true,
    beforeAI: {
      buffering: 8.7,
      resolution: '480p',
      startupDelay: 4.8,
      smoothness: 78.2
    }
  });

  const [aiAllocation, setAiAllocation] = useState({
    isActive: true,
    improvement: 0.0,
    serverLoad: 0.65,
    networkOptimization: 0.85,
    lastOptimization: new Date()
  });

  const [realTimeData, setRealTimeData] = useState(null);

  // Simulate real-time data updates
  const updateMetrics = useCallback(async () => {
    try {
      // Try to fetch real data first
      const realData = await fetchRealTimeData();
      
      // Generate new metrics using the service
      const newGamingMetrics = simulateGamingMetrics(realData);
      const newStreamingMetrics = simulateStreamingMetrics(realData);
      
      // Update gaming metrics
      setGamingMetrics(prev => ({
        ...prev,
        ...newGamingMetrics,
        serverIP: newGamingMetrics.serverIP.ip,
        serverName: newGamingMetrics.serverName
      }));

      // Update streaming metrics
      setYoutubeMetrics(prev => ({
        ...prev,
        ...newStreamingMetrics
      }));

      // Calculate AI improvements
      const improvements = calculateAIImprovements(
        { fps: newGamingMetrics.fps, ping: newGamingMetrics.ping, buffering: newStreamingMetrics.buffering },
        { fps: gamingMetrics.beforeAI.fps, ping: gamingMetrics.beforeAI.ping, buffering: youtubeMetrics.beforeAI.buffering }
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
      console.warn('Failed to update metrics:', error);
    }
  }, [gamingMetrics.beforeAI, youtubeMetrics.beforeAI]);

  // Fetch real data if available
  const fetchRealData = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/real-data');
      if (response.ok) {
        const realData = await response.json();
        setRealTimeData(realData);
        
        // Use real data to influence simulation
        if (realData.kpis) {
          setGamingMetrics(prev => ({
            ...prev,
            ping: realData.kpis.latency_ms || prev.ping,
            jitter: realData.kpis.jitter_ms || prev.jitter,
            packetLoss: realData.kpis.packet_loss_rate * 100 || prev.packetLoss
          }));
        }
      }
    } catch (error) {
      console.log('Real data not available, using simulation');
    }
  }, []);

  // Auto-refresh every 2 seconds
  useEffect(() => {
    const interval = setInterval(updateMetrics, 2000);
    return () => clearInterval(interval);
  }, [updateMetrics]);

  // Initialize before AI metrics
  useEffect(() => {
    const beforeAIMetrics = generateBeforeAIMetrics();
    setGamingMetrics(prev => ({
      ...prev,
      beforeAI: beforeAIMetrics.gaming
    }));
    setYoutubeMetrics(prev => ({
      ...prev,
      beforeAI: beforeAIMetrics.streaming
    }));
  }, []);

  // Fetch real data on mount
  useEffect(() => {
    fetchRealData();
  }, [fetchRealData]);

  const getFPSColor = (fps) => {
    if (fps >= 60) return 'text-green-500';
    if (fps >= 30) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getPingColor = (ping) => {
    if (ping <= 50) return 'text-green-500';
    if (ping <= 100) return 'text-yellow-500';
    return 'text-red-500';
  };

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

  const fpsImprovement = ((gamingMetrics.fps - gamingMetrics.beforeAI.fps) / gamingMetrics.beforeAI.fps * 100).toFixed(1);
  const pingImprovement = ((gamingMetrics.beforeAI.ping - gamingMetrics.ping) / gamingMetrics.beforeAI.ping * 100).toFixed(1);
  const bufferingImprovement = ((youtubeMetrics.beforeAI.buffering - youtubeMetrics.buffering) / youtubeMetrics.beforeAI.buffering * 100).toFixed(1);

  return (
    <div className="dashboard-panel-full dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            User Experience - Gaming & Streaming
          </h2>
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              <Activity className="w-4 h-4 text-blue-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">Live</span>
            </div>
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
        {/* Gaming Performance Section */}
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <Gamepad2 className="w-6 h-6 text-blue-500 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Gaming Performance</h3>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Current Gaming Metrics */}
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Current Performance</h4>
              
              {/* FPS Counter */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">FPS</span>
                  <motion.span 
                    className={`text-2xl font-bold ${getFPSColor(gamingMetrics.fps)}`}
                    key={gamingMetrics.fps}
                    initial={{ scale: 1.1 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    {Math.round(gamingMetrics.fps)}
                  </motion.span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                  <motion.div 
                    className="bg-blue-500 h-2 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${(gamingMetrics.fps / 144) * 100}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>

              {/* Ping */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Ping</span>
                  <motion.span 
                    className={`text-xl font-bold ${getPingColor(gamingMetrics.ping)}`}
                    key={gamingMetrics.ping}
                    initial={{ scale: 1.1 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    {Math.round(gamingMetrics.ping)}ms
                  </motion.span>
                </div>
                <div className="flex items-center space-x-2">
                  <Wifi className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {gamingMetrics.serverName} ({gamingMetrics.serverIP})
                  </span>
                </div>
              </div>

              {/* Jitter & Packet Loss */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Jitter</div>
                  <div className="text-lg font-semibold text-gray-900 dark:text-white">
                    {gamingMetrics.jitter.toFixed(1)}ms
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Packet Loss</div>
                  <div className="text-lg font-semibold text-gray-900 dark:text-white">
                    {gamingMetrics.packetLoss.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Before/After Comparison */}
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">AI Improvement</h4>
              
              {/* FPS Improvement */}
              <div className="bg-gradient-to-r from-red-50 to-green-50 dark:from-red-900/20 dark:to-green-900/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">FPS Improvement</span>
                  <span className="text-sm font-bold text-green-600">+{fpsImprovement}%</span>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Before AI</div>
                    <div className="text-lg font-bold text-red-500">{gamingMetrics.beforeAI.fps}</div>
                  </div>
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">After AI</div>
                    <div className="text-lg font-bold text-green-500">{Math.round(gamingMetrics.fps)}</div>
                  </div>
                </div>
              </div>

              {/* Ping Improvement */}
              <div className="bg-gradient-to-r from-red-50 to-green-50 dark:from-red-900/20 dark:to-green-900/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Ping Reduction</span>
                  <span className="text-sm font-bold text-green-600">-{pingImprovement}%</span>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Before AI</div>
                    <div className="text-lg font-bold text-red-500">{gamingMetrics.beforeAI.ping}ms</div>
                  </div>
                  <TrendingDown className="w-5 h-5 text-green-500" />
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">After AI</div>
                    <div className="text-lg font-bold text-green-500">{Math.round(gamingMetrics.ping)}ms</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* YouTube Streaming Section */}
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <Play className="w-6 h-6 text-red-500 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">YouTube Streaming</h3>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Current Streaming Metrics */}
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Current Performance</h4>
              
              {/* Buffering Indicator */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Buffering</span>
                  <motion.span 
                    className={`text-xl font-bold ${getBufferingColor(youtubeMetrics.buffering)}`}
                    key={youtubeMetrics.buffering}
                    initial={{ scale: 1.1 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    {youtubeMetrics.buffering.toFixed(1)}%
                  </motion.span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                  <motion.div 
                    className={`h-2 rounded-full ${
                      youtubeMetrics.buffering <= 2 ? 'bg-green-500' :
                      youtubeMetrics.buffering <= 5 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(youtubeMetrics.buffering * 5, 100)}%` }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              </div>

              {/* Resolution */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Resolution</span>
                  <motion.span 
                    className={`text-xl font-bold ${getResolutionColor(youtubeMetrics.resolution)}`}
                    key={youtubeMetrics.resolution}
                    initial={{ scale: 1.1 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.2 }}
                  >
                    {youtubeMetrics.resolution}
                  </motion.span>
                </div>
                <div className="flex items-center space-x-2">
                  <Monitor className="w-4 h-4 text-blue-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Smoothness: {youtubeMetrics.smoothness.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Startup Delay */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Startup Delay</span>
                  <span className="text-lg font-bold text-gray-900 dark:text-white">
                    {youtubeMetrics.startupDelay.toFixed(1)}s
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-purple-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {youtubeMetrics.isPlaying ? 'Playing' : 'Paused'}
                  </span>
                </div>
              </div>
            </div>

            {/* Before/After Comparison */}
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">AI Improvement</h4>
              
              {/* Buffering Improvement */}
              <div className="bg-gradient-to-r from-red-50 to-green-50 dark:from-red-900/20 dark:to-green-900/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Buffering Reduction</span>
                  <span className="text-sm font-bold text-green-600">-{bufferingImprovement}%</span>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Before AI</div>
                    <div className="text-lg font-bold text-red-500">{youtubeMetrics.beforeAI.buffering.toFixed(1)}%</div>
                  </div>
                  <TrendingDown className="w-5 h-5 text-green-500" />
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">After AI</div>
                    <div className="text-lg font-bold text-green-500">{youtubeMetrics.buffering.toFixed(1)}%</div>
                  </div>
                </div>
              </div>

              {/* Resolution Improvement */}
              <div className="bg-gradient-to-r from-orange-50 to-purple-50 dark:from-orange-900/20 dark:to-purple-900/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Resolution Upgrade</span>
                  <span className="text-sm font-bold text-purple-600">+2 Levels</span>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">Before AI</div>
                    <div className="text-lg font-bold text-orange-500">{youtubeMetrics.beforeAI.resolution}</div>
                  </div>
                  <TrendingUp className="w-5 h-5 text-purple-500" />
                  <div className="text-center">
                    <div className="text-xs text-gray-500 dark:text-gray-400">After AI</div>
                    <div className="text-lg font-bold text-purple-500">{youtubeMetrics.resolution}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* AI Allocation Status */}
        <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Zap className="w-6 h-6 text-blue-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">AI Network Allocation</h3>
            </div>
            <div className="flex items-center space-x-2">
              <Server className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {gamingMetrics.serverName} ({gamingMetrics.serverIP})
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
          
          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Last optimization: {aiAllocation.lastOptimization.toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserExperiencePanel;
