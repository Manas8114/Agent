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
  Activity,
  Play,
  Pause
} from 'lucide-react';

const YouTubeDemoPanel = ({ data, onRefresh }) => {
  const [showOverlay, setShowOverlay] = useState(true);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [streamingMetrics, setStreamingMetrics] = useState({
    buffering: 2.1,
    resolution: '1080p',
    startupDelay: 1.2,
    smoothness: 95.5,
    isPlaying: true,
    serverIP: '192.168.1.1',
    serverName: 'Server-East-1'
  });
  const [beforeAIMetrics, setBeforeAIMetrics] = useState({
    buffering: 8.7,
    resolution: '480p',
    startupDelay: 4.8,
    smoothness: 78.2,
    serverIP: '192.168.0.1',
    serverName: 'Server-Default'
  });
  const [aiAllocation, setAiAllocation] = useState({
    status: 'Optimized',
    improvement: 0.0,
    serverLoad: 0.65,
    networkOptimization: 0.85,
    lastOptimization: new Date()
  });

  // YouTube video list (trending videos)
  const youtubeVideos = [
    {
      id: 'dQw4w9WgXcQ', // Rick Astley - Never Gonna Give You Up
      title: 'Rick Astley - Never Gonna Give You Up',
      thumbnail: 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg'
    },
    {
      id: 'fJ9rUzIMcZQ', // Queen - Bohemian Rhapsody
      title: 'Queen - Bohemian Rhapsody',
      thumbnail: 'https://img.youtube.com/vi/fJ9rUzIMcZQ/maxresdefault.jpg'
    },
    {
      id: 'kfVsfOSOFMk', // Luis Fonsi - Despacito ft. Daddy Yankee
      title: 'Luis Fonsi - Despacito ft. Daddy Yankee',
      thumbnail: 'https://img.youtube.com/vi/kfVsfOSOFMk/maxresdefault.jpg'
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

  const updateStreamingMetrics = useCallback(async () => {
    try {
      // Generate new metrics with realistic variations
      const newStreamingMetrics = {
        buffering: Math.max(0, Math.min(15, streamingMetrics.buffering + (Math.random() - 0.5) * 1.5)),
        resolution: streamingMetrics.resolution,
        startupDelay: Math.max(0.3, Math.min(8, streamingMetrics.startupDelay + (Math.random() - 0.5) * 0.3)),
        smoothness: Math.max(75, Math.min(100, streamingMetrics.smoothness + (Math.random() - 0.5) * 3)),
        isPlaying: streamingMetrics.isPlaying,
        serverIP: streamingMetrics.serverIP,
        serverName: streamingMetrics.serverName
      };

      setStreamingMetrics(newStreamingMetrics);

      // Calculate AI improvements
      const bufferingImprovement = ((beforeAIMetrics.buffering - newStreamingMetrics.buffering) / beforeAIMetrics.buffering * 100);
      const resolutionImprovement = 50; // Simulated improvement
      const startupImprovement = ((beforeAIMetrics.startupDelay - newStreamingMetrics.startupDelay) / beforeAIMetrics.startupDelay * 100);
      
      const overallImprovement = (bufferingImprovement + resolutionImprovement + startupImprovement) / 3;

      setAiAllocation(prev => ({
        ...prev,
        improvement: overallImprovement / 100,
        serverLoad: Math.max(0.3, Math.min(0.9, prev.serverLoad + (Math.random() - 0.5) * 0.05)),
        networkOptimization: Math.max(0.7, Math.min(1.0, prev.networkOptimization + (Math.random() - 0.3) * 0.05)),
        lastOptimization: new Date()
      }));
    } catch (error) {
      console.warn('Failed to update streaming metrics:', error);
    }
  }, [streamingMetrics, beforeAIMetrics]);

  // Auto-refresh every 2 seconds
  useEffect(() => {
    const interval = setInterval(updateStreamingMetrics, 2000);
    return () => clearInterval(interval);
  }, [updateStreamingMetrics]);

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

  const bufferingImprovement = ((beforeAIMetrics.buffering - streamingMetrics.buffering) / beforeAIMetrics.buffering * 100).toFixed(1);
  const startupImprovement = ((beforeAIMetrics.startupDelay - streamingMetrics.startupDelay) / beforeAIMetrics.startupDelay * 100).toFixed(1);

  return (
    <div id="youtube-demo" className="dashboard-panel-full dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            YouTube Live Demo
          </h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className="flex items-center space-x-1 px-3 py-1 rounded-md bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              {showOverlay ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              <span className="text-sm">{showOverlay ? 'Hide' : 'Show'} Overlay</span>
            </button>
            <button
              onClick={selectRandomVideo}
              className="flex items-center space-x-1 px-3 py-1 rounded-md bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span className="text-sm">New Video</span>
            </button>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* YouTube Video Player */}
        <div className="mb-6">
          <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
            {currentVideo && (
              <iframe
                src={`https://www.youtube.com/embed/${currentVideo.id}?autoplay=1&mute=1&controls=1&modestbranding=1&rel=0`}
                title={currentVideo.title}
                className="w-full h-full"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            )}
            
            {/* Real-time Metrics Overlay */}
            <AnimatePresence>
              {showOverlay && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 text-white min-w-[200px]"
                >
                  <div className="flex items-center space-x-2 mb-3">
                    <Activity className="w-4 h-4 text-green-400" />
                    <span className="text-sm font-medium">Live Metrics</span>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-300">Buffering</span>
                      <span className={`text-sm font-bold ${getBufferingColor(streamingMetrics.buffering)}`}>
                        {streamingMetrics.buffering.toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-300">Resolution</span>
                      <span className={`text-sm font-bold ${getResolutionColor(streamingMetrics.resolution)}`}>
                        {streamingMetrics.resolution}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-300">Startup</span>
                      <span className="text-sm font-bold text-blue-400">
                        {streamingMetrics.startupDelay.toFixed(1)}s
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-300">Smoothness</span>
                      <span className="text-sm font-bold text-green-400">
                        {streamingMetrics.smoothness.toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-300">Server</span>
                      <span className="text-xs text-gray-400">
                        {streamingMetrics.serverName}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          
          {currentVideo && (
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                {currentVideo.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Live streaming with AI-optimized delivery
              </p>
            </div>
          )}
        </div>

        {/* Performance Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Before AI Performance */}
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-3 h-3 bg-red-500 rounded-full"></div>
              <h3 className="text-lg font-semibold text-red-800 dark:text-red-300">Before AI Optimization</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-700 dark:text-red-400">Buffering</span>
                <span className="text-lg font-bold text-red-600">{beforeAIMetrics.buffering.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-700 dark:text-red-400">Resolution</span>
                <span className="text-lg font-bold text-red-600">{beforeAIMetrics.resolution}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-700 dark:text-red-400">Startup Delay</span>
                <span className="text-lg font-bold text-red-600">{beforeAIMetrics.startupDelay.toFixed(1)}s</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-700 dark:text-red-400">Smoothness</span>
                <span className="text-lg font-bold text-red-600">{beforeAIMetrics.smoothness.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-red-700 dark:text-red-400">Server</span>
                <span className="text-sm text-red-600">{beforeAIMetrics.serverName}</span>
              </div>
            </div>
          </div>

          {/* After AI Performance */}
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <h3 className="text-lg font-semibold text-green-800 dark:text-green-300">After AI Optimization</h3>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-green-700 dark:text-green-400">Buffering</span>
                <span className="text-lg font-bold text-green-600">{streamingMetrics.buffering.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-green-700 dark:text-green-400">Resolution</span>
                <span className="text-lg font-bold text-green-600">{streamingMetrics.resolution}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-green-700 dark:text-green-400">Startup Delay</span>
                <span className="text-lg font-bold text-green-600">{streamingMetrics.startupDelay.toFixed(1)}s</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-green-700 dark:text-green-400">Smoothness</span>
                <span className="text-lg font-bold text-green-600">{streamingMetrics.smoothness.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-green-700 dark:text-green-400">Server</span>
                <span className="text-sm text-green-600">{streamingMetrics.serverName}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Improvement Metrics */}
        <div className="mt-6 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Zap className="w-6 h-6 text-blue-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">AI Performance Improvements</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">-{bufferingImprovement}%</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Buffering Reduction</div>
              <div className="flex items-center justify-center space-x-1 mt-1">
                <TrendingDown className="w-4 h-4 text-green-500" />
                <span className="text-xs text-green-600">Better</span>
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">+2 Levels</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Resolution Upgrade</div>
              <div className="flex items-center justify-center space-x-1 mt-1">
                <TrendingUp className="w-4 h-4 text-blue-500" />
                <span className="text-xs text-blue-600">Higher Quality</span>
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">-{startupImprovement}%</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Startup Time</div>
              <div className="flex items-center justify-center space-x-1 mt-1">
                <TrendingDown className="w-4 h-4 text-purple-500" />
                <span className="text-xs text-purple-600">Faster</span>
              </div>
            </div>
          </div>
          
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Server className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Optimized Server: {streamingMetrics.serverName}
              </span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Last optimization: {aiAllocation.lastOptimization.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default YouTubeDemoPanel;
