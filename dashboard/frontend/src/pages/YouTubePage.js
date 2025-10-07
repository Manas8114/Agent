import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Volume2, 
  VolumeX, 
  Settings, 
  RefreshCw,
  Eye,
  EyeOff,
  Wifi,
  WifiOff,
  Monitor,
  Clock,
  Server,
  Globe
} from 'lucide-react';

// Import the YouTube Demo Panel component
import YouTubeDemoPanel from '../components/ai4/YouTubeDemoPanel';

const YouTubePage = () => {
  const [systemData, setSystemData] = useState({
    health: {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      metrics: {
        buffering: 2.1,
        resolution: '1080p',
        startup_delay: 1.8,
        smoothness: 98.5,
        server: 'Server-East-1',
        ip: '192.168.1.45'
      }
    }
  });

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showMetrics, setShowMetrics] = useState(true);

  const refreshData = () => {
    // Simulate data refresh
    setSystemData(prev => ({
      ...prev,
      health: {
        ...prev.health,
        timestamp: new Date().toISOString(),
        metrics: {
          ...prev.health.metrics,
          buffering: Math.random() * 5,
          startup_delay: Math.random() * 3,
          smoothness: 95 + Math.random() * 5
        }
      }
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Play className="w-8 h-8 text-red-500" />
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  YouTube Streaming Demo
                </h1>
              </div>
              <div className="hidden md:flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Live Streaming</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Wifi className="w-4 h-4" />
                  <span>AI-Optimized</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowMetrics(!showMetrics)}
                className="flex items-center px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                {showMetrics ? <EyeOff className="w-4 h-4 mr-2" /> : <Eye className="w-4 h-4 mr-2" />}
                {showMetrics ? 'Hide' : 'Show'} Metrics
              </button>
              
              <button
                onClick={refreshData}
                className="flex items-center px-3 py-2 text-sm bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </button>
              
              <button
                onClick={() => setIsFullscreen(!isFullscreen)}
                className="flex items-center px-3 py-2 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
              >
                <Monitor className="w-4 h-4 mr-2" />
                {isFullscreen ? 'Exit' : 'Fullscreen'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Status Banner */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 bg-gradient-to-r from-red-50 to-blue-50 dark:from-red-900/20 dark:to-blue-900/20 rounded-lg p-6 border border-red-200 dark:border-red-800"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Play className="w-6 h-6 text-red-500" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  AI-Enhanced YouTube Streaming
                </h2>
              </div>
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-gray-600 dark:text-gray-400">Live</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Server className="w-4 h-4 text-blue-500" />
                  <span className="text-gray-600 dark:text-gray-400">{systemData.health.metrics.server}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Globe className="w-4 h-4 text-green-500" />
                  <span className="text-gray-600 dark:text-gray-400">{systemData.health.metrics.ip}</span>
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <div className="text-2xl font-bold text-green-600">
                {systemData.health.metrics.smoothness.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Stream Quality</div>
            </div>
          </div>
        </motion.div>

        {/* YouTube Demo Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={`${isFullscreen ? 'fixed inset-0 z-50 bg-black' : ''}`}
        >
          <YouTubeDemoPanel
            data={systemData.health}
            onRefresh={refreshData}
            showMetrics={showMetrics}
            isFullscreen={isFullscreen}
          />
        </motion.div>

        {/* Performance Metrics */}
        {showMetrics && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {/* Buffering Performance */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Buffering Performance
                </h3>
                <div className={`w-3 h-3 rounded-full ${
                  systemData.health.metrics.buffering < 3 ? 'bg-green-500' : 
                  systemData.health.metrics.buffering < 5 ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.health.metrics.buffering.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Time spent loading
              </div>
            </div>

            {/* Resolution Quality */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Resolution Quality
                </h3>
                <Monitor className="w-5 h-5 text-blue-500" />
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.health.metrics.resolution}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Current playback quality
              </div>
            </div>

            {/* Startup Delay */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Startup Delay
                </h3>
                <Clock className="w-5 h-5 text-purple-500" />
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.health.metrics.startup_delay.toFixed(1)}s
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Time to start playback
              </div>
            </div>

            {/* Server Allocation */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Server Allocation
                </h3>
                <Server className="w-5 h-5 text-green-500" />
              </div>
              <div className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                {systemData.health.metrics.server}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                {systemData.health.metrics.ip}
              </div>
            </div>
          </motion.div>
        )}

        {/* AI Enhancement Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-8 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            🚀 AI Enhancement Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Before AI Optimization:
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Higher buffering rates (8-15%)</li>
                <li>• Lower resolution (480p-720p)</li>
                <li>• Longer startup delays (5-10s)</li>
                <li>• Inconsistent server allocation</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                After AI Optimization:
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Reduced buffering (1-3%)</li>
                <li>• Stable high resolution (1080p-4K)</li>
                <li>• Faster startup (1-3s)</li>
                <li>• Optimal server allocation</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default YouTubePage;




