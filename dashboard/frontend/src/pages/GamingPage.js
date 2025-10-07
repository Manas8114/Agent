import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Gamepad2, 
  Zap, 
  Wifi, 
  WifiOff, 
  Monitor, 
  Clock,
  Server,
  Globe,
  RefreshCw,
  Eye,
  EyeOff,
  Activity,
  TrendingUp,
  TrendingDown
} from 'lucide-react';

// Import the User Experience Panel component
import UserExperiencePanel from '../components/ai4/UserExperiencePanel';

const GamingPage = () => {
  const [systemData, setSystemData] = useState({
    health: {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      metrics: {
        fps: 144,
        ping: 12,
        jitter: 1.2,
        packet_loss: 0.05,
        server: 'Gaming-Server-West',
        ip: '192.168.1.100'
      }
    }
  });

  const [showMetrics, setShowMetrics] = useState(true);
  const [isGamingMode, setIsGamingMode] = useState(true);

  const refreshData = () => {
    // Simulate realistic gaming data
    setSystemData(prev => ({
      ...prev,
      health: {
        ...prev.health,
        timestamp: new Date().toISOString(),
        metrics: {
          ...prev.health.metrics,
          fps: 120 + Math.random() * 50, // 120-170 FPS
          ping: 8 + Math.random() * 20, // 8-28ms
          jitter: 0.5 + Math.random() * 2, // 0.5-2.5ms
          packet_loss: Math.random() * 0.1 // 0-0.1%
        }
      }
    }));
  };

  useEffect(() => {
    // Auto-refresh data every 2 seconds in gaming mode
    if (isGamingMode) {
      const interval = setInterval(refreshData, 2000);
      return () => clearInterval(interval);
    }
  }, [isGamingMode]);

  const getFPSColor = (fps) => {
    if (fps >= 120) return 'text-green-500';
    if (fps >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getPingColor = (ping) => {
    if (ping <= 20) return 'text-green-500';
    if (ping <= 50) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Gamepad2 className="w-8 h-8 text-blue-500" />
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Gaming Performance Monitor
                </h1>
              </div>
              <div className="hidden md:flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                <div className="flex items-center space-x-1">
                  <div className={`w-2 h-2 rounded-full ${isGamingMode ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
                  <span>{isGamingMode ? 'Gaming Mode' : 'Monitoring Mode'}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Zap className="w-4 h-4" />
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
                onClick={() => setIsGamingMode(!isGamingMode)}
                className={`flex items-center px-3 py-2 text-sm rounded-lg transition-colors ${
                  isGamingMode 
                    ? 'bg-green-600 text-white hover:bg-green-700' 
                    : 'bg-gray-600 text-white hover:bg-gray-700'
                }`}
              >
                <Gamepad2 className="w-4 h-4 mr-2" />
                {isGamingMode ? 'Gaming Mode' : 'Monitor Mode'}
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
          className="mb-8 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Gamepad2 className="w-6 h-6 text-blue-500" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  AI-Enhanced Gaming Performance
                </h2>
              </div>
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-gray-600 dark:text-gray-400">Connected</span>
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
              <div className={`text-2xl font-bold ${getFPSColor(systemData.health.metrics.fps)}`}>
                {Math.round(systemData.health.metrics.fps)} FPS
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Current Performance</div>
            </div>
          </div>
        </motion.div>

        {/* User Experience Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <UserExperiencePanel
            data={systemData.health}
            onRefresh={refreshData}
            showMetrics={showMetrics}
            isGamingMode={isGamingMode}
          />
        </motion.div>

        {/* Real-time Gaming Metrics */}
        {showMetrics && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {/* FPS Counter */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  FPS Counter
                </h3>
                <div className={`w-3 h-3 rounded-full ${
                  systemData.health.metrics.fps >= 120 ? 'bg-green-500' : 
                  systemData.health.metrics.fps >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
              </div>
              <motion.div 
                className={`text-4xl font-bold mb-2 ${getFPSColor(systemData.health.metrics.fps)}`}
                key={systemData.health.metrics.fps}
                initial={{ scale: 1.1 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.2 }}
              >
                {Math.round(systemData.health.metrics.fps)}
              </motion.div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Frames per second
              </div>
            </div>

            {/* Ping/Latency */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Ping/Latency
                </h3>
                <Wifi className="w-5 h-5 text-blue-500" />
              </div>
              <motion.div 
                className={`text-4xl font-bold mb-2 ${getPingColor(systemData.health.metrics.ping)}`}
                key={systemData.health.metrics.ping}
                initial={{ scale: 1.1 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.2 }}
              >
                {Math.round(systemData.health.metrics.ping)}ms
              </motion.div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Network latency
              </div>
            </div>

            {/* Jitter */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Jitter
                </h3>
                <Activity className="w-5 h-5 text-purple-500" />
              </div>
              <div className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.health.metrics.jitter.toFixed(1)}ms
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Network stability
              </div>
            </div>

            {/* Packet Loss */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Packet Loss
                </h3>
                <div className={`w-3 h-3 rounded-full ${
                  systemData.health.metrics.packet_loss < 0.1 ? 'bg-green-500' : 
                  systemData.health.metrics.packet_loss < 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
              </div>
              <div className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.health.metrics.packet_loss.toFixed(2)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Data loss rate
              </div>
            </div>
          </motion.div>
        )}

        {/* Performance Trends */}
        {showMetrics && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* Before AI vs After AI */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Performance Comparison
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <span className="text-sm font-medium text-gray-900 dark:text-white">Before AI</span>
                  <div className="flex items-center space-x-2">
                    <TrendingDown className="w-4 h-4 text-red-500" />
                    <span className="text-sm text-red-600 dark:text-red-400">60-80 FPS, 50-100ms ping</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <span className="text-sm font-medium text-gray-900 dark:text-white">After AI</span>
                  <div className="flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-green-600 dark:text-green-400">120-170 FPS, 8-28ms ping</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Server Allocation */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Server Allocation
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Current Server</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {systemData.health.metrics.server}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">IP Address</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {systemData.health.metrics.ip}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Connection Status</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-green-600 dark:text-green-400">Optimal</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* AI Enhancement Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-8 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg p-6 border border-blue-200 dark:border-blue-800"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            🎮 AI Gaming Enhancement Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Before AI Optimization:
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Lower FPS (60-80)</li>
                <li>• Higher ping (50-100ms)</li>
                <li>• Network jitter and packet loss</li>
                <li>• Suboptimal server allocation</li>
                <li>• Inconsistent performance</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                After AI Optimization:
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• High FPS (120-170)</li>
                <li>• Low ping (8-28ms)</li>
                <li>• Stable network connection</li>
                <li>• Optimal server allocation</li>
                <li>• Consistent performance</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default GamingPage;




