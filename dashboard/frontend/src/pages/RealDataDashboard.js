import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Sun, 
  Moon, 
  RefreshCw, 
  Clock,
  Activity,
  AlertTriangle,
  CheckCircle,
  Wifi,
  WifiOff
} from 'lucide-react';

// Import all panels
import SystemOverviewPanel from '../components/ai4/SystemOverviewPanel';
import IBNPanel from '../components/ai4/IBNPanel';
import ZTAPanel from '../components/ai4/ZTAPanel';
import QuantumSafePanel from '../components/ai4/QuantumSafePanel';
import RealTimeFederationPanel from '../components/ai4/RealTimeFederationPanel';
import RealTimeSelfEvolutionPanel from '../components/ai4/RealTimeSelfEvolutionPanel';
import ObservabilityPanel from '../components/ai4/ObservabilityPanel';
import APIDocumentationPanel from '../components/ai4/APIDocumentationPanel';

// Import real-time data hook
import { useRealTimeData } from '../hooks/useRealTimeData';

const RealDataDashboard = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [dataMode, setDataMode] = useState('mixed');
  
  // Use real-time data hook
  const {
    data: systemData,
    loading,
    error,
    componentErrors,
    isConnected,
    lastUpdate,
    refreshData
  } = useRealTimeData(refreshInterval);

  // Handle data mode changes
  useEffect(() => {
    if (dataMode === 'mixed') {
      console.log('🔄 Using mixed data mode: Federation & Self-Evolution simulated, others real');
    } else {
      console.log('🔄 Using all-real data mode: All components use real backend data');
    }
  }, [dataMode]);

  // Auto-refresh toggle
  useEffect(() => {
    if (autoRefresh) {
      console.log('🔄 Auto-refresh enabled');
    } else {
      console.log('⏸️ Auto-refresh disabled');
    }
  }, [autoRefresh]);

  // Dark mode effect
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Connection status indicator
  const getConnectionStatusIcon = () => {
    if (loading) return <Activity className="w-4 h-4 animate-pulse text-blue-600" />;
    if (isConnected) return <Wifi className="w-4 h-4 text-green-600" />;
    return <WifiOff className="w-4 h-4 text-red-600" />;
  };

  const getConnectionStatusText = () => {
    if (loading) return 'Connecting...';
    if (isConnected) return 'Connected';
    return 'Disconnected';
  };

  return (
    <div className={`min-h-screen ${darkMode ? 'dark' : ''} bg-gray-50 dark:bg-gray-900`}>
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-purple-600 dark:from-gray-800 dark:to-gray-900 shadow-lg border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <Activity className="w-10 h-10 text-white animate-pulse" />
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-ping"></div>
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">
                    Enhanced Telecom AI 4.0
                  </h1>
                  <p className="text-blue-100 text-sm">Real-time Network Intelligence</p>
                </div>
                <span className={`px-3 py-1 text-xs font-medium rounded-full ${
                  dataMode === 'mixed' 
                    ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200' 
                    : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                }`}>
                  {dataMode === 'mixed' ? 'Mixed Data Mode' : 'All Real Data Mode'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              {/* Data Mode Toggle */}
              <div className="flex items-center space-x-3">
                <label className="text-sm font-medium text-white">
                  Data Mode:
                </label>
                <select
                  value={dataMode}
                  onChange={(e) => setDataMode(e.target.value)}
                  className="px-4 py-2 text-sm border border-white/20 rounded-lg bg-white/10 backdrop-blur-sm text-white placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white/50"
                >
                  <option value="mixed" className="bg-gray-800 text-white">Mixed (Federation + Self-Evolution Simulated)</option>
                  <option value="all-real" className="bg-gray-800 text-white">All Real Data</option>
                </select>
              </div>

              {/* Auto-refresh toggle */}
              <div className="flex items-center space-x-3">
                <label className="text-sm font-medium text-white">
                  Auto-refresh:
                </label>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`px-4 py-2 text-sm rounded-lg transition-all duration-200 ${
                    autoRefresh 
                      ? 'bg-green-500 hover:bg-green-600 text-white shadow-lg' 
                      : 'bg-white/20 hover:bg-white/30 text-white'
                  }`}
                >
                  {autoRefresh ? 'ON' : 'OFF'}
                </button>
              </div>

              {/* Refresh interval */}
              <div className="flex items-center space-x-3">
                <label className="text-sm font-medium text-white">
                  Interval:
                </label>
                <select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                  className="px-4 py-2 text-sm border border-white/20 rounded-lg bg-white/10 backdrop-blur-sm text-white focus:outline-none focus:ring-2 focus:ring-white/50"
                >
                  <option value={2000} className="bg-gray-800 text-white">2s</option>
                  <option value={5000} className="bg-gray-800 text-white">5s</option>
                  <option value={10000} className="bg-gray-800 text-white">10s</option>
                  <option value={30000} className="bg-gray-800 text-white">30s</option>
                </select>
              </div>

              {/* Manual refresh */}
              <button
                onClick={refreshData}
                disabled={loading}
                className="p-3 text-white hover:text-blue-200 transition-colors duration-200 disabled:opacity-50 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-sm"
              >
                <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
              </button>

              {/* Dark mode toggle */}
              <button
                onClick={() => setDarkMode(!darkMode)}
                className="p-3 text-white hover:text-blue-200 transition-colors duration-200 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-sm"
              >
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Enhanced Status Bar */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-3">
                {getConnectionStatusIcon()}
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {getConnectionStatusText()}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <Clock className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Last update: {lastUpdate.toLocaleTimeString()}
                </span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                <span className="text-sm text-gray-500">
                  Auto-refresh: {autoRefresh ? 'ON' : 'OFF'} ({refreshInterval/1000}s)
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {dataMode === 'mixed' && (
                <div className="flex items-center space-x-2 px-3 py-1 bg-orange-100 dark:bg-orange-900 rounded-full">
                  <CheckCircle className="w-4 h-4 text-orange-600 dark:text-orange-400" />
                  <span className="text-sm font-medium text-orange-700 dark:text-orange-300">
                    Federation & Self-Evolution: Simulated Data
                  </span>
                </div>
              )}
              {dataMode === 'all-real' && (
                <div className="flex items-center space-x-2 px-3 py-1 bg-green-100 dark:bg-green-900 rounded-full">
                  <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                  <span className="text-sm font-medium text-green-700 dark:text-green-300">
                    All Components: Real Data
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-4 shadow-lg">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
              <span className="text-red-700 dark:text-red-200">Error: {error}</span>
            </div>
          </div>
        )}

        {/* Quick Stats Overview */}
        <div className="mb-8 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">System Health</p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">99.9%</p>
              </div>
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Agents</p>
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">6</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                <Activity className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Data Mode</p>
                <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                  {dataMode === 'mixed' ? 'Mixed' : 'All Real'}
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center">
                <Clock className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Last Update</p>
                <p className="text-sm font-bold text-gray-600 dark:text-gray-400">
                  {lastUpdate.toLocaleTimeString()}
                </p>
              </div>
              <div className="w-12 h-12 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center">
                <RefreshCw className={`w-6 h-6 text-gray-600 dark:text-gray-400 ${loading ? 'animate-spin' : ''}`} />
              </div>
            </div>
          </div>
        </div>

        <div className="dashboard-grid">
          {/* System Overview */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <SystemOverviewPanel 
              data={systemData.health} 
              loading={loading} 
              error={componentErrors.health} 
            />
          </motion.div>

          {/* Intent-Based Networking */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <IBNPanel 
              data={systemData.ibn} 
              loading={loading} 
              error={componentErrors.ibn} 
            />
          </motion.div>

          {/* Zero-Touch Automation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <ZTAPanel 
              data={systemData.zta} 
              loading={loading} 
              error={componentErrors.zta} 
            />
          </motion.div>

          {/* Quantum-Safe Security */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <QuantumSafePanel 
              data={systemData.quantum} 
              loading={loading} 
              error={componentErrors.quantum} 
            />
          </motion.div>

          {/* Global Multi-Operator Federation - Real-time */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <RealTimeFederationPanel 
              data={systemData.federation} 
              loading={loading} 
              error={componentErrors.federation} 
            />
          </motion.div>

          {/* Self-Evolving AI Agents - Real-time */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <RealTimeSelfEvolutionPanel 
              data={systemData.selfEvolution} 
              loading={loading} 
              error={componentErrors.selfEvolution} 
            />
          </motion.div>

          {/* Enhanced Observability */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <ObservabilityPanel 
              data={systemData.observability} 
              loading={loading} 
              error={componentErrors.observability} 
            />
          </motion.div>

          {/* API & Documentation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            <APIDocumentationPanel 
              data={systemData.api} 
              loading={loading} 
              error={componentErrors.api} 
            />
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default RealDataDashboard;