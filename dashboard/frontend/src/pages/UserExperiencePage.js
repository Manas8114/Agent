import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Download,
  Moon,
  Sun,
  Clock,
  RefreshCw
} from 'lucide-react';

// Components
import UserExperiencePanel from '../components/ai4/UserExperiencePanel';

// Services
import { fetchAI4Data } from '../services/ai4Service';

const UserExperiencePage = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(2000);
  const [systemData, setSystemData] = useState({
    health: null,
    kpis: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Auto-refresh functionality
  const refreshData = useCallback(async () => {
    try {
      setError(null);
      
      // Try to fetch real data
      let realData = {};
      try {
        realData = await fetchAI4Data();
      } catch (error) {
        console.warn('Using fallback data:', error.message);
        // Use simulated data if real data is not available
        realData = {
          health: { status: 'healthy', uptime: 86400 },
          kpis: {
            latency_ms: 45,
            throughput_mbps: 125,
            signal_strength: -65,
            connection_quality: 95,
            user_count: 1250,
            data_volume_gb: 2.5
          }
        };
      }

      setSystemData(realData);
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Failed to refresh data:', err);
      setError(err.message);
    }
  }, []);

  // Initial data load
  useEffect(() => {
    const loadInitialData = async () => {
      setLoading(true);
      try {
        await refreshData();
      } catch (err) {
        console.error('Failed to load initial data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, [refreshData]);

  // Auto-refresh setup
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(refreshData, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, refreshData]);

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  // Export functionality
  const exportReport = (format) => {
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `user-experience-report-${timestamp}.${format}`;
    
    if (format === 'pdf') {
      console.log(`Exporting PDF report: ${filename}`);
    } else if (format === 'csv') {
      console.log(`Exporting CSV report: ${filename}`);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading User Experience Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${darkMode ? 'dark' : ''}`}>
      <div className="bg-gray-50 dark:bg-gray-900 min-h-screen flex flex-col">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <div className="px-4 lg:px-6 py-4">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
              <div className="flex items-center space-x-4">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  User Experience Dashboard
                </h1>
                <div className="flex items-center space-x-2">
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    systemData.health?.status === 'healthy' 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : systemData.health?.status === 'warning'
                      ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {systemData.health?.status === 'healthy' ? (
                      <><CheckCircle className="w-3 h-3 inline mr-1" /> Healthy</>
                    ) : systemData.health?.status === 'warning' ? (
                      <><AlertTriangle className="w-3 h-3 inline mr-1" /> Warning</>
                    ) : (
                      <><XCircle className="w-3 h-3 inline mr-1" /> Critical</>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col lg:flex-row lg:items-center space-y-2 lg:space-y-0 lg:space-x-4">
                {/* Auto-refresh toggle */}
                <div className="flex items-center space-x-2">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={autoRefresh}
                      onChange={(e) => setAutoRefresh(e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Auto-refresh</span>
                  </label>
                  <select
                    value={refreshInterval}
                    onChange={(e) => setRefreshInterval(Number(e.target.value))}
                    className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  >
                    <option value={1000}>1s</option>
                    <option value={2000}>2s</option>
                    <option value={5000}>5s</option>
                    <option value={10000}>10s</option>
                  </select>
                </div>

                {/* Manual refresh */}
                <button
                  onClick={refreshData}
                  className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  <RefreshCw className="w-5 h-5" />
                </button>

                {/* Dark mode toggle */}
                <button
                  onClick={toggleDarkMode}
                  className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                </button>

                {/* Export buttons */}
                <div className="flex space-x-2">
                  <button
                    onClick={() => exportReport('pdf')}
                    className="flex items-center px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    PDF
                  </button>
                  <button
                    onClick={() => exportReport('csv')}
                    className="flex items-center px-3 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    CSV
                  </button>
                </div>

                {/* Last update */}
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  <Clock className="w-4 h-4 inline mr-1" />
                  {lastUpdate.toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900 border-l-4 border-red-400 p-4">
            <div className="flex">
              <XCircle className="w-5 h-5 text-red-400" />
              <div className="ml-3">
                <p className="text-sm text-red-700 dark:text-red-200">
                  Error loading data: {error}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Main content */}
        <div className="flex-1 p-4 lg:p-6 space-y-4 lg:space-y-6 overflow-y-auto">
          {/* User Experience Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <UserExperiencePanel 
              data={systemData.health}
              onRefresh={refreshData}
            />
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default UserExperiencePage;
