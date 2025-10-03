import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Download,
  Moon,
  Sun,
  Clock
} from 'lucide-react';

// AI 4.0 Components
import SystemOverviewPanel from '../components/ai4/SystemOverviewPanel';
import IBNPanel from '../components/ai4/IBNPanel';
import ZTAPanel from '../components/ai4/ZTAPanel';
import QuantumSafePanel from '../components/ai4/QuantumSafePanel';
import FederationPanel from '../components/ai4/FederationPanel';
import SelfEvolutionPanel from '../components/ai4/SelfEvolutionPanel';
import ObservabilityPanel from '../components/ai4/ObservabilityPanel';
import APIDocumentationPanel from '../components/ai4/APIDocumentationPanel';

// Services
import { fetchAI4Data } from '../services/ai4Service';

const AI4Dashboard = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [systemData, setSystemData] = useState({
    health: null,
    kpis: null,
    ibn: null,
    zta: null,
    quantum: null,
    federation: null,
    selfEvolution: null,
    observability: null,
    api: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [componentErrors] = useState({});
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Auto-refresh functionality
  const refreshData = useCallback(async () => {
    try {
      setError(null);
      
      // Create simulated data for Federation and Self-Evolution
      const simulatedFederationData = {
        total_nodes: 5,
        active_nodes: 4,
        updates_shared: 12,
        aggregations_total: 8,
        avg_model_accuracy: 0.913,
        cooperative_scenarios_handled: 3,
        operators: [
          { name: "Operator Alpha", status: "active", region: "North America" },
          { name: "Operator Beta", status: "active", region: "Europe" },
          { name: "Operator Gamma", status: "active", region: "Asia Pacific" },
          { name: "Operator Delta", status: "active", region: "South America" }
        ],
        cooperation_events: [
          { id: 1, type: "Traffic Spike", description: "Coordinated load balancing across regions", timestamp: new Date().toISOString() },
          { id: 2, type: "Model Update", description: "Shared federated learning model", timestamp: new Date().toISOString() },
          { id: 3, type: "Failure Recovery", description: "Cross-operator backup activation", timestamp: new Date().toISOString() }
        ],
        update_metrics: {
          successful_updates: 10,
          failed_updates: 2,
          success_rate: 0.833
        }
      };

      const simulatedSelfEvolutionData = {
        agent_id: "multi_agent_system",
        evolution_round: 12,
        architecture_improvement: 0.15,
        hyperparameter_optimization: {
          learning_rate: 0.0012,
          batch_size: 128,
          hidden_layers: 4,
          dropout_rate: 0.3
        },
        performance_improvement: 0.22,
        evolution_status: "evolving",
        active_tasks: [
          { id: 1, name: "Neural Architecture Search", status: "running", progress: 75 },
          { id: 2, name: "Hyperparameter Optimization", status: "running", progress: 60 },
          { id: 3, name: "Performance Evaluation", status: "pending", progress: 0 }
        ],
        kpi_improvements: {
          latency_ms: { baseline: 50.0, current: 42.5, improvement_percent: 15.0, confidence: 0.95 },
          throughput_mbps: { baseline: 100.0, current: 125.0, improvement_percent: 25.0, confidence: 0.92 },
          energy_efficiency: { baseline: 0.8, current: 0.92, improvement_percent: 15.0, confidence: 0.88 },
          accuracy: { baseline: 0.85, current: 0.91, improvement_percent: 7.1, confidence: 0.97 }
        },
        real_time_metrics: {
          latency_ms: 42.5,
          throughput_mbps: 125.0,
          energy_efficiency: 0.92,
          active_agents: 6
        }
      };

      // Try to fetch real data for other components
      let realData = {};
      try {
        realData = await fetchAI4Data();
      } catch (error) {
        console.warn('Using fallback data for some components:', error.message);
      }

      // Combine real data with simulated data
      setSystemData({
        ...realData,
        federation: simulatedFederationData,
        selfEvolution: simulatedSelfEvolutionData
      });
      
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
    const filename = `telecom-ai4-report-${timestamp}.${format}`;
    
    if (format === 'pdf') {
      // PDF export logic would go here
      console.log(`Exporting PDF report: ${filename}`);
    } else if (format === 'csv') {
      // CSV export logic would go here
      console.log(`Exporting CSV report: ${filename}`);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading AI 4.0 Dashboard...</p>
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
                  Telecom AI 4.0 Dashboard
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
                    <option value={5000}>5s</option>
                    <option value={10000}>10s</option>
                    <option value={30000}>30s</option>
                    <option value={60000}>1m</option>
                  </select>
                </div>

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
          {/* System Overview Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <SystemOverviewPanel 
              data={systemData.health}
              kpis={systemData.kpis}
              onRefresh={refreshData}
            />
          </motion.div>

          {/* AI 4.0 Feature Panels */}
          <div className="dashboard-grid">
            {/* Intent-Based Networking */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <IBNPanel data={systemData.ibn} />
            </motion.div>

            {/* Zero-Touch Automation */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <ZTAPanel data={systemData.zta} />
            </motion.div>

            {/* Quantum-Safe Security */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <QuantumSafePanel data={systemData.quantum} />
            </motion.div>

            {/* Global Multi-Operator Federation */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <FederationPanel 
                data={systemData.federation} 
                loading={!systemData.federation && loading}
                error={componentErrors.federation}
              />
            </motion.div>

            {/* Self-Evolving AI Agents */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <SelfEvolutionPanel 
                data={systemData.selfEvolution} 
                loading={!systemData.selfEvolution && loading}
                error={componentErrors.selfEvolution}
              />
            </motion.div>

            {/* Enhanced Observability */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              <ObservabilityPanel data={systemData.observability} />
            </motion.div>
          </div>

          {/* API & Documentation Panel - Full width */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <APIDocumentationPanel data={systemData.api} />
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default AI4Dashboard;
