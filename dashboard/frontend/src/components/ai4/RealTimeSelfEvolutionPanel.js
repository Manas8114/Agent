import React from 'react';
import { 
  Brain, 
  TrendingUp, 
  Activity, 
  AlertCircle,
  Loader2,
  CheckCircle,
  XCircle,
  Target,
  Zap
} from 'lucide-react';

const RealTimeSelfEvolutionPanel = ({ data, loading = false, error = null }) => {
  // Handle loading state
  if (loading) {
    return (
      <div className="dashboard-panel dark:bg-gray-800">
        <div className="dashboard-panel-content">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-600" />
              Self-Evolving AI Agents
            </h3>
            <div className="flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-purple-600" />
              <span className="text-sm text-gray-500">Loading...</span>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="animate-pulse">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
            </div>
            <div className="animate-pulse">
              <div className="h-20 bg-gray-200 dark:bg-gray-700 rounded"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Handle error state
  if (error) {
    return (
      <div className="dashboard-panel dark:bg-gray-800">
        <div className="dashboard-panel-content">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-600" />
              Self-Evolving AI Agents
            </h3>
            <div className="flex items-center space-x-2 text-red-600">
              <XCircle className="w-4 h-4" />
              <span className="text-sm">Connection Error</span>
            </div>
          </div>
          
          <div className="flex items-center justify-center h-32 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
            <div className="text-center">
              <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
              <p className="text-red-700 dark:text-red-300 font-medium">Failed to load evolution data</p>
              <p className="text-red-600 dark:text-red-400 text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Handle no data state
  if (!data) {
    return (
      <div className="dashboard-panel dark:bg-gray-800">
        <div className="dashboard-panel-content">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-600" />
              Self-Evolving AI Agents
            </h3>
            <div className="flex items-center space-x-2 text-yellow-600">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">Awaiting Data</span>
            </div>
          </div>
          
          <div className="flex items-center justify-center h-32 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <div className="text-center">
              <Activity className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
              <p className="text-yellow-700 dark:text-yellow-300 font-medium">No evolution data available</p>
              <p className="text-yellow-600 dark:text-yellow-400 text-sm mt-1">Waiting for backend connection...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render real data
  const {
    agent_id = "multi_agent_system",
    evolution_round = 0,
    architecture_improvement = 0,
    hyperparameter_optimization = {},
    performance_improvement = 0,
    evolution_status = "idle",
    active_tasks = [],
    kpi_improvements = {},
    real_time_metrics = {}
  } = data;

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'evolving': return 'text-green-600 dark:text-green-400';
      case 'idle': return 'text-gray-600 dark:text-gray-400';
      case 'error': return 'text-red-600 dark:text-red-400';
      default: return 'text-blue-600 dark:text-blue-400';
    }
  };

  const getStatusIcon = (status) => {
    switch (status.toLowerCase()) {
      case 'evolving': return <Activity className="w-4 h-4" />;
      case 'idle': return <Target className="w-4 h-4" />;
      case 'error': return <XCircle className="w-4 h-4" />;
      default: return <Brain className="w-4 h-4" />;
    }
  };

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="dashboard-panel-content">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
            <Brain className="w-5 h-5 mr-2 text-purple-600" />
            Self-Evolving AI Agents
          </h3>
          <div className="flex items-center space-x-2 text-green-600">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm">Live Data</span>
          </div>
        </div>

        {/* Evolution Status */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Agent ID</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">{agent_id}</p>
          </div>
          <div className="text-right">
            <div className={`flex items-center space-x-2 ${getStatusColor(evolution_status)}`}>
              {getStatusIcon(evolution_status)}
              <span className="font-medium">{evolution_status.toUpperCase()}</span>
            </div>
            <p className="text-sm text-gray-500">Round {evolution_round}</p>
          </div>
        </div>

        {/* KPI Improvements */}
        {Object.keys(kpi_improvements).length > 0 && (
          <div className="mb-6">
            <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3">KPI Improvement Over Time</h4>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(kpi_improvements).map(([metric, data]) => (
                <div key={metric} className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                        {metric.replace('_', ' ')}
                      </p>
                      <p className="text-lg font-bold text-gray-900 dark:text-white">
                        {data.improvement_percent > 0 ? '+' : ''}{data.improvement_percent.toFixed(1)}%
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-gray-500">Current: {data.current}</p>
                      <p className="text-xs text-gray-500">Baseline: {data.baseline}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Real-time Metrics */}
        {Object.keys(real_time_metrics).length > 0 && (
          <div className="mb-6">
            <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3">Real-Time Metrics</h4>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(real_time_metrics).map(([metric, value]) => (
                <div key={metric} className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                  <p className="text-sm text-blue-600 dark:text-blue-400 capitalize">
                    {metric.replace('_', ' ')}
                  </p>
                  <p className="text-xl font-bold text-blue-900 dark:text-blue-100">
                    {typeof value === 'number' ? value.toFixed(2) : value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Hyperparameter Optimization */}
        {Object.keys(hyperparameter_optimization).length > 0 && (
          <div className="mb-6">
            <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3">Hyperparameter Optimization</h4>
            <div className="space-y-2">
              {Object.entries(hyperparameter_optimization).map(([param, value]) => (
                <div key={param} className="flex items-center justify-between bg-gray-50 dark:bg-gray-700 p-2 rounded">
                  <span className="text-sm text-gray-900 dark:text-white capitalize">
                    {param.replace('_', ' ')}
                  </span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Active Tasks */}
        {active_tasks.length > 0 && (
          <div className="mb-6">
            <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3">Active Evolution Tasks</h4>
            <div className="space-y-2">
              {active_tasks.slice(0, 3).map((task, index) => (
                <div key={index} className="flex items-center justify-between bg-green-50 dark:bg-green-900/20 p-2 rounded">
                  <span className="text-sm text-green-900 dark:text-green-100">
                    {task.name || `Task ${index + 1}`}
                  </span>
                  <span className="text-xs text-green-600 dark:text-green-400">Running</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Performance Summary */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg text-center">
            <Zap className="w-6 h-6 text-purple-600 mx-auto mb-1" />
            <p className="text-sm text-purple-600 dark:text-purple-400">Architecture</p>
            <p className="text-lg font-bold text-purple-900 dark:text-purple-100">
              +{(architecture_improvement * 100).toFixed(1)}%
            </p>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg text-center">
            <TrendingUp className="w-6 h-6 text-green-600 mx-auto mb-1" />
            <p className="text-sm text-green-600 dark:text-green-400">Performance</p>
            <p className="text-lg font-bold text-green-900 dark:text-green-100">
              +{(performance_improvement * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeSelfEvolutionPanel;
