import React from 'react';
import { 
  Globe, 
  Users, 
  TrendingUp, 
  Activity, 
  AlertCircle,
  Loader2,
  CheckCircle,
  XCircle
} from 'lucide-react';

const RealTimeFederationPanel = ({ data, loading = false, error = null }) => {
  // Handle loading state
  if (loading) {
    return (
      <div className="dashboard-panel dark:bg-gray-800">
        <div className="dashboard-panel-content">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
              <Globe className="w-5 h-5 mr-2 text-blue-600" />
              Global Multi-Operator Federation
            </h3>
            <div className="flex items-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
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
              <Globe className="w-5 h-5 mr-2 text-blue-600" />
              Global Multi-Operator Federation
            </h3>
            <div className="flex items-center space-x-2 text-red-600">
              <XCircle className="w-4 h-4" />
              <span className="text-sm">Connection Error</span>
            </div>
          </div>
          
          <div className="flex items-center justify-center h-32 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
            <div className="text-center">
              <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-2" />
              <p className="text-red-700 dark:text-red-300 font-medium">Failed to load federation data</p>
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
              <Globe className="w-5 h-5 mr-2 text-blue-600" />
              Global Multi-Operator Federation
            </h3>
            <div className="flex items-center space-x-2 text-yellow-600">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">Awaiting Data</span>
            </div>
          </div>
          
          <div className="flex items-center justify-center h-32 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <div className="text-center">
              <Activity className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
              <p className="text-yellow-700 dark:text-yellow-300 font-medium">No federation data available</p>
              <p className="text-yellow-600 dark:text-yellow-400 text-sm mt-1">Waiting for backend connection...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render real data
  const {
    total_nodes = 0,
    active_nodes = 0,
    updates_shared = 0,
    aggregations_total = 0,
    avg_model_accuracy = 0,
    cooperative_scenarios_handled = 0,
    operators = [],
    cooperation_events = [],
    update_metrics = {}
  } = data;

  const successRate = update_metrics.success_rate || 0;
  const successfulUpdates = update_metrics.successful_updates || 0;
  const failedUpdates = update_metrics.failed_updates || 0;

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="dashboard-panel-content">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
            <Globe className="w-5 h-5 mr-2 text-blue-600" />
            Global Multi-Operator Federation
          </h3>
          <div className="flex items-center space-x-2 text-green-600">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm">Live Data</span>
          </div>
        </div>

        {/* Federation Status */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Active Nodes</p>
                <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                  {active_nodes}/{total_nodes}
                </p>
              </div>
              <Users className="w-8 h-8 text-blue-600" />
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-green-600 dark:text-green-400">Model Accuracy</p>
                <p className="text-2xl font-bold text-green-900 dark:text-green-100">
                  {(avg_model_accuracy * 100).toFixed(1)}%
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
          </div>
        </div>

        {/* Federation Metrics */}
        <div className="space-y-4">
          <h4 className="text-md font-semibold text-gray-900 dark:text-white">Federation Metrics</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Model Updates Shared</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">{updates_shared}</p>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-400">Model Aggregations</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">{aggregations_total}</p>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">Cooperation Events</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">{cooperative_scenarios_handled}</p>
          </div>
        </div>

        {/* Update Metrics */}
        {Object.keys(update_metrics).length > 0 && (
          <div className="mt-6">
            <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3">Update Performance</h4>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg text-center">
                <p className="text-sm text-green-600 dark:text-green-400">Successful</p>
                <p className="text-lg font-bold text-green-900 dark:text-green-100">{successfulUpdates}</p>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg text-center">
                <p className="text-sm text-red-600 dark:text-red-400">Failed</p>
                <p className="text-lg font-bold text-red-900 dark:text-red-100">{failedUpdates}</p>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg text-center">
                <p className="text-sm text-blue-600 dark:text-blue-400">Success Rate</p>
                <p className="text-lg font-bold text-blue-900 dark:text-blue-100">{(successRate * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
        )}

        {/* Active Operators */}
        {operators.length > 0 && (
          <div className="mt-6">
            <h4 className="text-md font-semibold text-gray-900 dark:text-white mb-3">Active Operators</h4>
            <div className="space-y-2">
              {operators.slice(0, 3).map((operator, index) => (
                <div key={index} className="flex items-center justify-between bg-gray-50 dark:bg-gray-700 p-2 rounded">
                  <span className="text-sm text-gray-900 dark:text-white">{operator.name || `Operator ${index + 1}`}</span>
                  <span className="text-xs text-green-600 dark:text-green-400">Active</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RealTimeFederationPanel;
