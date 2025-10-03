import React from 'react';
import { 
  Globe, 
  Users, 
  TrendingUp, 
  XCircle, 
  Activity,
  AlertTriangle,
  Clock
} from 'lucide-react';

const FederationPanel = ({ data, loading = false, error = null }) => {
  const federation = data || {};
  const operators = federation.operators || [];
  const cooperationEvents = federation.cooperation_events || [];
  const updateMetrics = federation.update_metrics || {};

  const getOperatorStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-50 dark:bg-green-900 dark:text-green-200';
      case 'inactive':
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900 dark:text-yellow-200';
      case 'disconnected':
        return 'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getEventTypeIcon = (type) => {
    switch (type) {
      case 'traffic_spike':
        return <TrendingUp className="w-4 h-4 text-orange-500" />;
      case 'network_failure':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'load_balancing':
        return <Activity className="w-4 h-4 text-blue-500" />;
      case 'security_incident':
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
      default:
        return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  if (loading) {
    return (
      <div className="dashboard-panel dark:bg-gray-800">
        <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Globe className="w-5 h-5 text-blue-500" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Global Multi-Operator Federation
              </h2>
            </div>
          </div>
        </div>
        <div className="dashboard-panel-content p-4 lg:p-6">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600 dark:text-gray-400">Loading federation data...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-panel dark:bg-gray-800">
        <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Globe className="w-5 h-5 text-blue-500" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Global Multi-Operator Federation
              </h2>
            </div>
          </div>
        </div>
        <div className="dashboard-panel-content p-4 lg:p-6">
          <div className="bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-4">
            <div className="flex items-center">
              <XCircle className="w-5 h-5 text-red-500 mr-2" />
              <span className="text-red-700 dark:text-red-200">Failed to load federation data</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Globe className="w-5 h-5 text-blue-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Global Multi-Operator Federation
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {federation.active_nodes || 0}/{federation.total_nodes || 0} Active
            </div>
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* Federation Metrics */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Federation Metrics
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">Model Updates Shared</div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {federation.updates_shared || 0}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">Model Aggregations</div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {federation.aggregations_total || 0}
              </div>
            </div>
          </div>
        </div>

        {/* Model Performance */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Model Performance Across Operators
          </h3>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Average Model Accuracy</span>
              <span className="text-lg font-semibold text-gray-900 dark:text-white">
                {((federation.avg_model_accuracy || 0) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Active Operator Nodes */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Active Operator Nodes ({operators.length})
          </h3>
          <div className="space-y-3 max-h-48 overflow-y-auto">
            {operators.length > 0 ? (
              operators.map((operator, index) => (
                <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                        <Users className="w-5 h-5 text-blue-500" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {operator.name}
                        </p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Accuracy: {((operator.accuracy || 0) * 100).toFixed(1)}% | 
                          Location: {operator.location || 'Unknown'}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-500">
                          Data Samples: {operator.data_samples || 0} | 
                          Participation: {((operator.participation_score || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getOperatorStatusColor(operator.status)}`}>
                        {operator.status}
                      </span>
                      <div className={`w-2 h-2 rounded-full ${
                        operator.status === 'active' ? 'bg-green-500' : 'bg-red-500'
                      }`}></div>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <Users className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No operator data available</p>
              </div>
            )}
          </div>
        </div>

        {/* Cooperation Events */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Cooperation Events ({cooperationEvents.length})
          </h3>
          <div className="space-y-3 max-h-48 overflow-y-auto">
            {cooperationEvents.length > 0 ? (
              cooperationEvents.map((event, index) => (
                <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {getEventTypeIcon(event.type)}
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                          {event.type.replace('_', ' ')}
                        </p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Participants: {event.participants?.join(', ') || 'Unknown'}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-500">
                          Impact: {((event.impact_score || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        event.status === 'resolved' ? 'text-green-600 bg-green-50 dark:bg-green-900 dark:text-green-200' :
                        event.status === 'active' ? 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900 dark:text-yellow-200' :
                        'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200'
                      }`}>
                        {event.status}
                      </span>
                      <Clock className="w-4 h-4 text-gray-400" />
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No cooperation events</p>
              </div>
            )}
          </div>
        </div>

        {/* Cross-Operator Secure Updates */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Cross-Operator Secure Updates
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-3">
              <div className="text-xs text-green-600 dark:text-green-400">Successful Updates</div>
              <div className="text-lg font-semibold text-green-700 dark:text-green-200">
                {updateMetrics.successful_updates || 0}
              </div>
            </div>
            <div className="bg-red-50 dark:bg-red-900 rounded-lg p-3">
              <div className="text-xs text-red-600 dark:text-red-400">Failed Updates</div>
              <div className="text-lg font-semibold text-red-700 dark:text-red-200">
                {updateMetrics.failed_updates || 0}
              </div>
            </div>
          </div>
          {updateMetrics.success_rate && (
            <div className="mt-3 text-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Success Rate: {((updateMetrics.success_rate || 0) * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FederationPanel;