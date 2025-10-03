import React from 'react';
import { 
  Globe, 
  CheckCircle, 
  XCircle, 
  Clock,
  TrendingUp,
  Activity
} from 'lucide-react';

const IBNPanel = ({ data, loading = false, error = null }) => {
  const intents = Array.isArray(data?.active_intents) ? data.active_intents : [];
  const violations = Array.isArray(data?.violations) ? data.violations : [];
  const complianceRate = data?.compliance_rate || 0;

  const getIntentStatusIcon = (status) => {
    switch (status) {
      case 'enforced':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'active':
        return <Activity className="w-4 h-4 text-blue-500" />;
      case 'violated':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getViolationSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900 dark:text-yellow-200';
      case 'info':
        return 'text-blue-600 bg-blue-50 dark:bg-blue-900 dark:text-blue-200';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Globe className="w-5 h-5 text-blue-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Intent-Based Networking
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Compliance: {complianceRate.toFixed(1)}%
            </div>
            <div className={`w-3 h-3 rounded-full ${
              complianceRate >= 95 ? 'bg-green-500' : 
              complianceRate >= 85 ? 'bg-yellow-500' : 'bg-red-500'
            }`}></div>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* Active Intents */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Active Intents ({intents.length})
          </h3>
          <div className="space-y-3">
            {intents.map((intent, index) => (
              <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      {getIntentStatusIcon(intent.status)}
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {intent.description}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-xs text-gray-600 dark:text-gray-400">
                      <span>Compliance: {intent.compliance?.toFixed(1)}%</span>
                      <span>Created: {new Date(intent.created_at).toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Violations Detected */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Violations Detected ({violations.length})
          </h3>
          <div className="space-y-2">
            {violations.map((violation, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className={`w-2 h-2 rounded-full ${
                  violation.severity === 'critical' ? 'bg-red-500' :
                  violation.severity === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                }`}></div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">{violation.details}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(violation.timestamp).toLocaleString()}
                  </p>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getViolationSeverityColor(violation.severity)}`}>
                  {violation.severity}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Network Heatmap Simulation */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Network Performance Heatmap
          </h3>
          <div className="grid grid-cols-4 gap-2">
            {Array.from({ length: 16 }, (_, i) => {
              const latency = Math.random() * 50 + 10;
              const intensity = Math.min(latency / 60, 1);
              return (
                <div
                  key={i}
                  className="aspect-square rounded flex items-center justify-center text-xs font-medium"
                  style={{
                    backgroundColor: `rgba(59, 130, 246, ${intensity})`,
                    color: intensity > 0.5 ? 'white' : 'black'
                  }}
                >
                  {latency.toFixed(0)}ms
                </div>
              );
            })}
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
            <span>Low Latency</span>
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-blue-100 rounded"></div>
              <div className="w-3 h-3 bg-blue-300 rounded"></div>
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <div className="w-3 h-3 bg-blue-700 rounded"></div>
            </div>
            <span>High Latency</span>
          </div>
        </div>

        {/* Intent Translation Status */}
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Intent Translation Status
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span className="text-sm font-medium text-green-800 dark:text-green-200">
                  QoS Rules Applied
                </span>
              </div>
              <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                {data?.enforced_intents || 0} of {data?.total_intents || 0} intents
              </p>
            </div>
            
            <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-blue-500" />
                <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                  MARL Constraints
                </span>
              </div>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                Active optimization policies
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IBNPanel;

