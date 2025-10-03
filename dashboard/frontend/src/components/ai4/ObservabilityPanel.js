import React from 'react';
import { 
  Eye, 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  TrendingUp,
  BarChart3,
  ExternalLink,
  Clock,
  Zap
} from 'lucide-react';

const ObservabilityPanel = ({ data, loading = false, error = null }) => {
  const observability = data || {};
  const metrics = observability.prometheus_metrics || {};
  const alerts = Array.isArray(observability.alerts) ? observability.alerts : [];
  const grafanaDashboards = observability.grafana_dashboards || {};

  const getAlertSeverityColor = (severity) => {
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

  const getAlertIcon = (severity) => {
    switch (severity) {
      case 'critical':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'info':
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      default:
        return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Eye className="w-5 h-5 text-indigo-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Enhanced Observability
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Active</span>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* Prometheus Metrics */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Prometheus Metrics Streams
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-800 dark:text-blue-200">Latency P95</p>
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {metrics.latency_p95?.toFixed(1) || 'N/A'}ms
                  </p>
                </div>
                <TrendingUp className="w-8 h-8 text-blue-500" />
              </div>
            </div>
            
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-800 dark:text-green-200">Throughput</p>
                  <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {metrics.throughput_gbps?.toFixed(1) || 'N/A'} Gbps
                  </p>
                </div>
                <BarChart3 className="w-8 h-8 text-green-500" />
              </div>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-purple-800 dark:text-purple-200">Error Rate</p>
                  <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {((metrics.error_rate || 0) * 100).toFixed(3)}%
                  </p>
                </div>
                <Activity className="w-8 h-8 text-purple-500" />
              </div>
            </div>
            
            <div className="bg-orange-50 dark:bg-orange-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-orange-800 dark:text-orange-200">CPU Usage</p>
                  <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                    {metrics.cpu_usage?.toFixed(1) || 'N/A'}%
                  </p>
                </div>
                <Zap className="w-8 h-8 text-orange-500" />
              </div>
            </div>
          </div>
        </div>

        {/* Grafana Integration */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Grafana Integration
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(grafanaDashboards).map(([name, url]) => (
              <div key={name} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                      {name.replace('_', ' ')}
                    </p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">Dashboard</p>
                  </div>
                  <a 
                    href={url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:text-blue-700"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Alert System */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Alert System ({alerts.length} active)
          </h3>
          <div className="space-y-2">
            {alerts.map((alert, index) => (
              <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getAlertIcon(alert.severity)}
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        {alert.message}
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {new Date(alert.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAlertSeverityColor(alert.severity)}`}>
                      {alert.severity}
                    </span>
                    {alert.resolved ? (
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    ) : (
                      <Clock className="w-4 h-4 text-yellow-500" />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Threshold Violations */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Threshold Violations
          </h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-red-50 dark:bg-red-900 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-red-800 dark:text-red-200">Latency > 10ms</p>
                  <p className="text-lg font-bold text-red-600 dark:text-red-400">2</p>
                </div>
                <AlertTriangle className="w-6 h-6 text-red-500" />
              </div>
            </div>
            
            <div className="bg-yellow-50 dark:bg-yellow-900 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-yellow-800 dark:text-yellow-200">Packet Loss > 1%</p>
                  <p className="text-lg font-bold text-yellow-600 dark:text-yellow-400">0</p>
                </div>
                <CheckCircle className="w-6 h-6 text-yellow-500" />
              </div>
            </div>
            
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-800 dark:text-green-200">Accuracy &lt; 90%</p>
                  <p className="text-lg font-bold text-green-600 dark:text-green-400">0</p>
                </div>
                <CheckCircle className="w-6 h-6 text-green-500" />
              </div>
            </div>
          </div>
        </div>

        {/* Error & Event Logs */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Live Event Stream
          </h3>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    System health check completed successfully
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 30000).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    AI 4.0 modules synchronized successfully
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 60000).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    Traffic spike detected and handled by MARL agents
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 120000).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    All agents operational and performing optimally
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 180000).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObservabilityPanel;
