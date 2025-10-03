import React from 'react';
import { 
  Zap, 
  CheckCircle, 
  Clock, 
  XCircle, 
  AlertTriangle,
  Database,
  Activity,
  RotateCcw,
  Play
} from 'lucide-react';

const ZTAPanel = ({ data }) => {
  const pipeline = data || {};
  const logs = pipeline.execution_logs || [];
  const successRate = pipeline.success_rate || 0;
  const digitalTwinValidation = pipeline.digital_twin_validation || {};

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'running':
        return <Activity className="w-5 h-5 text-blue-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50 dark:bg-green-900 dark:text-green-200';
      case 'running':
        return 'text-blue-600 bg-blue-50 dark:bg-blue-900 dark:text-blue-200';
      case 'failed':
        return 'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200';
      case 'pending':
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900 dark:text-yellow-200';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const pipelineSteps = [
    { name: 'Pending', status: 'completed', icon: Clock },
    { name: 'Validation', status: 'completed', icon: Database },
    { name: 'Deployment', status: 'completed', icon: Play },
    { name: 'Completed', status: 'completed', icon: CheckCircle }
  ];

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Zap className="w-5 h-5 text-yellow-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Zero-Touch Automation
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            {getStatusIcon(pipeline.status)}
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(pipeline.status)}`}>
              {pipeline.status?.toUpperCase() || 'UNKNOWN'}
            </span>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* Deployment Pipeline View */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Deployment Pipeline
          </h3>
          <div className="flex items-center justify-between">
            {pipelineSteps.map((step, index) => (
              <div key={index} className="flex flex-col items-center space-y-2">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  step.status === 'completed' ? 'bg-green-500 text-white' :
                  step.status === 'running' ? 'bg-blue-500 text-white' :
                  'bg-gray-300 text-gray-600'
                }`}>
                  <step.icon className="w-5 h-5" />
                </div>
                <span className="text-xs text-gray-600 dark:text-gray-400">{step.name}</span>
                {index < pipelineSteps.length - 1 && (
                  <div className="absolute left-1/2 top-5 w-full h-0.5 bg-gray-300 dark:bg-gray-600 transform translate-x-1/2"></div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Current Updates in Progress */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Current Updates ({pipeline.updates_count || 0})
          </h3>
          <div className="space-y-2">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Database className="w-4 h-4 text-blue-500" />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    AI Model Update v2.0
                  </span>
                </div>
                <span className="text-xs text-green-600 dark:text-green-400">Completed</span>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="w-4 h-4 text-blue-500" />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    MARL Agent Configuration
                  </span>
                </div>
                <span className="text-xs text-blue-600 dark:text-blue-400">Deploying</span>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-yellow-500" />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    Security Policy Update
                  </span>
                </div>
                <span className="text-xs text-yellow-600 dark:text-yellow-400">Pending</span>
              </div>
            </div>
          </div>
        </div>

        {/* Success vs Failure Rate */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Success vs Failure Rate
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-800 dark:text-green-200">Success Rate</p>
                  <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {successRate.toFixed(1)}%
                  </p>
                </div>
                <CheckCircle className="w-8 h-8 text-green-500" />
              </div>
            </div>
            
            <div className="bg-red-50 dark:bg-red-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-red-800 dark:text-red-200">Failure Rate</p>
                  <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                    {(100 - successRate).toFixed(1)}%
                  </p>
                </div>
                <XCircle className="w-8 h-8 text-red-500" />
              </div>
            </div>
          </div>
        </div>

        {/* Digital Twin Validation Results */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Digital Twin Validation
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                  Passed: {digitalTwinValidation.passed || 0}
                </span>
              </div>
            </div>
            
            <div className="bg-red-50 dark:bg-red-900 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <XCircle className="w-4 h-4 text-red-500" />
                <span className="text-sm font-medium text-red-800 dark:text-red-200">
                  Failed: {digitalTwinValidation.failed || 0}
                </span>
              </div>
            </div>
          </div>
          
          <div className="mt-3 bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Average Validation Latency</span>
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {digitalTwinValidation.avg_latency_ms?.toFixed(1) || 'N/A'} ms
              </span>
            </div>
          </div>
        </div>

        {/* Rollback History Log */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Rollback History
          </h3>
          <div className="space-y-2">
            {logs.map((log, index) => (
              <div key={index} className="flex items-center space-x-3 p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <div className="flex-shrink-0">
                  {log.includes('failed') || log.includes('error') ? (
                    <XCircle className="w-4 h-4 text-red-500" />
                  ) : log.includes('completed') || log.includes('successful') ? (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  ) : (
                    <Activity className="w-4 h-4 text-blue-500" />
                  )}
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">{log}</p>
                </div>
                <div className="flex-shrink-0">
                  <RotateCcw className="w-4 h-4 text-gray-400" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ZTAPanel;

