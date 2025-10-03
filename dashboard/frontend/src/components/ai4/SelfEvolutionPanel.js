import React from 'react';
import { 
  Brain, 
  TrendingUp, 
  BarChart3,
  Cpu,
  Battery,
  Clock,
  AlertTriangle
} from 'lucide-react';

const SelfEvolutionPanel = ({ data, loading = false, error = null }) => {
  const evolution = data || {};
  const activeTasks = evolution.active_tasks || [];
  const kpiImprovements = evolution.kpi_improvements || {};
  const realTimeMetrics = evolution.real_time_metrics || {};

  const getTaskStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'text-blue-600 bg-blue-50 dark:bg-blue-900 dark:text-blue-200';
      case 'completed':
        return 'text-green-600 bg-green-50 dark:bg-green-900 dark:text-green-200';
      case 'failed':
        return 'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getImprovementColor = (value) => {
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  if (loading) {
    return (
      <div className="dashboard-panel dark:bg-gray-800">
        <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-purple-500" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Self-Evolving AI Agents
              </h2>
            </div>
          </div>
        </div>
        <div className="dashboard-panel-content p-4 lg:p-6">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mx-auto mb-4"></div>
              <p className="text-gray-600 dark:text-gray-400">Loading evolution data...</p>
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
              <Brain className="w-5 h-5 text-purple-500" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Self-Evolving AI Agents
              </h2>
            </div>
          </div>
        </div>
        <div className="dashboard-panel-content p-4 lg:p-6">
          <div className="bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-4">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
              <span className="text-red-700 dark:text-red-200">Failed to load evolution data</span>
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
            <Brain className="w-5 h-5 text-purple-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Self-Evolving AI Agents
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              evolution.evolution_status === 'evolving' 
                ? 'text-blue-600 bg-blue-50 dark:bg-blue-900 dark:text-blue-200'
                : 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200'
            }`}>
              {evolution.evolution_status?.toUpperCase() || 'UNKNOWN'}
            </span>
            <div className={`w-3 h-3 rounded-full ${
              evolution.evolution_status === 'evolving' ? 'bg-blue-500' : 'bg-gray-500'
            }`}></div>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* Active Evolution Tasks */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Active Evolution Tasks ({activeTasks.length})
          </h3>
          <div className="space-y-3 max-h-48 overflow-y-auto">
            {activeTasks.length > 0 ? (
              activeTasks.map((task, index) => (
                <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center">
                        <Brain className="w-4 h-4 text-purple-500" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {task.type} - {task.agent_id}
                        </p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Target Accuracy: {(task.target_accuracy * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTaskStatusColor(task.status)}`}>
                      {task.status}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2 mr-3">
                      <div 
                        className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${task.progress}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-gray-600 dark:text-gray-400">
                      {task.progress.toFixed(1)}%
                    </span>
                  </div>
                  {task.duration_minutes && (
                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                      Duration: {task.duration_minutes} minutes
                    </p>
                  )}
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <Brain className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No active evolution tasks</p>
              </div>
            )}
          </div>
        </div>

        {/* KPI Improvement Over Time */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            KPI Improvement Over Time
          </h3>
          <div className="grid grid-cols-1 gap-4">
            {Object.entries(kpiImprovements).map(([metric, data]) => (
              <div key={metric} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {metric === 'latency_ms' && <Clock className="w-4 h-4 text-blue-500" />}
                    {metric === 'throughput_mbps' && <TrendingUp className="w-4 h-4 text-green-500" />}
                    {metric === 'energy_efficiency' && <Battery className="w-4 h-4 text-yellow-500" />}
                    {metric === 'accuracy' && <BarChart3 className="w-4 h-4 text-purple-500" />}
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                        {metric.replace('_', ' ')}
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        Baseline: {data.baseline?.toFixed(2) || 'N/A'} | 
                        Current: {data.current?.toFixed(2) || 'N/A'}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`text-lg font-semibold ${getImprovementColor(data.improvement_percent || 0)}`}>
                      {data.improvement_percent > 0 ? '+' : ''}{data.improvement_percent?.toFixed(1) || 0}%
                    </span>
                    <p className="text-xs text-gray-500 dark:text-gray-500">
                      Confidence: {((data.confidence || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Real-time Metrics */}
        {realTimeMetrics && Object.keys(realTimeMetrics).length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              Real-time Performance Metrics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <Clock className="w-4 h-4 text-blue-500" />
                  <span className="text-xs text-blue-600 dark:text-blue-400">Latency</span>
                </div>
                <div className="text-lg font-semibold text-blue-700 dark:text-blue-200">
                  {realTimeMetrics.latency_ms?.toFixed(1) || 'N/A'} ms
                </div>
              </div>
              <div className="bg-green-50 dark:bg-green-900 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-green-500" />
                  <span className="text-xs text-green-600 dark:text-green-400">Throughput</span>
                </div>
                <div className="text-lg font-semibold text-green-700 dark:text-green-200">
                  {realTimeMetrics.throughput_mbps?.toFixed(1) || 'N/A'} Mbps
                </div>
              </div>
              <div className="bg-yellow-50 dark:bg-yellow-900 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <Battery className="w-4 h-4 text-yellow-500" />
                  <span className="text-xs text-yellow-600 dark:text-yellow-400">Energy Efficiency</span>
                </div>
                <div className="text-lg font-semibold text-yellow-700 dark:text-yellow-200">
                  {realTimeMetrics.energy_efficiency?.toFixed(3) || 'N/A'}
                </div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900 rounded-lg p-3">
                <div className="flex items-center space-x-2">
                  <Cpu className="w-4 h-4 text-purple-500" />
                  <span className="text-xs text-purple-600 dark:text-purple-400">Active Agents</span>
                </div>
                <div className="text-lg font-semibold text-purple-700 dark:text-purple-200">
                  {realTimeMetrics.active_agents || 0}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Evolution Summary */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Evolution Summary
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">Evolution Round</div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {evolution.evolution_round || 0}
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-500 dark:text-gray-400">Architecture Improvement</div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {((evolution.architecture_improvement || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SelfEvolutionPanel;