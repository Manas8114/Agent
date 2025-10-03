import React from 'react';
import { 
  Activity, 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  TrendingUp,
  Users,
  Database,
  Zap,
  Globe,
  Brain,
  Eye,
  Shield
} from 'lucide-react';

const SystemOverviewPanel = ({ data, kpis, onRefresh }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'critical':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Activity className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-50 dark:bg-green-900 dark:text-green-200';
      case 'warning':
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900 dark:text-yellow-200';
      case 'critical':
        return 'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const modules = [
    { name: 'IBN', icon: Globe, status: 'healthy', description: 'Intent-Based Networking' },
    { name: 'ZTA', icon: Zap, status: 'healthy', description: 'Zero-Touch Automation' },
    { name: 'Quantum-Safe', icon: Shield, status: 'healthy', description: 'Quantum-Safe Security' },
    { name: 'Federation', icon: Globe, status: 'healthy', description: 'Global Federation' },
    { name: 'Self-Evolution', icon: Brain, status: 'healthy', description: 'Self-Evolving Agents' },
    { name: 'Observability', icon: Eye, status: 'healthy', description: 'Enhanced Observability' }
  ];

  return (
    <div className="dashboard-panel-full dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            System Overview
          </h2>
          <div className="flex items-center space-x-2">
            {getStatusIcon(data?.status)}
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(data?.status)}`}>
              {data?.status?.toUpperCase() || 'UNKNOWN'}
            </span>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* Global System Health */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Global System Health
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div className="flex items-center">
                <Activity className="w-5 h-5 text-blue-500 mr-2" />
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Uptime</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {data?.uptime ? `${Math.floor(data.uptime / 3600)}h ${Math.floor((data.uptime % 3600) / 60)}m` : 'N/A'}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div className="flex items-center">
                <Users className="w-5 h-5 text-green-500 mr-2" />
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Active Users</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {kpis?.user_count?.toLocaleString() || 'N/A'}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div className="flex items-center">
                <TrendingUp className="w-5 h-5 text-purple-500 mr-2" />
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Throughput</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {kpis?.throughput_mbps ? `${kpis.throughput_mbps.toFixed(1)} Mbps` : 'N/A'}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <div className="flex items-center">
                <Database className="w-5 h-5 text-orange-500 mr-2" />
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Data Volume</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {kpis?.data_volume_gb ? `${kpis.data_volume_gb.toFixed(1)} GB` : 'N/A'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Active Modules Status */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Active Modules Status
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {modules.map((module, index) => (
              <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <module.icon className="w-5 h-5 text-blue-500" />
                  <CheckCircle className="w-4 h-4 text-green-500" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">{module.name}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">{module.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Current Traffic Load */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Current Traffic Load
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-4 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">Latency</p>
                  <p className="text-2xl font-bold">{kpis?.latency_ms?.toFixed(1) || 'N/A'} ms</p>
                </div>
                <Activity className="w-8 h-8 opacity-80" />
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-lg p-4 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">Connection Quality</p>
                  <p className="text-2xl font-bold">{kpis?.connection_quality?.toFixed(1) || 'N/A'}%</p>
                </div>
                <CheckCircle className="w-8 h-8 opacity-80" />
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-4 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">Signal Strength</p>
                  <p className="text-2xl font-bold">{kpis?.signal_strength?.toFixed(1) || 'N/A'} dBm</p>
                </div>
                <TrendingUp className="w-8 h-8 opacity-80" />
              </div>
            </div>
          </div>
        </div>

        {/* Recent Events Timeline */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Recent Events Timeline
          </h3>
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm text-gray-900 dark:text-white">System health check passed</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">2 minutes ago</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm text-gray-900 dark:text-white">AI 4.0 modules synchronized</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">5 minutes ago</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm text-gray-900 dark:text-white">Traffic spike detected and handled</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">12 minutes ago</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm text-gray-900 dark:text-white">All agents operational</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">15 minutes ago</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemOverviewPanel;

