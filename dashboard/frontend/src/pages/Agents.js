import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CpuChipIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';

const Agents = () => {
  const [agents] = useState([
    {
      id: 'qos_anomaly',
      name: 'QoS Anomaly Detection',
      status: 'healthy',
      accuracy: 94.2,
      predictions: 1247,
      lastUpdate: '2 minutes ago',
      description: 'Detects anomalies in Quality of Service metrics'
    },
    {
      id: 'failure_prediction',
      name: 'Failure Prediction',
      status: 'healthy',
      accuracy: 91.8,
      predictions: 892,
      lastUpdate: '1 minute ago',
      description: 'Predicts equipment failures before they occur'
    },
    {
      id: 'traffic_forecast',
      name: 'Traffic Forecasting',
      status: 'healthy',
      accuracy: 89.5,
      predictions: 2156,
      lastUpdate: '3 minutes ago',
      description: 'Forecasts network traffic patterns'
    },
    {
      id: 'energy_optimize',
      name: 'Energy Optimization',
      status: 'healthy',
      accuracy: 96.1,
      predictions: 743,
      lastUpdate: '1 minute ago',
      description: 'Optimizes energy consumption in telecom networks'
    },
    {
      id: 'security_detection',
      name: 'Security Detection',
      status: 'healthy',
      accuracy: 92.7,
      predictions: 1834,
      lastUpdate: '2 minutes ago',
      description: 'Detects security threats and intrusions'
    },
    {
      id: 'data_quality',
      name: 'Data Quality Monitoring',
      status: 'healthy',
      accuracy: 88.9,
      predictions: 967,
      lastUpdate: '4 minutes ago',
      description: 'Monitors and ensures data quality'
    }
  ]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
      case 'error':
        return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />;
      default:
        return <ClockIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-100 text-green-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Agents</h1>
        <p className="text-gray-600">Monitor and manage all AI agents in the Enhanced Telecom AI System</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent, index) => (
          <motion.div
            key={agent.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <CpuChipIcon className="w-8 h-8 text-blue-600" />
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{agent.name}</h3>
                  <p className="text-sm text-gray-500">{agent.description}</p>
                </div>
              </div>
              {getStatusIcon(agent.status)}
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Status</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
                  {agent.status}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Accuracy</span>
                <div className="flex items-center space-x-2">
                  <ArrowTrendingUpIcon className="w-4 h-4 text-green-500" />
                  <span className="text-sm font-medium text-gray-900">{agent.accuracy}%</span>
                </div>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Predictions Today</span>
                <span className="text-sm font-medium text-gray-900">{agent.predictions.toLocaleString()}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Last Update</span>
                <span className="text-sm text-gray-500">{agent.lastUpdate}</span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="flex space-x-2">
                <button className="flex-1 bg-blue-600 text-white px-3 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors">
                  View Details
                </button>
                <button className="flex-1 bg-gray-100 text-gray-700 px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors">
                  Configure
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-8 bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Agent Performance Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600">6</div>
            <div className="text-sm text-gray-600">Active Agents</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">92.3%</div>
            <div className="text-sm text-gray-600">Average Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600">7,839</div>
            <div className="text-sm text-gray-600">Total Predictions</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Agents;
