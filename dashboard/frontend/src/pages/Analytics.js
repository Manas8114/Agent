import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  EyeIcon,
  ClockIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';

const Analytics = () => {
  const [analytics] = useState({
    energySavings: {
      current: 3240,
      previous: 2890,
      trend: 'up'
    },
    costReduction: {
      current: 28.5,
      previous: 24.2,
      trend: 'up'
    },
    serviceQuality: {
      current: 99.8,
      previous: 98.9,
      trend: 'up'
    },
    anomaliesDetected: {
      current: 156,
      previous: 142,
      trend: 'up'
    }
  });

  const [timeSeriesData] = useState([
    { time: '00:00', value: 45 },
    { time: '04:00', value: 38 },
    { time: '08:00', value: 67 },
    { time: '12:00', value: 89 },
    { time: '16:00', value: 92 },
    { time: '20:00', value: 78 },
    { time: '24:00', value: 52 }
  ]);

  const getTrendIcon = (trend) => {
    return trend === 'up' ? (
      <ArrowTrendingUpIcon className="w-5 h-5 text-green-500" />
    ) : (
      <ArrowTrendingDownIcon className="w-5 h-5 text-red-500" />
    );
  };

  const getTrendColor = (trend) => {
    return trend === 'up' ? 'text-green-600' : 'text-red-600';
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Analytics</h1>
        <p className="text-gray-600">Comprehensive analytics and insights from the Enhanced Telecom AI System</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Energy Savings</p>
              <p className="text-2xl font-bold text-gray-900">{analytics.energySavings.current.toLocaleString()} kWh</p>
            </div>
            <div className="flex items-center space-x-2">
              {getTrendIcon(analytics.energySavings.trend)}
              <span className={`text-sm font-medium ${getTrendColor(analytics.energySavings.trend)}`}>
                +{((analytics.energySavings.current - analytics.energySavings.previous) / analytics.energySavings.previous * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Cost Reduction</p>
              <p className="text-2xl font-bold text-gray-900">{analytics.costReduction.current}%</p>
            </div>
            <div className="flex items-center space-x-2">
              {getTrendIcon(analytics.costReduction.trend)}
              <span className={`text-sm font-medium ${getTrendColor(analytics.costReduction.trend)}`}>
                +{((analytics.costReduction.current - analytics.costReduction.previous) / analytics.costReduction.previous * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Service Quality</p>
              <p className="text-2xl font-bold text-gray-900">{analytics.serviceQuality.current}%</p>
            </div>
            <div className="flex items-center space-x-2">
              {getTrendIcon(analytics.serviceQuality.trend)}
              <span className={`text-sm font-medium ${getTrendColor(analytics.serviceQuality.trend)}`}>
                +{((analytics.serviceQuality.current - analytics.serviceQuality.previous) / analytics.serviceQuality.previous * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Anomalies Detected</p>
              <p className="text-2xl font-bold text-gray-900">{analytics.anomaliesDetected.current}</p>
            </div>
            <div className="flex items-center space-x-2">
              {getTrendIcon(analytics.anomaliesDetected.trend)}
              <span className={`text-sm font-medium ${getTrendColor(analytics.anomaliesDetected.trend)}`}>
                +{((analytics.anomaliesDetected.current - analytics.anomaliesDetected.previous) / analytics.anomaliesDetected.previous * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Performance Over Time</h3>
            <ChartBarIcon className="w-6 h-6 text-blue-600" />
          </div>
          <div className="h-64 flex items-end space-x-2">
            {timeSeriesData.map((point, index) => (
              <div key={index} className="flex-1 flex flex-col items-center">
                <div
                  className="bg-blue-500 rounded-t"
                  style={{ height: `${point.value}%`, minHeight: '20px' }}
                ></div>
                <span className="text-xs text-gray-500 mt-2">{point.time}</span>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">AI Agent Performance</h3>
            <EyeIcon className="w-6 h-6 text-green-600" />
          </div>
          <div className="space-y-4">
            {[
              { name: 'QoS Anomaly', accuracy: 94.2, color: 'bg-blue-500' },
              { name: 'Failure Prediction', accuracy: 91.8, color: 'bg-green-500' },
              { name: 'Traffic Forecast', accuracy: 89.5, color: 'bg-yellow-500' },
              { name: 'Energy Optimize', accuracy: 96.1, color: 'bg-purple-500' },
              { name: 'Security Detection', accuracy: 92.7, color: 'bg-red-500' },
              { name: 'Data Quality', accuracy: 88.9, color: 'bg-indigo-500' }
            ].map((agent, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{agent.name}</span>
                <div className="flex items-center space-x-3">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${agent.color}`}
                      style={{ width: `${agent.accuracy}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12 text-right">{agent.accuracy}%</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Business Impact */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white rounded-lg shadow-md p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Business Impact Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <CurrencyDollarIcon className="w-8 h-8 text-green-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">$2.4M</div>
            <div className="text-sm text-gray-600">Cost Savings</div>
          </div>
          <div className="text-center">
            <ClockIcon className="w-8 h-8 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">99.8%</div>
            <div className="text-sm text-gray-600">Uptime</div>
          </div>
          <div className="text-center">
            <ArrowTrendingUpIcon className="w-8 h-8 text-purple-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">+28.5%</div>
            <div className="text-sm text-gray-600">Efficiency Gain</div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Analytics;
