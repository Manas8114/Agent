import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  EyeIcon,
  ArrowTrendingUpIcon
} from '@heroicons/react/24/outline';

const DataQuality = () => {
  const [qualityData] = useState({
    overallScore: 92.3,
    metrics: [
      {
        name: 'Completeness',
        score: 94.2,
        status: 'good',
        issues: 12
      },
      {
        name: 'Accuracy',
        score: 91.8,
        status: 'good',
        issues: 8
      },
      {
        name: 'Consistency',
        score: 89.5,
        status: 'warning',
        issues: 23
      },
      {
        name: 'Timeliness',
        score: 96.1,
        status: 'good',
        issues: 5
      },
      {
        name: 'Validity',
        score: 88.9,
        status: 'warning',
        issues: 18
      }
    ],
    recentIssues: [
      {
        id: 1,
        type: 'Missing Values',
        severity: 'medium',
        dataset: 'QoS Data',
        count: 45,
        timestamp: '2 hours ago'
      },
      {
        id: 2,
        type: 'Data Format Error',
        severity: 'low',
        dataset: 'Traffic Data',
        count: 12,
        timestamp: '4 hours ago'
      },
      {
        id: 3,
        type: 'Outlier Detection',
        severity: 'high',
        dataset: 'Energy Data',
        count: 8,
        timestamp: '6 hours ago'
      },
      {
        id: 4,
        type: 'Duplicate Records',
        severity: 'low',
        dataset: 'Security Data',
        count: 23,
        timestamp: '8 hours ago'
      }
    ]
  });

  const getStatusColor = (status) => {
    switch (status) {
      case 'good':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'low':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'good':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
      case 'error':
        return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />;
      default:
        return <ClockIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Data Quality Monitoring</h1>
        <p className="text-gray-600">Monitor and ensure the quality of data across all AI agents</p>
      </div>

      {/* Overall Quality Score */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md p-6 mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Overall Data Quality Score</h3>
            <p className="text-sm text-gray-600">Based on completeness, accuracy, consistency, timeliness, and validity</p>
          </div>
          <div className="text-right">
            <div className="text-4xl font-bold text-blue-600">{qualityData.overallScore}%</div>
            <div className="flex items-center text-sm text-green-600">
              <ArrowTrendingUpIcon className="w-4 h-4 mr-1" />
              +2.3% from last week
            </div>
          </div>
        </div>
      </motion.div>

      {/* Quality Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {qualityData.metrics.map((metric, index) => (
          <motion.div
            key={metric.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white rounded-lg shadow-md p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-gray-900">{metric.name}</h4>
              {getStatusIcon(metric.status)}
            </div>
            
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Score</span>
                <span className={`text-lg font-bold ${getStatusColor(metric.status)}`}>
                  {metric.score}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    metric.status === 'good' ? 'bg-green-500' :
                    metric.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${metric.score}%` }}
                ></div>
              </div>
            </div>
            
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Issues Found</span>
              <span className="font-medium text-gray-900">{metric.issues}</span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Recent Issues */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="bg-white rounded-lg shadow-md p-6 mb-8"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Recent Data Quality Issues</h3>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors">
            View All Issues
          </button>
        </div>
        
        <div className="space-y-4">
          {qualityData.recentIssues.map((issue, index) => (
            <motion.div
              key={issue.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  {issue.severity === 'high' ? (
                    <ExclamationTriangleIcon className="w-6 h-6 text-red-500" />
                  ) : issue.severity === 'medium' ? (
                    <ExclamationTriangleIcon className="w-6 h-6 text-yellow-500" />
                  ) : (
                    <CheckCircleIcon className="w-6 h-6 text-green-500" />
                  )}
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900">{issue.type}</h4>
                  <p className="text-sm text-gray-500">Dataset: {issue.dataset}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                  {issue.severity}
                </span>
                <span className="text-sm text-gray-900">{issue.count} records</span>
                <span className="text-sm text-gray-500">{issue.timestamp}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Data Quality Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality Trends</h3>
          <div className="space-y-4">
            {[
              { name: 'Completeness', trend: 'up', change: '+2.1%' },
              { name: 'Accuracy', trend: 'up', change: '+1.8%' },
              { name: 'Consistency', trend: 'down', change: '-0.5%' },
              { name: 'Timeliness', trend: 'up', change: '+3.2%' },
              { name: 'Validity', trend: 'up', change: '+1.4%' }
            ].map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{item.name}</span>
                <div className="flex items-center space-x-2">
                  <ArrowTrendingUpIcon className={`w-4 h-4 ${item.trend === 'up' ? 'text-green-500' : 'text-red-500'}`} />
                  <span className={`text-sm font-medium ${item.trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                    {item.change}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Quality Actions</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <CheckCircleIcon className="w-5 h-5 text-green-600" />
                <span className="text-sm text-gray-900">Data Validation Rules Updated</span>
              </div>
              <span className="text-xs text-gray-500">1 hour ago</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <EyeIcon className="w-5 h-5 text-blue-600" />
                <span className="text-sm text-gray-900">Outlier Detection Enhanced</span>
              </div>
              <span className="text-xs text-gray-500">3 hours ago</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600" />
                <span className="text-sm text-gray-900">Data Cleaning Scheduled</span>
              </div>
              <span className="text-xs text-gray-500">6 hours ago</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DataQuality;
