import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  EyeIcon,
  LockClosedIcon
} from '@heroicons/react/24/outline';

const Security = () => {
  const [securityData] = useState({
    threats: [
      {
        id: 1,
        type: 'DDoS Attack',
        severity: 'high',
        source: '192.168.1.100',
        timestamp: '2 minutes ago',
        status: 'blocked'
      },
      {
        id: 2,
        type: 'Unauthorized Access',
        severity: 'medium',
        source: '10.0.0.50',
        timestamp: '15 minutes ago',
        status: 'investigating'
      },
      {
        id: 3,
        type: 'Malware Detection',
        severity: 'high',
        source: '172.16.0.25',
        timestamp: '1 hour ago',
        status: 'contained'
      },
      {
        id: 4,
        type: 'Suspicious Traffic',
        severity: 'low',
        source: '203.0.113.45',
        timestamp: '2 hours ago',
        status: 'monitoring'
      }
    ],
    stats: {
      totalThreats: 23,
      blockedThreats: 18,
      activeThreats: 2,
      securityScore: 94.2
    }
  });

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

  const getStatusColor = (status) => {
    switch (status) {
      case 'blocked':
        return 'bg-green-100 text-green-800';
      case 'contained':
        return 'bg-blue-100 text-blue-800';
      case 'investigating':
        return 'bg-yellow-100 text-yellow-800';
      case 'monitoring':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Security Dashboard</h1>
        <p className="text-gray-600">Monitor security threats and protect your telecom infrastructure</p>
      </div>

      {/* Security Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center">
            <ShieldCheckIcon className="w-8 h-8 text-green-600 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Security Score</p>
              <p className="text-2xl font-bold text-gray-900">{securityData.stats.securityScore}%</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center">
            <ExclamationTriangleIcon className="w-8 h-8 text-red-600 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Total Threats</p>
              <p className="text-2xl font-bold text-gray-900">{securityData.stats.totalThreats}</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center">
            <LockClosedIcon className="w-8 h-8 text-blue-600 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Blocked Threats</p>
              <p className="text-2xl font-bold text-gray-900">{securityData.stats.blockedThreats}</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center">
            <EyeIcon className="w-8 h-8 text-yellow-600 mr-3" />
            <div>
              <p className="text-sm text-gray-600">Active Threats</p>
              <p className="text-2xl font-bold text-gray-900">{securityData.stats.activeThreats}</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Recent Threats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-white rounded-lg shadow-md p-6 mb-8"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Recent Security Threats</h3>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors">
            View All
          </button>
        </div>
        
        <div className="space-y-4">
          {securityData.threats.map((threat, index) => (
            <motion.div
              key={threat.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  {threat.severity === 'high' ? (
                    <ExclamationTriangleIcon className="w-6 h-6 text-red-500" />
                  ) : threat.severity === 'medium' ? (
                    <ExclamationTriangleIcon className="w-6 h-6 text-yellow-500" />
                  ) : (
                    <ShieldCheckIcon className="w-6 h-6 text-green-500" />
                  )}
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-900">{threat.type}</h4>
                  <p className="text-sm text-gray-500">Source: {threat.source}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(threat.severity)}`}>
                  {threat.severity}
                </span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(threat.status)}`}>
                  {threat.status}
                </span>
                <span className="text-sm text-gray-500">{threat.timestamp}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Security Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Threat Distribution</h3>
          <div className="space-y-3">
            {[
              { type: 'DDoS Attacks', count: 8, percentage: 35 },
              { type: 'Malware', count: 6, percentage: 26 },
              { type: 'Unauthorized Access', count: 5, percentage: 22 },
              { type: 'Suspicious Traffic', count: 4, percentage: 17 }
            ].map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{item.type}</span>
                <div className="flex items-center space-x-3">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${item.percentage}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-8 text-right">{item.count}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Security Actions</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <ShieldCheckIcon className="w-5 h-5 text-green-600" />
                <span className="text-sm text-gray-900">Firewall Rules Updated</span>
              </div>
              <span className="text-xs text-gray-500">5 min ago</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <LockClosedIcon className="w-5 h-5 text-blue-600" />
                <span className="text-sm text-gray-900">Access Control Modified</span>
              </div>
              <span className="text-xs text-gray-500">12 min ago</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <EyeIcon className="w-5 h-5 text-yellow-600" />
                <span className="text-sm text-gray-900">Monitoring Enhanced</span>
              </div>
              <span className="text-xs text-gray-500">1 hour ago</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Security;
