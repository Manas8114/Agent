import React from 'react';
import { motion } from 'framer-motion';

const StatusCard = ({ title, value, status, icon: Icon, trend, subtitle }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
      case 'good':
      case 'success':
        return 'text-green-600';
      case 'warning':
      case 'caution':
        return 'text-yellow-600';
      case 'error':
      case 'danger':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusBgColor = (status) => {
    switch (status) {
      case 'healthy':
      case 'good':
      case 'success':
        return 'bg-green-100';
      case 'warning':
      case 'caution':
        return 'bg-yellow-100';
      case 'error':
      case 'danger':
        return 'bg-red-100';
      default:
        return 'bg-gray-100';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          {Icon && <Icon className="w-6 h-6 text-blue-600" />}
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        {status && (
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBgColor(status)} ${getStatusColor(status)}`}>
            {status}
          </span>
        )}
      </div>
      
      <div className="mb-2">
        <div className="text-2xl font-bold text-gray-900">{value}</div>
        {subtitle && <div className="text-sm text-gray-500">{subtitle}</div>}
      </div>
      
      {trend && (
        <div className="flex items-center text-sm">
          <span className={`font-medium ${trend.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
            {trend.direction === 'up' ? '↗' : '↘'} {trend.value}
          </span>
          <span className="text-gray-500 ml-1">{trend.label}</span>
        </div>
      )}
    </motion.div>
  );
};

export default StatusCard;
