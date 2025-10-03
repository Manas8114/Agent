import React from 'react';
import { motion } from 'framer-motion';

const MetricCard = ({ 
  title, 
  value, 
  change, 
  changeType = 'neutral', 
  icon: Icon, 
  color = 'primary',
  className = '' 
}) => {
  const colorClasses = {
    primary: 'text-primary-600',
    success: 'text-success-600',
    warning: 'text-warning-600',
    danger: 'text-danger-600',
  };

  const changeColorClasses = {
    positive: 'text-success-600 bg-success-50',
    negative: 'text-danger-600 bg-danger-50',
    neutral: 'text-gray-600 bg-gray-50',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      className={`metric-card ${className}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className={`text-2xl font-bold ${colorClasses[color]}`}>
            {value}
          </p>
          {change && (
            <div className="mt-2">
              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${changeColorClasses[changeType]}`}>
                {change}
              </span>
            </div>
          )}
        </div>
        
        {Icon && (
          <div className={`p-3 rounded-lg bg-gray-50 ${colorClasses[color]}`}>
            <Icon className="w-6 h-6" />
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default MetricCard;
