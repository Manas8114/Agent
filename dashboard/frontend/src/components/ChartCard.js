import React from 'react';
import { motion } from 'framer-motion';

const ChartCard = ({ title, children, icon: Icon, className = '' }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        {Icon && <Icon className="w-6 h-6 text-blue-600" />}
      </div>
      
      <div className="h-64">
        {children}
      </div>
    </motion.div>
  );
};

export default ChartCard;
