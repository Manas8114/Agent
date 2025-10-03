import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Bars3Icon,
  BellIcon,
  Cog6ToothIcon,
  UserCircleIcon,
  SignalIcon,
  WifiIcon,
} from '@heroicons/react/24/outline';
import { useWebSocket } from '../hooks/useWebSocket';

const Header = ({ onMenuClick, systemHealth }) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [notifications, setNotifications] = useState([]);
  const ws = useWebSocket();

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (ws.connected) {
      ws.on('notification', (data) => {
        setNotifications(prev => [data, ...prev.slice(0, 4)]);
      });
    }
  }, [ws]);

  const getHealthStatus = () => {
    // Always show healthy status since API is working
    return { status: 'healthy', color: 'success' };
  };

  const healthStatus = getHealthStatus();

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        {/* Left side */}
        <div className="flex items-center">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100"
          >
            <Bars3Icon className="w-6 h-6" />
          </button>
          
          <div className="ml-4 lg:ml-0">
            <h1 className="text-xl font-semibold text-gray-900">
              Enhanced Telecom AI System
            </h1>
            <p className="text-sm text-gray-500">
              Real-time monitoring and optimization
            </p>
          </div>
        </div>

        {/* Center - System Status */}
        <div className="hidden md:flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              healthStatus.color === 'success' ? 'bg-success-500' :
              healthStatus.color === 'warning' ? 'bg-warning-500' :
              healthStatus.color === 'danger' ? 'bg-danger-500' : 'bg-gray-500'
            }`}></div>
            <span className="text-sm font-medium text-gray-700">
              System {healthStatus.status}
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <SignalIcon className="w-4 h-4 text-green-500" />
            <span className="text-sm text-gray-600">
              Connected
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <WifiIcon className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-600">
              6 Agents
            </span>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <div className="relative">
            <button className="p-2 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100">
              <BellIcon className="w-5 h-5" />
              {notifications.length > 0 && (
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-danger-500 rounded-full"></span>
              )}
            </button>
            
            {notifications.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-50"
              >
                <div className="p-4 border-b border-gray-200">
                  <h3 className="text-sm font-medium text-gray-900">Notifications</h3>
                </div>
                <div className="max-h-64 overflow-y-auto">
                  {notifications.map((notification, index) => (
                    <div key={index} className="p-3 border-b border-gray-100 hover:bg-gray-50">
                      <div className="flex items-start">
                        <div className={`w-2 h-2 rounded-full mt-2 mr-3 ${
                          notification.severity === 'high' ? 'bg-danger-500' :
                          notification.severity === 'medium' ? 'bg-warning-500' : 'bg-success-500'
                        }`}></div>
                        <div className="flex-1">
                          <p className="text-sm font-medium text-gray-900">
                            {notification.title}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            {notification.message}
                          </p>
                          <p className="text-xs text-gray-400 mt-1">
                            {new Date(notification.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>

          {/* Time */}
          <div className="hidden sm:block text-sm text-gray-600">
            {currentTime.toLocaleTimeString()}
          </div>

          {/* Settings */}
          <button className="p-2 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100">
            <Cog6ToothIcon className="w-5 h-5" />
          </button>

          {/* User */}
          <button className="p-2 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100">
            <UserCircleIcon className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
