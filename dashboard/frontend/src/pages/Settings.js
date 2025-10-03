import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BellIcon,
  ShieldCheckIcon,
  ServerIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

const Settings = () => {
  const [settings, setSettings] = useState({
    notifications: {
      email: true,
      sms: false,
      push: true,
      alerts: true
    },
    system: {
      autoUpdate: true,
      maintenanceMode: false,
      debugMode: false,
      logLevel: 'info'
    },
    security: {
      twoFactor: true,
      sessionTimeout: 30,
      ipWhitelist: false,
      encryption: true
    },
    ai: {
      autoTraining: true,
      modelRefresh: 24,
      confidenceThreshold: 0.8,
      maxPredictions: 1000
    }
  });

  const handleSettingChange = (category, setting, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [setting]: value
      }
    }));
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Settings</h1>
        <p className="text-gray-600">Configure system settings and preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Notifications Settings */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center mb-6">
            <BellIcon className="w-6 h-6 text-blue-600 mr-3" />
            <h3 className="text-lg font-semibold text-gray-900">Notifications</h3>
          </div>
          
          <div className="space-y-4">
            {Object.entries(settings.notifications).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-900 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <p className="text-xs text-gray-500">
                    {key === 'email' && 'Receive notifications via email'}
                    {key === 'sms' && 'Receive notifications via SMS'}
                    {key === 'push' && 'Receive push notifications'}
                    {key === 'alerts' && 'Receive system alerts'}
                  </p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    className="sr-only peer"
                    checked={value}
                    onChange={(e) => handleSettingChange('notifications', key, e.target.checked)}
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>
            ))}
          </div>
        </motion.div>

        {/* System Settings */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center mb-6">
            <ServerIcon className="w-6 h-6 text-green-600 mr-3" />
            <h3 className="text-lg font-semibold text-gray-900">System</h3>
          </div>
          
          <div className="space-y-4">
            {Object.entries(settings.system).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-900 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <p className="text-xs text-gray-500">
                    {key === 'autoUpdate' && 'Automatically update system components'}
                    {key === 'maintenanceMode' && 'Enable maintenance mode'}
                    {key === 'debugMode' && 'Enable debug logging'}
                    {key === 'logLevel' && 'Set logging verbosity level'}
                  </p>
                </div>
                {key === 'logLevel' ? (
                  <select
                    value={value}
                    onChange={(e) => handleSettingChange('system', key, e.target.value)}
                    className="px-3 py-1 border border-gray-300 rounded-md text-sm"
                  >
                    <option value="debug">Debug</option>
                    <option value="info">Info</option>
                    <option value="warn">Warning</option>
                    <option value="error">Error</option>
                  </select>
                ) : (
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={value}
                      onChange={(e) => handleSettingChange('system', key, e.target.checked)}
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                )}
              </div>
            ))}
          </div>
        </motion.div>

        {/* Security Settings */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center mb-6">
            <ShieldCheckIcon className="w-6 h-6 text-red-600 mr-3" />
            <h3 className="text-lg font-semibold text-gray-900">Security</h3>
          </div>
          
          <div className="space-y-4">
            {Object.entries(settings.security).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-900 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <p className="text-xs text-gray-500">
                    {key === 'twoFactor' && 'Enable two-factor authentication'}
                    {key === 'sessionTimeout' && 'Session timeout in minutes'}
                    {key === 'ipWhitelist' && 'Restrict access to whitelisted IPs'}
                    {key === 'encryption' && 'Enable data encryption'}
                  </p>
                </div>
                {key === 'sessionTimeout' ? (
                  <input
                    type="number"
                    value={value}
                    onChange={(e) => handleSettingChange('security', key, parseInt(e.target.value))}
                    className="w-20 px-3 py-1 border border-gray-300 rounded-md text-sm"
                    min="5"
                    max="480"
                  />
                ) : (
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={value}
                      onChange={(e) => handleSettingChange('security', key, e.target.checked)}
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                )}
              </div>
            ))}
          </div>
        </motion.div>

        {/* AI Settings */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <div className="flex items-center mb-6">
            <ChartBarIcon className="w-6 h-6 text-purple-600 mr-3" />
            <h3 className="text-lg font-semibold text-gray-900">AI Configuration</h3>
          </div>
          
          <div className="space-y-4">
            {Object.entries(settings.ai).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-900 capitalize">
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <p className="text-xs text-gray-500">
                    {key === 'autoTraining' && 'Automatically retrain models'}
                    {key === 'modelRefresh' && 'Model refresh interval (hours)'}
                    {key === 'confidenceThreshold' && 'Minimum confidence threshold'}
                    {key === 'maxPredictions' && 'Maximum predictions per hour'}
                  </p>
                </div>
                {key === 'modelRefresh' ? (
                  <input
                    type="number"
                    value={value}
                    onChange={(e) => handleSettingChange('ai', key, parseInt(e.target.value))}
                    className="w-20 px-3 py-1 border border-gray-300 rounded-md text-sm"
                    min="1"
                    max="168"
                  />
                ) : key === 'confidenceThreshold' ? (
                  <input
                    type="number"
                    step="0.1"
                    value={value}
                    onChange={(e) => handleSettingChange('ai', key, parseFloat(e.target.value))}
                    className="w-20 px-3 py-1 border border-gray-300 rounded-md text-sm"
                    min="0"
                    max="1"
                  />
                ) : key === 'maxPredictions' ? (
                  <input
                    type="number"
                    value={value}
                    onChange={(e) => handleSettingChange('ai', key, parseInt(e.target.value))}
                    className="w-20 px-3 py-1 border border-gray-300 rounded-md text-sm"
                    min="100"
                    max="10000"
                  />
                ) : (
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={value}
                      onChange={(e) => handleSettingChange('ai', key, e.target.checked)}
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Save Button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="mt-8 flex justify-end"
      >
        <button className="bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700 transition-colors">
          Save Settings
        </button>
      </motion.div>
    </div>
  );
};

export default Settings;
