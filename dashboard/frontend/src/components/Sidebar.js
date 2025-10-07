import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  HomeIcon,
  CpuChipIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  DocumentCheckIcon,
  Cog6ToothIcon,
  XMarkIcon,
  PlayIcon,
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'AI 4.0 Dashboard', href: '/', icon: HomeIcon },
  { name: 'User Experience', href: '/d/user-experience', icon: PlayIcon },
  { name: 'YouTube Demo', href: '/d/youtube-demo', icon: PlayIcon },
  { name: 'Quantum Security', href: '/d/quantum-security', icon: ShieldCheckIcon },
  { name: 'Real Data Dashboard', href: '/real-data', icon: HomeIcon },
  { name: 'Legacy Dashboard', href: '/dashboard', icon: ChartBarIcon },
  { name: 'AI Agents', href: '/agents', icon: CpuChipIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  { name: 'Security', href: '/security', icon: ShieldCheckIcon },
  { name: 'Data Quality', href: '/data-quality', icon: DocumentCheckIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
];

const Sidebar = ({ isOpen, onClose }) => {
  const location = useLocation();

  return (
    <>
      {/* Mobile sidebar overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 lg:hidden"
          onClick={onClose}
        >
          <div className="absolute inset-0 bg-gray-600 bg-opacity-75" />
        </div>
      )}

      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{ x: isOpen ? 0 : -280 }}
        transition={{ type: 'tween', duration: 0.3 }}
        className={`
          fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:static lg:inset-0
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-200">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center">
                <CpuChipIcon className="w-5 h-5 text-white" />
              </div>
            </div>
            <div className="ml-3">
              <h1 className="text-lg font-semibold text-gray-900">Telecom AI</h1>
              <p className="text-xs text-gray-500">Enhanced System</p>
            </div>
          </div>
          
          <button
            onClick={onClose}
            className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        <nav className="mt-6 px-3">
          <div className="space-y-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={onClose}
                  className={`
                    group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors duration-200
                    ${isActive
                      ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-700'
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                    }
                  `}
                >
                  <item.icon
                    className={`
                      mr-3 h-5 w-5 flex-shrink-0
                      ${isActive ? 'text-primary-700' : 'text-gray-400 group-hover:text-gray-500'}
                    `}
                  />
                  {item.name}
                </Link>
              );
            })}
          </div>
        </nav>

        {/* System Status */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="w-2 h-2 bg-success-500 rounded-full mr-2"></div>
              <span className="text-xs text-gray-600">System Online</span>
            </div>
            <div className="text-xs text-gray-500">
              v1.0.0
            </div>
          </div>
        </div>
      </motion.div>
    </>
  );
};

export default Sidebar;
