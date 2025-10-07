import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Shield, 
  Lock, 
  CheckCircle, 
  XCircle, 
  Activity, 
  AlertTriangle, 
  Eye, 
  EyeOff, 
  RefreshCw, 
  Zap, 
  Server, 
  Globe,
  Key,
  Cpu,
  Database
} from 'lucide-react';

// Import the Quantum Safe Panel component
import QuantumSafePanel from '../components/ai4/QuantumSafePanel';

const QuantumSecurityPage = () => {
  const [systemData, setSystemData] = useState({
    quantum: {
      status: 'active',
      algorithms: {
        dilithium: { status: 'active', keys: 12 },
        kyber: { status: 'active', keys: 8 },
        sha3: { status: 'active', hashes: 156 }
      },
      audit_logs: {
        total_operations: 1250,
        successful_operations: 1248,
        failed_operations: 2,
        last_audit: new Date().toISOString()
      }
    }
  });

  const [showDetails, setShowDetails] = useState(true);
  const [isQuantumMode, setIsQuantumMode] = useState(true);

  const refreshData = () => {
    // Simulate quantum security data refresh
    setSystemData(prev => ({
      ...prev,
      quantum: {
        ...prev.quantum,
        audit_logs: {
          ...prev.quantum.audit_logs,
          total_operations: prev.quantum.audit_logs.total_operations + Math.floor(Math.random() * 10),
          successful_operations: prev.quantum.audit_logs.successful_operations + Math.floor(Math.random() * 8),
          last_audit: new Date().toISOString()
        }
      }
    }));
  };

  useEffect(() => {
    // Auto-refresh data every 5 seconds in quantum mode
    if (isQuantumMode) {
      const interval = setInterval(refreshData, 5000);
      return () => clearInterval(interval);
    }
  }, [isQuantumMode]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Shield className="w-8 h-8 text-purple-500" />
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Quantum Security Dashboard
                </h1>
              </div>
              <div className="hidden md:flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                <div className="flex items-center space-x-1">
                  <div className={`w-2 h-2 rounded-full ${isQuantumMode ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></div>
                  <span>{isQuantumMode ? 'Quantum-Safe' : 'Classical'}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Lock className="w-4 h-4" />
                  <span>PQC Active</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="flex items-center px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                {showDetails ? <EyeOff className="w-4 h-4 mr-2" /> : <Eye className="w-4 h-4 mr-2" />}
                {showDetails ? 'Hide' : 'Show'} Details
              </button>
              
              <button
                onClick={refreshData}
                className="flex items-center px-3 py-2 text-sm bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </button>
              
              <button
                onClick={() => setIsQuantumMode(!isQuantumMode)}
                className={`flex items-center px-3 py-2 text-sm rounded-lg transition-colors ${
                  isQuantumMode 
                    ? 'bg-purple-600 text-white hover:bg-purple-700' 
                    : 'bg-gray-600 text-white hover:bg-gray-700'
                }`}
              >
                <Shield className="w-4 h-4 mr-2" />
                {isQuantumMode ? 'Quantum-Safe' : 'Classical'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Status Banner */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-6 border border-purple-200 dark:border-purple-800"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Shield className="w-6 h-6 text-purple-500" />
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Post-Quantum Cryptography (PQC) Security
                </h2>
              </div>
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-gray-600 dark:text-gray-400">Active</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Key className="w-4 h-4 text-blue-500" />
                  <span className="text-gray-600 dark:text-gray-400">20 Keys</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Database className="w-4 h-4 text-green-500" />
                  <span className="text-gray-600 dark:text-gray-400">Secure Vault</span>
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <div className="text-2xl font-bold text-green-600">
                95%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Security Score</div>
            </div>
          </div>
        </motion.div>

        {/* Quantum Safe Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <QuantumSafePanel
            data={systemData.quantum}
            onRefresh={refreshData}
            showDetails={showDetails}
            isQuantumMode={isQuantumMode}
          />
        </motion.div>

        {/* Quantum Security Metrics */}
        {showDetails && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {/* Dilithium Keys */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Dilithium Keys
                </h3>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.quantum.algorithms.dilithium.keys}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Digital signatures
              </div>
            </div>

            {/* Kyber Keys */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Kyber Keys
                </h3>
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.quantum.algorithms.kyber.keys}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Key encapsulation
              </div>
            </div>

            {/* SHA-3 Hashes */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  SHA-3 Hashes
                </h3>
                <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.quantum.algorithms.sha3.hashes}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Hash operations
              </div>
            </div>

            {/* Audit Operations */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Audit Operations
                </h3>
                <Activity className="w-5 h-5 text-green-500" />
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                {systemData.quantum.audit_logs.total_operations}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Total operations
              </div>
            </div>
          </motion.div>
        )}

        {/* Security Comparison */}
        {showDetails && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* Before Quantum Upgrade */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center mb-4">
                <XCircle className="w-6 h-6 text-red-500 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Before Quantum Upgrade
                </h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">RSA Keys</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-red-500">12</span>
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">ECC Keys</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-red-500">8</span>
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">SHA-256 Hashes</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-red-500">156</span>
                    <AlertTriangle className="w-4 h-4 text-red-500" />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Security Score</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-red-500">35%</span>
                    <XCircle className="w-4 h-4 text-red-500" />
                  </div>
                </div>
              </div>
            </div>

            {/* After Quantum Upgrade */}
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center mb-4">
                <CheckCircle className="w-6 h-6 text-green-500 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  After Quantum Upgrade
                </h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Dilithium Keys</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-green-500">12</span>
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Kyber Keys</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-green-500">8</span>
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">SHA-3-256 Hashes</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-green-500">156</span>
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Security Score</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-bold text-green-500">95%</span>
                    <Shield className="w-4 h-4 text-green-500" />
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Quantum Security Info */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-8 bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-lg p-6 border border-purple-200 dark:border-purple-800"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            🔐 Post-Quantum Cryptography (PQC) Features
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Classical Cryptography (Vulnerable):
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• RSA-2048/4096 (vulnerable to Shor's algorithm)</li>
                <li>• ECC P-256/P-384 (vulnerable to quantum attacks)</li>
                <li>• SHA-256 (vulnerable to Grover's algorithm)</li>
                <li>• AES-128 (reduced security with quantum computers)</li>
                <li>• Security score: 35%</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Post-Quantum Cryptography (Safe):
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Dilithium (quantum-safe digital signatures)</li>
                <li>• Kyber (quantum-safe key encapsulation)</li>
                <li>• SHA-3-256 (quantum-resistant hash function)</li>
                <li>• AES-256 (quantum-safe with proper key sizes)</li>
                <li>• Security score: 95%</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default QuantumSecurityPage;




