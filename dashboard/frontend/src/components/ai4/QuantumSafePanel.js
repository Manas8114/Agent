import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
  Globe
} from 'lucide-react';

const QuantumSafePanel = ({ data }) => {
  const metrics = data || {};
  const algorithms = metrics.algorithms || {};
  const auditLogs = metrics.audit_logs || {};
  
  // Interactive state
  const [showDetails, setShowDetails] = useState(false);
  const [animationPhase, setAnimationPhase] = useState('after'); // 'before', 'transition', 'after'
  const [dataPackets, setDataPackets] = useState([]);
  const [isAnimating, setIsAnimating] = useState(false);
  
  // Security metrics simulation
  const [securityMetrics, setSecurityMetrics] = useState({
    before: {
      rsaKeys: 12,
      eccKeys: 8,
      sha256Hashes: 156,
      vulnerableTokens: 23,
      exposedSecrets: 5,
      weakCiphers: 7,
      securityScore: 35
    },
    after: {
      dilithiumKeys: 12,
      kyberKeys: 8,
      sha3Hashes: 156,
      secureTokens: 23,
      vaultedSecrets: 5,
      pqCiphers: 7,
      securityScore: 95
    }
  });
  
  // Create data packet for animation
  const createDataPacket = () => {
    const packet = {
      id: Date.now() + Math.random(),
      type: animationPhase === 'before' ? 'vulnerable' : 'secure',
      startTime: Date.now(),
      progress: 0,
      encryption: animationPhase === 'before' ? 'RSA-2048' : 'Dilithium',
      keyExchange: animationPhase === 'before' ? 'ECDH' : 'Kyber',
      hash: animationPhase === 'before' ? 'SHA-256' : 'SHA-3-256'
    };
    
    setDataPackets(prev => [...prev, packet]);
    
    // Remove packet after animation
    setTimeout(() => {
      setDataPackets(prev => prev.filter(p => p.id !== packet.id));
    }, 5000);
  };
  
  // Animate data packets
  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setDataPackets(prev => prev.map(packet => ({
          ...packet,
          progress: Math.min(packet.progress + 2, 100)
        })));
      }, 50);
      
      return () => clearInterval(interval);
    }
  }, [isAnimating]);
  
  // Create new packets periodically
  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(createDataPacket, 2000);
      return () => clearInterval(interval);
    }
  }, [isAnimating, animationPhase]);
  
  const toggleAnimation = () => {
    setIsAnimating(true);
    
    if (animationPhase === 'before') {
      setAnimationPhase('transition');
      setTimeout(() => {
        setAnimationPhase('after');
        setIsAnimating(false);
      }, 2000);
    } else {
      setAnimationPhase('transition');
      setTimeout(() => {
        setAnimationPhase('before');
        setIsAnimating(false);
      }, 2000);
    }
  };
  
  const currentMetrics = animationPhase === 'before' ? securityMetrics.before : securityMetrics.after;

  const getAlgorithmColor = (algorithm) => {
    switch (algorithm) {
      case 'kyber':
        return 'text-blue-600 bg-blue-50 dark:bg-blue-900 dark:text-blue-200';
      case 'dilithium':
        return 'text-purple-600 bg-purple-50 dark:bg-purple-900 dark:text-purple-200';
      default:
        return 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Shield className="w-5 h-5 text-purple-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Quantum-Safe Security
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center px-3 py-1 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              {showDetails ? <EyeOff className="w-4 h-4 mr-1" /> : <Eye className="w-4 h-4 mr-1" />}
              {showDetails ? 'Hide' : 'Show'} Details
            </button>
            <button
              onClick={toggleAnimation}
              className="flex items-center px-3 py-1 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              <RefreshCw className="w-4 h-4 mr-1" />
              Toggle {animationPhase === 'before' ? 'Quantum-Safe' : 'Classical'}
            </button>
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Active</span>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* PQC Algorithm Usage */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            PQC Algorithm Usage
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(algorithms).map(([algorithm, stats]) => (
              <div key={algorithm} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                    {algorithm}
                  </span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAlgorithmColor(algorithm)}`}>
                    {stats.usage}%
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${stats.usage}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    {(stats.success_rate * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Verified Messages vs Failed Signatures */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Message Verification
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-800 dark:text-green-200">Verified Messages</p>
                  <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                    {metrics.pqc_verifications_total?.toLocaleString() || 'N/A'}
                  </p>
                </div>
                <CheckCircle className="w-8 h-8 text-green-500" />
              </div>
            </div>
            
            <div className="bg-red-50 dark:bg-red-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-red-800 dark:text-red-200">Failed Signatures</p>
                  <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                    {((metrics.pqc_verifications_total || 0) - (metrics.pqc_verifications_total || 0) * (metrics.pqc_verification_success_rate || 0)).toFixed(0)}
                  </p>
                </div>
                <XCircle className="w-8 h-8 text-red-500" />
              </div>
            </div>
          </div>
        </div>

        {/* Audit Log Integrity */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Audit Log Integrity
          </h3>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {auditLogs.total?.toLocaleString() || 'N/A'}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Logs</p>
              </div>
              
              <div className="text-center">
                <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {auditLogs.tamper_attempts || 0}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Tamper Attempts</p>
              </div>
              
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {auditLogs.integrity_score?.toFixed(1) || 'N/A'}%
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Integrity Score</p>
              </div>
            </div>
          </div>
        </div>

        {/* Encryption/Decryption Latency */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Performance Metrics
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-800 dark:text-blue-200">Encryption Latency</p>
                  <p className="text-lg font-bold text-blue-600 dark:text-blue-400">
                    {((metrics.pqc_encryptions_total || 0) / 1000).toFixed(1)}ms
                  </p>
                </div>
                <Lock className="w-6 h-6 text-blue-500" />
              </div>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-purple-800 dark:text-purple-200">Decryption Latency</p>
                  <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                    {((metrics.pqc_decryptions_total || 0) / 1000).toFixed(1)}ms
                  </p>
                </div>
                <Shield className="w-6 h-6 text-purple-500" />
              </div>
            </div>
          </div>
        </div>

        {/* Blockchain Transactions */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Blockchain Transactions
          </h3>
          <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-4 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm opacity-90">Transactions per Second</p>
                <p className="text-2xl font-bold">
                  {Math.floor((metrics.pqc_signatures_total || 0) / 60)} TPS
                </p>
              </div>
              <Activity className="w-8 h-8 opacity-80" />
            </div>
          </div>
        </div>

        {/* Security Status */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Security Status
          </h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900 rounded-lg">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-800 dark:text-green-200">
                  Quantum-Safe Encryption Active
                </span>
              </div>
              <span className="text-xs text-green-600 dark:text-green-400">
                {(metrics.pqc_encryption_success_rate * 100).toFixed(1)}% success
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900 rounded-lg">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-800 dark:text-green-200">
                  Digital Signatures Verified
                </span>
              </div>
              <span className="text-xs text-green-600 dark:text-green-400">
                {(metrics.pqc_verification_success_rate * 100).toFixed(1)}% success
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-yellow-50 dark:bg-yellow-900 rounded-lg">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500" />
                <span className="text-sm text-yellow-800 dark:text-yellow-200">
                  {auditLogs.tamper_attempts || 0} Tamper Attempts Detected
                </span>
              </div>
              <span className="text-xs text-yellow-600 dark:text-yellow-400">
                Last: 2 hours ago
              </span>
            </div>
          </div>
        </div>

        {/* Interactive Security Visualization */}
        {showDetails && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-6 border-t border-gray-200 dark:border-gray-700 pt-6"
          >
            {/* Security Status Banner */}
            <div className={`mb-6 p-4 rounded-lg border-2 ${
              animationPhase === 'before' 
                ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
                : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {animationPhase === 'before' ? (
                    <AlertTriangle className="w-8 h-8 text-red-500" />
                  ) : (
                    <Shield className="w-8 h-8 text-green-500" />
                  )}
                  <div>
                    <h3 className={`text-lg font-semibold ${
                      animationPhase === 'before' ? 'text-red-800 dark:text-red-300' : 'text-green-800 dark:text-green-300'
                    }`}>
                      {animationPhase === 'before' ? 'Classical Security (Vulnerable)' : 'Post-Quantum Security (Safe)'}
                    </h3>
                    <p className={`text-sm ${
                      animationPhase === 'before' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
                    }`}>
                      {animationPhase === 'before' 
                        ? 'Current system vulnerable to quantum attacks' 
                        : 'System protected against quantum computing threats'
                      }
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-3xl font-bold ${
                    animationPhase === 'before' ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {currentMetrics.securityScore}%
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Security Score</div>
                </div>
              </div>
            </div>

            {/* Real-time Data Packet Flow */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Real-time Data Packet Flow
              </h3>
              
              <div className="relative bg-gray-100 dark:bg-gray-700 rounded-lg p-6 h-32 overflow-hidden">
                {/* Network path */}
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full h-1 bg-gray-300 dark:bg-gray-600 rounded-full relative">
                    <div className="absolute left-0 top-0 w-4 h-4 bg-blue-500 rounded-full -mt-1.5 flex items-center justify-center">
                      <Server className="w-2 h-2 text-white" />
                    </div>
                    <div className="absolute right-0 top-0 w-4 h-4 bg-green-500 rounded-full -mt-1.5 flex items-center justify-center">
                      <Globe className="w-2 h-2 text-white" />
                    </div>
                  </div>
                </div>
                
                {/* Animated data packets */}
                <AnimatePresence>
                  {dataPackets.map((packet) => (
                    <motion.div
                      key={packet.id}
                      initial={{ x: 0, opacity: 0 }}
                      animate={{ x: `${packet.progress}%`, opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute top-1/2 transform -translate-y-1/2"
                    >
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                        packet.type === 'vulnerable' 
                          ? 'bg-red-500' 
                          : 'bg-green-500'
                      }`}>
                        <Lock className="w-3 h-3 text-white" />
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
              
              <div className="mt-4 flex items-center justify-center space-x-6">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {animationPhase === 'before' ? 'Vulnerable (RSA/ECC)' : 'Classical Encryption'}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {animationPhase === 'after' ? 'Quantum-Safe (Dilithium/Kyber)' : 'Post-Quantum Encryption'}
                  </span>
                </div>
              </div>
            </div>

            {/* Security Metrics Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Before Security Upgrade */}
              <div className={`p-6 rounded-lg border-2 ${
                animationPhase === 'before' 
                  ? 'bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700' 
                  : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
              }`}>
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
                      <span className="text-lg font-bold text-red-500">{securityMetrics.before.rsaKeys}</span>
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">ECC Keys</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg font-bold text-red-500">{securityMetrics.before.eccKeys}</span>
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">SHA-256 Hashes</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg font-bold text-red-500">{securityMetrics.before.sha256Hashes}</span>
                      <AlertTriangle className="w-4 h-4 text-red-500" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Exposed Secrets</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg font-bold text-red-500">{securityMetrics.before.exposedSecrets}</span>
                      <XCircle className="w-4 h-4 text-red-500" />
                    </div>
                  </div>
                </div>
              </div>

              {/* After Quantum Upgrade */}
              <div className={`p-6 rounded-lg border-2 ${
                animationPhase === 'after' 
                  ? 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700' 
                  : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
              }`}>
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
                      <span className="text-lg font-bold text-green-500">{securityMetrics.after.dilithiumKeys}</span>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Kyber Keys</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg font-bold text-green-500">{securityMetrics.after.kyberKeys}</span>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">SHA-3-256 Hashes</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg font-bold text-green-500">{securityMetrics.after.sha3Hashes}</span>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Vaulted Secrets</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg font-bold text-green-500">{securityMetrics.after.vaultedSecrets}</span>
                      <Shield className="w-4 h-4 text-green-500" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default QuantumSafePanel;

