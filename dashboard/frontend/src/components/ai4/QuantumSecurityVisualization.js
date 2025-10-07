import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  Lock, 
  Key, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  ArrowRight,
  Zap,
  Eye,
  EyeOff,
  RefreshCw,
  Database,
  Server,
  Globe,
  Cpu
} from 'lucide-react';

const QuantumSecurityVisualization = ({ data, onRefresh }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [animationPhase, setAnimationPhase] = useState('before'); // 'before', 'transition', 'after'
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

  const [dataPackets, setDataPackets] = useState([]);
  const [isAnimating, setIsAnimating] = useState(false);

  // Simulate data packet flow
  const createDataPacket = useCallback(() => {
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
  }, [animationPhase]);

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
  }, [isAnimating, createDataPacket]);

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

  const getSecurityIcon = (type, isSecure) => {
    if (isSecure) {
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    } else {
      return <XCircle className="w-5 h-5 text-red-500" />;
    }
  };

  const getEncryptionColor = (type) => {
    if (animationPhase === 'before') {
      return 'text-red-500 bg-red-50 dark:bg-red-900/20';
    } else {
      return 'text-green-500 bg-green-50 dark:bg-green-900/20';
    }
  };

  const currentMetrics = animationPhase === 'before' ? securityMetrics.before : securityMetrics.after;

  return (
    <div id="quantum-security-viz" className="dashboard-panel-full dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Quantum Security Visualization
          </h2>
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
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
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

        {/* Three-Panel Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Before Security Upgrade */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className={`p-6 rounded-lg border-2 ${
              animationPhase === 'before' 
                ? 'bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700' 
                : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
            }`}
          >
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
          </motion.div>

          {/* Transition Animation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="flex items-center justify-center"
          >
            <div className="text-center">
              <motion.div
                animate={{ 
                  rotate: isAnimating ? 360 : 0,
                  scale: isAnimating ? 1.2 : 1
                }}
                transition={{ duration: 2, repeat: isAnimating ? Infinity : 0 }}
                className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mb-4"
              >
                <Zap className="w-8 h-8 text-white" />
              </motion.div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Quantum Upgrade
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {animationPhase === 'transition' ? 'Upgrading...' : 'Click to toggle'}
              </p>
            </div>
          </motion.div>

          {/* After Quantum Upgrade */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className={`p-6 rounded-lg border-2 ${
              animationPhase === 'after' 
                ? 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700' 
                : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
            }`}
          >
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
          </motion.div>
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

        {/* Detailed Security Metrics */}
        {showDetails && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {/* Encryption Status */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Encryption Status
              </h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Digital Signatures</span>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${getEncryptionColor()}`}>
                    {animationPhase === 'before' ? 'RSA-2048' : 'Dilithium'}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Key Exchange</span>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${getEncryptionColor()}`}>
                    {animationPhase === 'before' ? 'ECDH' : 'Kyber'}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Hash Function</span>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${getEncryptionColor()}`}>
                    {animationPhase === 'before' ? 'SHA-256' : 'SHA-3-256'}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">TLS Version</span>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${getEncryptionColor()}`}>
                    {animationPhase === 'before' ? 'TLS 1.2' : 'TLS 1.3 + PQC'}
                  </div>
                </div>
              </div>
            </div>

            {/* Security Recommendations */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Security Recommendations
              </h4>
              <div className="space-y-3">
                {animationPhase === 'before' ? (
                  <>
                    <div className="flex items-start space-x-2">
                      <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Replace RSA-2048 with Dilithium signatures
                      </span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Upgrade ECDH to Kyber key exchange
                      </span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Migrate SHA-256 to SHA-3-256
                      </span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Implement secure key vault
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Dilithium signatures implemented
                      </span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Kyber key exchange active
                      </span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        SHA-3-256 hashing enabled
                      </span>
                    </div>
                    <div className="flex items-start space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Secure key vault operational
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default QuantumSecurityVisualization;




