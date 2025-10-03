import React from 'react';
import { 
  Shield, 
  Lock, 
  CheckCircle, 
  XCircle, 
  Activity,
  AlertTriangle
} from 'lucide-react';

const QuantumSafePanel = ({ data }) => {
  const metrics = data || {};
  const algorithms = metrics.algorithms || {};
  const auditLogs = metrics.audit_logs || {};

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
      </div>
    </div>
  );
};

export default QuantumSafePanel;

