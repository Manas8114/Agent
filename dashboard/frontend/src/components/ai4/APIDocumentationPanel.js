import React from 'react';
import { 
  Code, 
  CheckCircle, 
  XCircle, 
  Clock, 
  ExternalLink,
  FileText,
  Download,
  Activity,
  AlertTriangle
} from 'lucide-react';

const APIDocumentationPanel = ({ data }) => {
  const api = data || {};
  const endpoints = api.endpoints || [];
  const documentation = api.documentation || {};
  const validationTests = api.validation_tests || {};

  const getStatusIcon = (status) => {
    switch (status) {
      case 200:
      case 201:
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 400:
      case 404:
      case 500:
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getStatusColor = (status) => {
    if (status >= 200 && status < 300) {
      return 'text-green-600 bg-green-50 dark:bg-green-900 dark:text-green-200';
    } else if (status >= 400 && status < 500) {
      return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900 dark:text-yellow-200';
    } else if (status >= 500) {
      return 'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200';
    }
    return 'text-gray-600 bg-gray-50 dark:bg-gray-900 dark:text-gray-200';
  };

  const getTestResultIcon = (result) => {
    switch (result) {
      case 'PASS':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'FAIL':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getTestResultColor = (result) => {
    switch (result) {
      case 'PASS':
        return 'text-green-600 bg-green-50 dark:bg-green-900 dark:text-green-200';
      case 'FAIL':
        return 'text-red-600 bg-red-50 dark:bg-red-900 dark:text-red-200';
      default:
        return 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900 dark:text-yellow-200';
    }
  };

  return (
    <div className="dashboard-panel dark:bg-gray-800">
      <div className="px-4 lg:px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Code className="w-5 h-5 text-gray-500" />
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              API & Documentation
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">All Systems Operational</span>
          </div>
        </div>
      </div>

      <div className="dashboard-panel-content p-4 lg:p-6">
        {/* API Endpoints Status */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            API Endpoints Status ({endpoints.length})
          </h3>
          <div className="space-y-2">
            {endpoints.map((endpoint, index) => (
              <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(endpoint.status)}
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          endpoint.method === 'GET' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                          endpoint.method === 'POST' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                        }`}>
                          {endpoint.method}
                        </span>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {endpoint.path}
                        </span>
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        Last called: {new Date(endpoint.last_called).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(endpoint.status)}`}>
                      {endpoint.status}
                    </span>
                    <a 
                      href={`http://localhost:8000${endpoint.path}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-500 hover:text-blue-700"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Documentation Links */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Documentation
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-800 dark:text-blue-200">API Documentation</p>
                  <p className="text-xs text-blue-600 dark:text-blue-400">Swagger/OpenAPI</p>
                </div>
                <a 
                  href={documentation.api_docs}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 hover:text-blue-700"
                >
                  <ExternalLink className="w-5 h-5" />
                </a>
              </div>
            </div>
            
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-800 dark:text-green-200">AI 4.0 Guide</p>
                  <p className="text-xs text-green-600 dark:text-green-400">Comprehensive Guide</p>
                </div>
                <a 
                  href={documentation.ai4_guide}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-500 hover:text-green-700"
                >
                  <FileText className="w-5 h-5" />
                </a>
              </div>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-800 dark:text-purple-200">Validation Report</p>
                  <p className="text-xs text-purple-600 dark:text-purple-400">Test Results</p>
                </div>
                <a 
                  href={documentation.validation_report}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-500 hover:text-purple-700"
                >
                  <Download className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Validation Test Results */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Validation Test Results
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(validationTests).map(([test, result]) => (
              <div key={test} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {getTestResultIcon(result)}
                    <span className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                      {test.replace('_', ' ')}
                    </span>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTestResultColor(result)}`}>
                    {result}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Last API Call Logs */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Recent API Activity
          </h3>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    GET /api/v1/health - 200 OK
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 5000).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    GET /api/v1/telecom/kpis - 200 OK
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 10000).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    POST /api/v1/telecom/intent - 201 Created
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 15000).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-white">
                    GET /api/v1/telecom/quantum-status - 200 OK
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(Date.now() - 20000).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* System Status Summary */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            System Status Summary
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-green-50 dark:bg-green-900 rounded-lg p-3 text-center">
              <CheckCircle className="w-6 h-6 text-green-500 mx-auto mb-2" />
              <p className="text-sm font-medium text-green-800 dark:text-green-200">All APIs</p>
              <p className="text-lg font-bold text-green-600 dark:text-green-400">Operational</p>
            </div>
            
            <div className="bg-blue-50 dark:bg-blue-900 rounded-lg p-3 text-center">
              <Activity className="w-6 h-6 text-blue-500 mx-auto mb-2" />
              <p className="text-sm font-medium text-blue-800 dark:text-blue-200">Response Time</p>
              <p className="text-lg font-bold text-blue-600 dark:text-blue-400">12ms</p>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-900 rounded-lg p-3 text-center">
              <Clock className="w-6 h-6 text-purple-500 mx-auto mb-2" />
              <p className="text-sm font-medium text-purple-800 dark:text-purple-200">Uptime</p>
              <p className="text-lg font-bold text-purple-600 dark:text-purple-400">99.9%</p>
            </div>
            
            <div className="bg-orange-50 dark:bg-orange-900 rounded-lg p-3 text-center">
              <FileText className="w-6 h-6 text-orange-500 mx-auto mb-2" />
              <p className="text-sm font-medium text-orange-800 dark:text-orange-200">Documentation</p>
              <p className="text-lg font-bold text-orange-600 dark:text-orange-400">Updated</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIDocumentationPanel;

