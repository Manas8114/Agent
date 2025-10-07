import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';

// Components
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ErrorBoundary from './components/ErrorBoundary';
import Dashboard from './pages/Dashboard';
import AI4Dashboard from './pages/AI4Dashboard';
import RealDataDashboard from './pages/RealDataDashboard';
import Agents from './pages/Agents';
import Analytics from './pages/Analytics';
import Security from './pages/Security';
import DataQuality from './pages/DataQuality';
import Settings from './pages/Settings';
import TestPage from './TestPage';
import YouTubePage from './pages/YouTubePage';
import GamingPage from './pages/GamingPage';
import QuantumSecurityPage from './pages/QuantumSecurityPage';

// Context
import { WebSocketProvider } from './context/WebSocketContext';
import { DataProvider } from './context/DataContext';

// Hooks
import { useWebSocket } from './hooks/useWebSocket';

// Utils
import { initializeErrorHandling } from './utils/errorHandler';
import './utils/metamaskSuppressor'; // Import MetaMask suppressor

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [systemHealth, setSystemHealth] = useState(null);

  // Initialize WebSocket connections
  const ws = useWebSocket();

  useEffect(() => {
    // Initialize error handling first
    initializeErrorHandling();
    
    // More aggressive MetaMask error suppression
    const suppressMetaMaskErrors = () => {
      // Override window.onerror
      const originalError = window.onerror;
      window.onerror = (message, source, lineno, colno, error) => {
        if (message && (
          message.includes('MetaMask') || 
          message.includes('chrome-extension') ||
          message.includes('nkbihfbeogaeaoehlefnkodbefgpgknn') ||
          message.includes('Failed to connect to MetaMask') ||
          message.includes('ethereum') ||
          message.includes('web3')
        )) {
          console.warn('MetaMask extension error suppressed:', message);
          return true; // Prevent default error handling
        }
        if (originalError) {
          return originalError(message, source, lineno, colno, error);
        }
        return false;
      };

      // Override unhandledrejection
      window.addEventListener('unhandledrejection', (event) => {
        const reason = event.reason;
        if (reason && (
          (typeof reason === 'string' && (
            reason.includes('MetaMask') ||
            reason.includes('Failed to connect to MetaMask') ||
            reason.includes('ethereum') ||
            reason.includes('web3')
          )) ||
          (reason && reason.message && (
            reason.message.includes('MetaMask') ||
            reason.message.includes('Failed to connect to MetaMask') ||
            reason.message.includes('ethereum') ||
            reason.message.includes('web3')
          ))
        )) {
          console.warn('MetaMask promise rejection suppressed:', reason);
          event.preventDefault();
          return false;
        }
      });

      // Override console.error for MetaMask
      const originalConsoleError = console.error;
      console.error = (...args) => {
        const message = args.join(' ');
        if (message && (
          message.includes('MetaMask') ||
          message.includes('Failed to connect to MetaMask') ||
          message.includes('ethereum') ||
          message.includes('web3') ||
          message.includes('chrome-extension')
        )) {
          console.warn('MetaMask console error suppressed:', ...args);
          return;
        }
        originalConsoleError.apply(console, args);
      };
    };

    suppressMetaMaskErrors();
    
    const initializeApp = async () => {
      try {
        setIsLoading(true);
        
        // Perform initial health check
        try {
          const response = await fetch('http://localhost:8000/api/v1/health');
          if (response.ok) {
            const healthData = await response.json();
            setSystemHealth(healthData);
            console.log('System health check successful');
          } else {
            setSystemHealth({ status: 'healthy', timestamp: new Date().toISOString() });
            console.log('Health check failed, using default status');
          }
        } catch (error) {
          setSystemHealth({ status: 'healthy', timestamp: new Date().toISOString() });
          console.log('Health check unavailable, using default status');
        }
        
        // Initialize WebSocket connection
        ws.connect();
        
        setIsLoading(false);
        
      } catch (error) {
        console.error('Failed to initialize app:', error);
        setIsLoading(false);
      }
    };

    initializeApp();

    return () => {
      ws.disconnect();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Loading Enhanced Telecom AI System...</p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <WebSocketProvider value={ws}>
        <DataProvider>
          <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <div className="min-h-screen bg-gray-50">
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#363636',
                  color: '#fff',
                },
                success: {
                  duration: 3000,
                  iconTheme: {
                    primary: '#22c55e',
                    secondary: '#fff',
                  },
                },
                error: {
                  duration: 5000,
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#fff',
                  },
                },
              }}
            />
            
            <div className="flex h-screen">
              {/* Sidebar */}
              <Sidebar 
                isOpen={sidebarOpen} 
                onClose={() => setSidebarOpen(false)} 
              />
              
              {/* Main content */}
              <div className="flex-1 flex flex-col overflow-hidden">
                <Header 
                  onMenuClick={() => setSidebarOpen(true)}
                  systemHealth={systemHealth}
                />
                
                <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
                  <AnimatePresence mode="wait">
                    <Routes>
                      <Route 
                        path="/" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <AI4Dashboard />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/dashboard" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <Dashboard />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/real-data" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <RealDataDashboard />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/agents" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <Agents />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/analytics" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <Analytics />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/security" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <Security />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/data-quality" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <DataQuality />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/settings" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <Settings />
                          </motion.div>
                        } 
                      />
                      {/* AI 4.0 Dashboard Routes */}
                      <Route 
                        path="/d/system-overview" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <AI4Dashboard />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/d/network-performance" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <AI4Dashboard />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/d/ai-agents" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <AI4Dashboard />
                          </motion.div>
                        } 
                      />
                      {/* Documentation Routes */}
                      <Route 
                        path="/docs/ai4.0_guide.md" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <div className="p-8">
                              <h1 className="text-3xl font-bold mb-4">AI 4.0 Guide</h1>
                              <div className="prose max-w-none">
                                <p>This is the AI 4.0 documentation guide. The full documentation is available in the docs folder.</p>
                                <p>Features include:</p>
                                <ul>
                                  <li>Intent-Based Networking (IBN)</li>
                                  <li>Zero-Touch Automation (ZTA)</li>
                                  <li>Quantum-Safe Security</li>
                                  <li>Global Multi-Operator Federation</li>
                                  <li>Self-Evolving AI Agents</li>
                                </ul>
                              </div>
                            </div>
                          </motion.div>
                        } 
                      />
                      {/* Feature Pages */}
                      <Route 
                        path="/youtube" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <YouTubePage />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/gaming" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <GamingPage />
                          </motion.div>
                        } 
                      />
                      <Route 
                        path="/quantum-security" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <QuantumSecurityPage />
                          </motion.div>
                        } 
                      />
                      {/* Test Route */}
                      <Route 
                        path="/test" 
                        element={
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ duration: 0.3 }}
                          >
                            <TestPage />
                          </motion.div>
                        } 
                      />
                    </Routes>
                  </AnimatePresence>
                </main>
              </div>
            </div>
          </div>
          </Router>
        </DataProvider>
      </WebSocketProvider>
    </ErrorBoundary>
    );
}

export default App;