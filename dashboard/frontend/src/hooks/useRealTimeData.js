/**
 * Custom Hook for Real-Time Data Management
 * Handles data fetching, error states, and loading indicators
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import realTimeService from '../services/realTimeService';

export const useRealTimeData = (pollingInterval = 5000) => {
  const [data, setData] = useState({
    health: null,
    kpis: null,
    ibn: null,
    zta: null,
    quantum: null,
    federation: null,
    selfEvolution: null,
    observability: null,
    api: null
  });
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [componentErrors, setComponentErrors] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  
  const isInitialized = useRef(false);
  const unsubscribeRef = useRef(null);

  /**
   * Handle incoming real-time data
   */
  const handleDataUpdate = useCallback((update) => {
    if (update.data) {
      setData(prevData => ({
        ...prevData,
        ...update.data
      }));
    }

    if (update.errors) {
      setComponentErrors(update.errors);
    } else {
      setComponentErrors({});
    }

    setIsConnected(update.isConnected);
    setLastUpdate(new Date(update.timestamp));
    setLoading(false);
    setError(null);
  }, []);

  /**
   * Handle polling errors
   */
  const handlePollingError = useCallback((error) => {
    console.error('Real-time data error:', error);
    setError(error.message || 'Failed to fetch real-time data');
    setIsConnected(false);
    setLoading(false);
  }, []);

  /**
   * Start real-time data polling
   */
  const startPolling = useCallback(() => {
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
    }

    // Subscribe to real-time updates
    unsubscribeRef.current = realTimeService.addListener(handleDataUpdate);
    
    // Start polling
    realTimeService.startPolling(pollingInterval);
    
    console.log('🚀 Real-time data polling started');
  }, [handleDataUpdate, pollingInterval]);

  /**
   * Stop real-time data polling
   */
  const stopPolling = useCallback(() => {
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
      unsubscribeRef.current = null;
    }
    
    realTimeService.stopPolling();
    console.log('⏹️ Real-time data polling stopped');
  }, []);

  /**
   * Manual data refresh
   */
  const refreshData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await realTimeService.fetchAllSystemData();
      handleDataUpdate(result);
    } catch (error) {
      handlePollingError(error);
    }
  }, [handleDataUpdate, handlePollingError]);

  /**
   * Test backend connection
   */
  const testConnection = useCallback(async () => {
    const isConnected = await realTimeService.testConnection();
    setIsConnected(isConnected);
    return isConnected;
  }, []);

  /**
   * Get connection status
   */
  const getConnectionStatus = useCallback(() => {
    return realTimeService.getConnectionStatus();
  }, []);

  // Initialize on mount
  useEffect(() => {
    if (!isInitialized.current) {
      isInitialized.current = true;
      startPolling();
    }

    return () => {
      stopPolling();
    };
  }, [startPolling, stopPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
      realTimeService.stopPolling();
    };
  }, []);

  return {
    // Data
    data,
    loading,
    error,
    componentErrors,
    isConnected,
    lastUpdate,
    
    // Actions
    refreshData,
    startPolling,
    stopPolling,
    testConnection,
    getConnectionStatus,
    
    // Status
    hasData: Object.values(data).some(value => value !== null),
    hasErrors: Object.keys(componentErrors).length > 0,
    isPolling: realTimeService.pollingInterval !== null
  };
};
