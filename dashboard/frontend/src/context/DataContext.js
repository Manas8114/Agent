import React, { createContext, useContext, useState } from 'react';
import useApi from '../hooks/useApi';

const DataContext = createContext();

export const useDataContext = () => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useDataContext must be used within a DataProvider');
  }
  return context;
};

export const DataProvider = ({ children }) => {
  const [kpis, setKpis] = useState(null);
  const [coordination, setCoordination] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [agents, setAgents] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const api = useApi();

  const fetchKpis = async () => {
    try {
      const data = await api.get('/telecom/kpis');
      setKpis(data);
      return data;
    } catch (error) {
      console.error('Failed to fetch KPIs:', error);
      throw error;
    }
  };

  const fetchCoordination = async () => {
    try {
      const data = await api.get('/telecom/coordination');
      setCoordination(data);
      return data;
    } catch (error) {
      console.error('Failed to fetch coordination:', error);
      throw error;
    }
  };

  const fetchOptimization = async () => {
    try {
      const data = await api.get('/telecom/optimization');
      setOptimization(data);
      return data;
    } catch (error) {
      console.error('Failed to fetch optimization:', error);
      throw error;
    }
  };

  const fetchAgents = async () => {
    try {
      const data = await api.get('/agents/status');
      setAgents(data);
      return data;
    } catch (error) {
      console.error('Failed to fetch agents:', error);
      throw error;
    }
  };

  const fetchMetrics = async (timeWindow = 24) => {
    try {
      const data = await api.get(`/metrics/report?time_window_hours=${timeWindow}`);
      setMetrics(data);
      return data;
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      throw error;
    }
  };

  const predict = async (agentType, data, confidenceThreshold = 0.8) => {
    try {
      const result = await api.post('/agents/predict', {
        agent_type: agentType,
        data: data,
        confidence_threshold: confidenceThreshold,
      });
      return result;
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  };

  const trainAgent = async (agentType, data, validationSplit = 0.2) => {
    try {
      const result = await api.post('/agents/train', {
        agent_type: agentType,
        data: data,
        validation_split: validationSplit,
      });
      return result;
    } catch (error) {
      console.error('Training failed:', error);
      throw error;
    }
  };

  const generateData = async (dataType, nSamples = 1000) => {
    try {
      const result = await api.post(`/data/generate?data_type=${dataType}&n_samples=${nSamples}`);
      return result;
    } catch (error) {
      console.error('Data generation failed:', error);
      throw error;
    }
  };

  const value = {
    // State
    kpis,
    coordination,
    optimization,
    agents,
    metrics,
    loading,
    
    // Actions
    fetchKpis,
    fetchCoordination,
    fetchOptimization,
    fetchAgents,
    fetchMetrics,
    predict,
    trainAgent,
    generateData,
    
    // Setters
    setKpis,
    setCoordination,
    setOptimization,
    setAgents,
    setMetrics,
    setLoading,
  };

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  );
};
