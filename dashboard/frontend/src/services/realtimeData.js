import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

class RealtimeDataService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
    });
    this.subscribers = new Map();
    this.intervals = new Map();
  }

  // Subscribe to real-time data updates
  subscribe(key, callback, interval = 5000) {
    this.subscribers.set(key, callback);
    
    // Start polling for data
    const pollData = async () => {
      try {
        const data = await this.fetchData(key);
        callback(data);
      } catch (error) {
        console.error(`Failed to fetch ${key} data:`, error);
      }
    };

    // Initial fetch
    pollData();
    
    // Set up interval
    const intervalId = setInterval(pollData, interval);
    this.intervals.set(key, intervalId);
  }

  // Unsubscribe from updates
  unsubscribe(key) {
    this.subscribers.delete(key);
    if (this.intervals.has(key)) {
      clearInterval(this.intervals.get(key));
      this.intervals.delete(key);
    }
  }

  // Fetch specific data types
  async fetchData(type) {
    switch (type) {
      case 'health':
        const healthResponse = await this.api.get('/health');
        return healthResponse.data;
      
      case 'kpis':
        const kpisResponse = await this.api.get('/telecom/kpis');
        return kpisResponse.data;
      
      case 'agents':
        const agentsResponse = await this.api.get('/agents/status');
        return agentsResponse.data;
      
      case 'analytics':
        const analyticsResponse = await this.api.get('/telecom/coordination');
        return analyticsResponse.data;
      
      case 'security':
        const securityResponse = await this.api.get('/security/report');
        return securityResponse.data;
      
      case 'data-quality':
        const qualityResponse = await this.api.get('/data/summary');
        return qualityResponse.data;
      
      default:
        throw new Error(`Unknown data type: ${type}`);
    }
  }

  // Get real-time metrics
  async getRealtimeMetrics() {
    try {
      const [health, kpis, agents] = await Promise.all([
        this.fetchData('health'),
        this.fetchData('kpis'),
        this.fetchData('agents')
      ]);

      return {
        health,
        kpis,
        agents,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Failed to fetch real-time metrics:', error);
      throw error;
    }
  }

  // Get AI agent performance data
  async getAgentPerformance() {
    try {
      const response = await this.api.get('/agents/status');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch agent performance:', error);
      throw error;
    }
  }

  // Get business metrics
  async getBusinessMetrics() {
    try {
      const response = await this.api.get('/telecom/coordination');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch business metrics:', error);
      throw error;
    }
  }

  // Get security data
  async getSecurityData() {
    try {
      const response = await this.api.get('/security/report');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch security data:', error);
      throw error;
    }
  }

  // Get data quality metrics
  async getDataQualityMetrics() {
    try {
      const response = await this.api.get('/data/summary');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch data quality metrics:', error);
      throw error;
    }
  }

  // Cleanup all subscriptions
  cleanup() {
    this.subscribers.clear();
    this.intervals.forEach(intervalId => clearInterval(intervalId));
    this.intervals.clear();
  }
}

const realtimeDataService = new RealtimeDataService();
export default realtimeDataService;
