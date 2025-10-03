/**
 * Real-Time Data Service for Telecom AI 4.0 Dashboard
 * Fetches live data from backend endpoints with error handling and retry logic
 */

const API_BASE_URL = 'http://localhost:8000/api/v1';

class RealTimeService {
  constructor() {
    this.pollingInterval = null;
    this.retryCount = 0;
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1 second
    this.isConnected = false;
    this.listeners = new Set();
  }

  /**
   * Add a listener for real-time data updates
   */
  addListener(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  /**
   * Notify all listeners with new data
   */
  notifyListeners(data) {
    this.listeners.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in listener callback:', error);
      }
    });
  }

  /**
   * Make HTTP request with retry logic
   */
  async makeRequest(url, options = {}) {
    const defaultOptions = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      timeout: 10000, // 10 seconds
    };

    const requestOptions = { ...defaultOptions, ...options };

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = await fetch(url, requestOptions);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        this.retryCount = 0;
        this.isConnected = true;
        return { success: true, data, error: null };
      } catch (error) {
        console.warn(`Request attempt ${attempt + 1} failed:`, error.message);
        
        if (attempt === this.maxRetries - 1) {
          this.isConnected = false;
          return { success: false, data: null, error: error.message };
        }
        
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, this.retryDelay * (attempt + 1)));
      }
    }
  }

  /**
   * Fetch all system data from backend
   */
  async fetchAllSystemData() {
    const endpoints = [
      { key: 'health', url: `${API_BASE_URL}/health` },
      { key: 'kpis', url: `${API_BASE_URL}/telecom/kpis` },
      { key: 'ibn', url: `${API_BASE_URL}/telecom/intent` },
      { key: 'zta', url: `${API_BASE_URL}/telecom/zta-status` },
      { key: 'quantum', url: `${API_BASE_URL}/telecom/quantum-status` },
      { key: 'federation', url: `${API_BASE_URL}/telecom/federation` },
      { key: 'selfEvolution', url: `${API_BASE_URL}/telecom/self-evolution` },
      { key: 'observability', url: `${API_BASE_URL}/telecom/observability` }
    ];

    const results = {};
    const errors = {};

    // Fetch all endpoints in parallel
    const promises = endpoints.map(async ({ key, url }) => {
      const result = await this.makeRequest(url);
      if (result.success) {
        results[key] = result.data;
      } else {
        errors[key] = result.error;
        results[key] = null;
      }
    });

    await Promise.all(promises);

    return {
      data: results,
      errors: Object.keys(errors).length > 0 ? errors : null,
      isConnected: this.isConnected,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Start real-time polling
   */
  startPolling(interval = 5000) {
    if (this.pollingInterval) {
      this.stopPolling();
    }

    console.log(`🔄 Starting real-time polling every ${interval}ms`);
    
    this.pollingInterval = setInterval(async () => {
      try {
        const systemData = await this.fetchAllSystemData();
        this.notifyListeners(systemData);
      } catch (error) {
        console.error('Polling error:', error);
        this.notifyListeners({
          data: null,
          errors: { polling: error.message },
          isConnected: false,
          timestamp: new Date().toISOString()
        });
      }
    }, interval);
  }

  /**
   * Stop real-time polling
   */
  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
      console.log('⏹️ Stopped real-time polling');
    }
  }

  /**
   * Get connection status
   */
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      retryCount: this.retryCount,
      hasListeners: this.listeners.size > 0
    };
  }

  /**
   * Test backend connectivity
   */
  async testConnection() {
    try {
      const result = await this.makeRequest(`${API_BASE_URL}/health`);
      return result.success;
    } catch (error) {
      return false;
    }
  }
}

// Create singleton instance
const realTimeService = new RealTimeService();

// Export service instance and utility functions
export default realTimeService;

export const {
  addListener,
  startPolling,
  stopPolling,
  getConnectionStatus,
  testConnection,
  fetchAllSystemData
} = realTimeService;
