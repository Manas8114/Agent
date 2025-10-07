/**
 * Real-Time Service for Telecom AI 4.0
 * Handles real-time data polling and WebSocket connections
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

class RealTimeService {
  constructor() {
    this.pollingInterval = null;
    this.listeners = new Set();
    this.isPolling = false;
    this.connectionStatus = 'disconnected';
    this.lastData = null;
    this.errorCount = 0;
    this.maxRetries = 3;
  }

  /**
   * Add event listener for real-time updates
   */
  addListener(callback) {
    this.listeners.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(callback);
    };
  }

  /**
   * Notify all listeners of data updates
   */
  notifyListeners(data) {
    this.listeners.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in real-time listener:', error);
      }
    });
  }

  /**
   * Start polling for real-time data
   */
  startPolling(interval = 5000) {
    if (this.isPolling) {
      this.stopPolling();
    }

    this.isPolling = true;
    this.pollingInterval = setInterval(async () => {
      try {
        const data = await this.fetchAllSystemData();
        this.notifyListeners(data);
        this.errorCount = 0;
      } catch (error) {
        this.errorCount++;
        console.error(`Real-time polling error (${this.errorCount}/${this.maxRetries}):`, error);
        
        if (this.errorCount >= this.maxRetries) {
          this.connectionStatus = 'error';
          this.notifyListeners({
            isConnected: false,
            error: error.message,
            timestamp: new Date().toISOString()
          });
        }
      }
    }, interval);

    console.log(`🔄 Real-time polling started (${interval}ms interval)`);
  }

  /**
   * Stop polling
   */
  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.isPolling = false;
    console.log('⏹️ Real-time polling stopped');
  }

  /**
   * Fetch all system data
   */
  async fetchAllSystemData() {
    try {
      const [
        health,
        kpis,
        zta,
        quantum,
        federation,
        selfEvolution
      ] = await Promise.all([
        this.fetchHealth(),
        this.fetchKPIs(),
        this.fetchZTA(),
        this.fetchQuantum(),
        this.fetchFederation(),
        this.fetchSelfEvolution()
      ]);

      const data = {
        health,
        kpis,
        zta,
        quantum,
        federation,
        selfEvolution,
        isConnected: true,
        timestamp: new Date().toISOString()
      };

      this.lastData = data;
      this.connectionStatus = 'connected';
      return data;
    } catch (error) {
      this.connectionStatus = 'error';
      throw error;
    }
  }

  /**
   * Fetch system health
   */
  async fetchHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
    return await response.json();
  }

  /**
   * Fetch KPIs
   */
  async fetchKPIs() {
    const response = await fetch(`${API_BASE_URL}/telecom/kpis`);
    if (!response.ok) throw new Error(`KPIs fetch failed: ${response.status}`);
    return await response.json();
  }

  /**
   * Fetch ZTA status
   */
  async fetchZTA() {
    const response = await fetch(`${API_BASE_URL}/telecom/zta-status`);
    if (!response.ok) throw new Error(`ZTA fetch failed: ${response.status}`);
    return await response.json();
  }

  /**
   * Fetch quantum security status
   */
  async fetchQuantum() {
    const response = await fetch(`${API_BASE_URL}/telecom/quantum-status`);
    if (!response.ok) throw new Error(`Quantum status fetch failed: ${response.status}`);
    return await response.json();
  }

  /**
   * Fetch federation data
   */
  async fetchFederation() {
    const response = await fetch(`${API_BASE_URL}/telecom/federation`);
    if (!response.ok) throw new Error(`Federation fetch failed: ${response.status}`);
    return await response.json();
  }

  /**
   * Fetch self-evolution data
   */
  async fetchSelfEvolution() {
    const response = await fetch(`${API_BASE_URL}/telecom/self-evolution`);
    if (!response.ok) throw new Error(`Self-evolution fetch failed: ${response.status}`);
    return await response.json();
  }

  /**
   * Test backend connection
   */
  async testConnection() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        timeout: 5000
      });
      this.connectionStatus = response.ok ? 'connected' : 'error';
      return response.ok;
    } catch (error) {
      this.connectionStatus = 'error';
      return false;
    }
  }

  /**
   * Get connection status
   */
  getConnectionStatus() {
    return {
      status: this.connectionStatus,
      isPolling: this.isPolling,
      lastData: this.lastData,
      errorCount: this.errorCount
    };
  }
}

// Export singleton instance
export default new RealTimeService();

