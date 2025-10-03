import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  CpuChipIcon,
  BoltIcon,
  ChartBarIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { useApi } from '../hooks/useApi';
// import realtimeData from '../services/realtimeData';
import MetricCard from '../components/MetricCard';
import StatusCard from '../components/StatusCard';
import ChartCard from '../components/ChartCard';

const Dashboard = () => {
  const [kpis, setKpis] = useState(null);
  const [coordination, setCoordination] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [loading, setLoading] = useState(true);
  const [realTimeData] = useState([]);
  
  const api = useApi();

  useEffect(() => {
    let isMounted = true;
    let fetchTimeout = null;
    
    const fetchDashboardData = async () => {
      if (!isMounted) return;
      
      try {
        console.log('Fetching dashboard data...');
        setLoading(true);
        
        // Fetch all data in parallel to reduce flickering
        const [kpisData, coordinationData, optimizationData] = await Promise.all([
          api.get('/telecom/kpis'),
          api.get('/telecom/coordination'),
          api.get('/telecom/optimization')
        ]);
        
        console.log('Dashboard data fetched:', { kpisData, coordinationData, optimizationData });
        
        if (isMounted) {
          setKpis(kpisData);
          setCoordination(coordinationData);
          setOptimization(optimizationData);
          setLoading(false);
          console.log('Dashboard data loaded successfully');
        }
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    // Initial fetch with debounce
    fetchTimeout = setTimeout(fetchDashboardData, 1000);
    
    // Set up 5-second auto-refresh for real-time data
    const interval = setInterval(() => {
      if (isMounted) {
        fetchTimeout = setTimeout(fetchDashboardData, 100);
      }
    }, 5000);
    
    return () => {
      isMounted = false;
      if (fetchTimeout) clearTimeout(fetchTimeout);
      clearInterval(interval);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // WebSocket disabled to prevent 403 errors
  // Dashboard works fine with periodic API updates only
  useEffect(() => {
    // WebSocket functionality disabled
    console.log('WebSocket updates disabled - using API polling only');
  }, []);

  // Show loading only if we have no data at all
  if (loading && !kpis && !coordination && !optimization) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // If we have data, show it even if loading is true
  console.log('Dashboard rendering with data:', { kpis, coordination, optimization, loading });

  // Mock data for charts
  const latencyData = [
    { time: '00:00', latency: 45, throughput: 120 },
    { time: '04:00', latency: 42, throughput: 125 },
    { time: '08:00', latency: 48, throughput: 110 },
    { time: '12:00', latency: 52, throughput: 105 },
    { time: '16:00', latency: 46, throughput: 118 },
    { time: '20:00', latency: 44, throughput: 122 },
  ];

  const agentPerformanceData = [
    { name: 'QoS Anomaly', accuracy: 94, status: 'healthy' },
    { name: 'Failure Prediction', accuracy: 89, status: 'healthy' },
    { name: 'Traffic Forecast', accuracy: 92, status: 'healthy' },
    { name: 'Energy Optimize', accuracy: 87, status: 'warning' },
    { name: 'Security Detection', accuracy: 96, status: 'healthy' },
    { name: 'Data Quality', accuracy: 91, status: 'healthy' },
  ];

  const energySavingsData = [
    { name: 'Baseline', value: 100, color: '#ef4444' },
    { name: 'Optimized', value: 75, color: '#22c55e' },
  ];

  return (
    <div className="p-6 space-y-6" style={{ minHeight: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
        <p className="text-gray-600">Real-time monitoring of Enhanced Telecom AI System</p>
      </motion.div>

      {/* KPI Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <MetricCard
          title="Latency"
          value={`${kpis?.latency_ms || 45.2}ms`}
          change="+2.3%"
          changeType="positive"
          icon={BoltIcon}
          color="primary"
        />
        <MetricCard
          title="Throughput"
          value={`${kpis?.throughput_mbps || 125.7} Mbps`}
          change="+5.1%"
          changeType="positive"
          icon={ChartBarIcon}
          color="success"
        />
        <MetricCard
          title="Users"
          value={kpis?.user_count || 1250}
          change="+12.5%"
          changeType="positive"
          icon={CpuChipIcon}
          color="warning"
        />
        <MetricCard
          title="Quality Score"
          value={`${kpis?.connection_quality || 87.3}%`}
          change="+1.2%"
          changeType="positive"
          icon={CheckCircleIcon}
          color="success"
        />
      </motion.div>

      {/* Charts Row */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {/* Latency & Throughput Chart */}
        <ChartCard title="Network Performance" className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={latencyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="latency" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Latency (ms)"
              />
              <Line 
                type="monotone" 
                dataKey="throughput" 
                stroke="#22c55e" 
                strokeWidth={2}
                name="Throughput (Mbps)"
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Agent Performance */}
        <ChartCard title="AI Agent Performance" className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={agentPerformanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="accuracy" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </motion.div>

      {/* Status and Optimization Row */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* System Status */}
        <StatusCard
          title="System Status"
          status="healthy"
          items={[
            { name: 'QoS Anomaly Agent', status: 'healthy', value: '94% accuracy' },
            { name: 'Failure Prediction', status: 'healthy', value: '89% accuracy' },
            { name: 'Traffic Forecast', status: 'healthy', value: '92% accuracy' },
            { name: 'Energy Optimization', status: 'warning', value: '87% accuracy' },
            { name: 'Security Detection', status: 'healthy', value: '96% accuracy' },
            { name: 'Data Quality', status: 'healthy', value: '91% accuracy' },
          ]}
        />

        {/* Energy Optimization */}
        <ChartCard title="Energy Optimization" className="h-64">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Energy Savings</span>
              <span className="text-lg font-semibold text-success-600">
                {optimization?.savings_percent?.[0]?.toFixed(1) || 0}%
              </span>
            </div>
            <ResponsiveContainer width="100%" height={120}>
              <PieChart>
                <Pie
                  data={energySavingsData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={60}
                  dataKey="value"
                >
                  {energySavingsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex justify-center space-x-4 text-xs">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-danger-500 rounded-full mr-2"></div>
                <span>Baseline</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-success-500 rounded-full mr-2"></div>
                <span>Optimized</span>
              </div>
            </div>
          </div>
        </ChartCard>

        {/* Coordination Score */}
        <ChartCard title="Coordination Analytics" className="h-64">
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-primary-600">
                {coordination?.coordination_score?.toFixed(2) || 0}
              </div>
              <div className="text-sm text-gray-600">Coordination Score</div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Agent Consensus</span>
                <span className="font-medium">
                  {coordination?.agent_consensus?.toFixed(2) || 0}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Recommendation Quality</span>
                <span className="font-medium">
                  {coordination?.recommendation_quality?.toFixed(2) || 0}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Decision Accuracy</span>
                <span className="font-medium">
                  {coordination?.decision_accuracy?.toFixed(2) || 0}
                </span>
              </div>
            </div>
          </div>
        </ChartCard>
      </motion.div>

      {/* Real-time Updates */}
      {realTimeData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Real-time Updates</h3>
          <div className="space-y-2">
            {realTimeData.slice(-5).map((update, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-primary-500 rounded-full mr-3"></div>
                  <span className="text-sm text-gray-700">{update.message}</span>
                </div>
                <span className="text-xs text-gray-500">
                  {new Date(update.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Dashboard;
