import React from 'react';
import UserExperiencePanel from './components/ai4/UserExperiencePanel';
import YouTubeDemoPanel from './components/ai4/YouTubeDemoPanel';
import QuantumSafePanel from './components/ai4/QuantumSafePanel';

const TestPage = () => {
  const mockData = {
    health: {
      status: 'healthy',
      timestamp: new Date().toISOString()
    }
  };

  const mockRefresh = () => {
    console.log('Refresh called');
  };

  return (
    <div style={{ padding: '20px', backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px', color: '#333' }}>
        🚀 Telecom AI 4.0 Component Test
      </h1>
      
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ color: '#666', marginBottom: '15px' }}>🎮 User Experience Panel</h2>
        <div style={{ border: '2px solid #007bff', borderRadius: '8px', overflow: 'hidden' }}>
          <UserExperiencePanel data={mockData.health} onRefresh={mockRefresh} />
        </div>
      </div>
      
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ color: '#666', marginBottom: '15px' }}>📺 YouTube Demo Panel</h2>
        <div style={{ border: '2px solid #28a745', borderRadius: '8px', overflow: 'hidden' }}>
          <YouTubeDemoPanel data={mockData.health} onRefresh={mockRefresh} />
        </div>
      </div>
      
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ color: '#666', marginBottom: '15px' }}>🔐 Quantum Security Panel</h2>
        <div style={{ border: '2px solid #6f42c1', borderRadius: '8px', overflow: 'hidden' }}>
          <QuantumSafePanel data={mockData.health} />
        </div>
      </div>
      
      <div style={{ textAlign: 'center', marginTop: '40px', padding: '20px', backgroundColor: '#e9ecef', borderRadius: '8px' }}>
        <p style={{ margin: 0, color: '#666' }}>
          If you can see all three panels above, the components are working correctly! 🎉
        </p>
      </div>
    </div>
  );
};

export default TestPage;




