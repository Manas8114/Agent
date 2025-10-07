// Test file to verify components are working
import React from 'react';
import UserExperiencePanel from './components/ai4/UserExperiencePanel';
import YouTubeDemoPanel from './components/ai4/YouTubeDemoPanel';
import QuantumSafePanel from './components/ai4/QuantumSafePanel';

// Test component rendering
const TestComponents = () => {
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
    <div style={{ padding: '20px' }}>
      <h1>Component Test Page</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <h2>User Experience Panel</h2>
        <UserExperiencePanel data={mockData.health} onRefresh={mockRefresh} />
      </div>
      
      <div style={{ marginBottom: '20px' }}>
        <h2>YouTube Demo Panel</h2>
        <YouTubeDemoPanel data={mockData.health} onRefresh={mockRefresh} />
      </div>
      
      <div style={{ marginBottom: '20px' }}>
        <h2>Quantum Security Panel</h2>
        <QuantumSafePanel data={mockData.health} />
      </div>
    </div>
  );
};

export default TestComponents;




