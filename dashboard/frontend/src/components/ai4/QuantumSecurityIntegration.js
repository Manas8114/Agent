// Quantum Security Integration Component
// This component can be imported and used in the main dashboard

import React from 'react';
import QuantumSecurityVisualization from './QuantumSecurityVisualization';

const QuantumSecurityIntegration = ({ data, onRefresh }) => {
  return (
    <div className="quantum-security-integration">
      <QuantumSecurityVisualization 
        data={data} 
        onRefresh={onRefresh}
      />
    </div>
  );
};

export default QuantumSecurityIntegration;




