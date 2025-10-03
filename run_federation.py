#!/usr/bin/env python3
"""
Run Global Multi-Operator Federation
"""

import sys
sys.path.append('.')

from core.global_federation import GlobalFederationManager, FederationRole

def run_federation():
    print('🌐 Starting Global Multi-Operator Federation...')
    
    fed = GlobalFederationManager()
    fed.start_federation_mode()
    
    node1 = fed.join_federation('Operator_A', 'http://operator-a.com', FederationRole.COORDINATOR)
    node2 = fed.join_federation('Operator_B', 'http://operator-b.com', FederationRole.PARTICIPANT)
    node3 = fed.join_federation('Operator_C', 'http://operator-c.com', FederationRole.PARTICIPANT)
    
    model_data = {'parameters': [1, 2, 3, 4, 5], 'accuracy': 0.9}
    update = fed.share_model_update(node1.node_id, model_data, [node2.node_id, node3.node_id])
    
    aggregated = fed.aggregate_models([node1.node_id, node2.node_id, node3.node_id])
    
    print(f'✅ Federation: {len(fed.federation_nodes)} nodes joined')
    print(f'   Coordinator: {node1.operator_name}')
    print(f'   Participants: {node2.operator_name}, {node3.operator_name}')
    print(f'   Model updates: {len(fed.model_updates)} shared')
    print(f'   Aggregation: {aggregated.get("method", "N/A")}')
    
    fed.stop_federation_mode()

if __name__ == "__main__":
    run_federation()
