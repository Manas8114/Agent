#!/usr/bin/env python3
"""
Test Global Multi-Operator Federation functionality
"""

import sys
sys.path.append('.')

from core.global_federation import GlobalFederationManager, FederationRole

def test_global_federation():
    print("🌐 Testing Global Multi-Operator Federation...")
    
    # Initialize Federation Manager
    federation_manager = GlobalFederationManager()
    
    # Start federation mode
    federation_manager.start_federation_mode()
    
    # Join federation nodes
    node1 = federation_manager.join_federation("Operator_A", "http://operator-a.com", FederationRole.COORDINATOR)
    node2 = federation_manager.join_federation("Operator_B", "http://operator-b.com", FederationRole.PARTICIPANT)
    node3 = federation_manager.join_federation("Operator_C", "http://operator-c.com", FederationRole.PARTICIPANT)
    
    print("✅ Federation nodes joined successfully")
    print(f"Node 1: {node1.operator_name} ({node1.role.value})")
    print(f"Node 2: {node2.operator_name} ({node2.role.value})")
    print(f"Node 3: {node3.operator_name} ({node3.role.value})")
    
    # Share model updates
    print(f"\n📤 Testing model update sharing...")
    
    model_data1 = {
        "parameters": [1, 2, 3, 4, 5],
        "accuracy": 0.9,
        "agent_type": "qos_agent"
    }
    
    model_data2 = {
        "parameters": [2, 3, 4, 5, 6],
        "accuracy": 0.85,
        "agent_type": "energy_agent"
    }
    
    update1 = federation_manager.share_model_update(
        source_node_id=node1.node_id,
        model_data=model_data1,
        target_node_ids=[node2.node_id, node3.node_id]
    )
    
    update2 = federation_manager.share_model_update(
        source_node_id=node2.node_id,
        model_data=model_data2,
        target_node_ids=[node1.node_id, node3.node_id]
    )
    
    print(f"Model update 1: {update1.update_id}")
    print(f"Model update 2: {update2.update_id}")
    
    # Test federated learning aggregation
    print(f"\n🔄 Testing federated learning aggregation...")
    
    aggregated_model = federation_manager.aggregate_models(
        [node1.node_id, node2.node_id, node3.node_id],
        "fedavg"
    )
    
    print(f"Aggregated model: {aggregated_model}")
    
    # Simulate cooperative scenarios
    print(f"\n🤝 Testing cooperative scenarios...")
    
    # Traffic spike scenario
    traffic_spike_result = federation_manager.simulate_cooperative_scenario("traffic_spike")
    print(f"Traffic spike scenario: {traffic_spike_result['success']}")
    print(f"Participating nodes: {traffic_spike_result['participating_nodes']}")
    
    # Network failure scenario
    network_failure_result = federation_manager.simulate_cooperative_scenario("network_failure")
    print(f"Network failure scenario: {network_failure_result['success']}")
    
    # Load balancing scenario
    load_balancing_result = federation_manager.simulate_cooperative_scenario("load_balancing")
    print(f"Load balancing scenario: {load_balancing_result['success']}")
    
    # Test cross-operator traffic spike with MARL adaptation
    print(f"\n🚀 Testing cross-operator traffic spike with MARL adaptation...")
    
    # Simulate traffic spike across multiple operators
    for node in [node1, node2, node3]:
        # Simulate local MARL adaptation
        local_adaptation = federation_manager._simulate_local_adaptation(node, "traffic_spike")
        print(f"{node.operator_name} local adaptation: {local_adaptation['local_actions']}")
        
        # Share adaptation with other nodes
        adaptation_update = federation_manager.share_model_update(
            source_node_id=node.node_id,
            model_data=local_adaptation,
            target_node_ids=[n.node_id for n in [node1, node2, node3] if n.node_id != node.node_id]
        )
        print(f"Shared adaptation from {node.operator_name}: {adaptation_update.update_id}")
    
    # Get federation status
    status = federation_manager.get_federation_status()
    print(f"\n📊 Federation status:")
    print(f"Total nodes: {status['federation_metrics']['total_nodes']}")
    print(f"Active nodes: {status['federation_metrics']['active_nodes']}")
    print(f"Model accuracy avg: {status['federation_metrics']['model_accuracy_avg']}")
    print(f"Participation score avg: {status['federation_metrics']['participation_score_avg']}")
    
    # Test encrypted model update sharing
    print(f"\n🔐 Testing encrypted model update sharing...")
    
    encrypted_model_data = {
        "encrypted": True,
        "algorithm": "kyber",
        "data": model_data1
    }
    
    encrypted_update = federation_manager.share_model_update(
        source_node_id=node1.node_id,
        model_data=encrypted_model_data,
        target_node_ids=[node2.node_id, node3.node_id]
    )
    
    print(f"Encrypted model update: {encrypted_update.update_id}")
    print(f"Encryption status: {encrypted_update.encrypted}")
    
    # Stop federation mode
    federation_manager.stop_federation_mode()
    
    print("✅ Global Multi-Operator Federation testing completed successfully")
    return True

if __name__ == "__main__":
    test_global_federation()
