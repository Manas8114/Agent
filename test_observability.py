#!/usr/bin/env python3
"""
Test Enhanced Observability functionality
"""

import sys
sys.path.append('.')

def test_observability():
    print("📊 Testing Enhanced Observability...")
    
    try:
        from monitoring.ai4_metrics import AI4MetricsCollector
        
        # Initialize AI 4.0 Metrics Collector
        metrics_collector = AI4MetricsCollector()
        
        # Start metrics collection
        metrics_collector.start_metrics_collection()
        
        print("✅ AI 4.0 Metrics Collector initialized")
        
        # Test IBN metrics
        print("\n🧠 Testing IBN metrics...")
        metrics_collector.record_ibn_intent("performance", "active")
        metrics_collector.record_ibn_intent("energy", "active")
        metrics_collector.record_ibn_violation("performance", "high")
        metrics_collector.update_ibn_success_rate("performance", 0.95)
        
        print("✅ IBN metrics recorded")
        
        # Test ZTA metrics
        print("\n🤖 Testing ZTA metrics...")
        metrics_collector.record_zta_pipeline("model_update", "success")
        metrics_collector.record_zta_pipeline("agent_update", "success")
        metrics_collector.record_zta_deployment_success("model_update")
        metrics_collector.record_zta_deployment_failure("agent_update", "validation_failed")
        metrics_collector.record_zta_rollback("pipeline_123")
        
        print("✅ ZTA metrics recorded")
        
        # Test Quantum-Safe Security metrics
        print("\n🔐 Testing Quantum-Safe Security metrics...")
        metrics_collector.record_pqc_signature("dilithium", "level_3")
        metrics_collector.record_pqc_signature("dilithium", "level_5")
        metrics_collector.record_pqc_encryption("kyber", "level_3")
        metrics_collector.record_pqc_verification_latency("dilithium", 0.05)
        metrics_collector.record_pqc_verification_latency("kyber", 0.03)
        
        print("✅ Quantum-Safe Security metrics recorded")
        
        # Test Federation metrics
        print("\n🌐 Testing Federation metrics...")
        metrics_collector.update_federation_nodes("coordinator", "active", 1)
        metrics_collector.update_federation_nodes("participant", "active", 3)
        metrics_collector.record_federation_model_update("node_1", "node_2", "parameter_update")
        metrics_collector.record_federation_model_update("node_2", "node_3", "gradient_update")
        metrics_collector.record_federation_communication_latency("model_update", 0.1)
        metrics_collector.record_federation_communication_latency("aggregation", 0.2)
        
        print("✅ Federation metrics recorded")
        
        # Test Self-Evolving metrics
        print("\n🧬 Testing Self-Evolving metrics...")
        metrics_collector.record_self_evolving_evolution("qos_agent", "architecture", "completed")
        metrics_collector.record_self_evolving_evolution("energy_agent", "hyperparameters", "completed")
        metrics_collector.record_self_evolving_evolution("security_agent", "features", "failed")
        metrics_collector.update_self_evolving_improvement("qos_agent", "architecture", 0.05)
        metrics_collector.update_self_evolving_improvement("energy_agent", "hyperparameters", 0.03)
        
        print("✅ Self-Evolving metrics recorded")
        
        # Get metrics summary
        summary = metrics_collector.get_metrics_summary()
        print(f"\n📊 Metrics Summary:")
        print(f"Collection status: {summary['collection_status']}")
        print(f"Prometheus available: {summary['prometheus_available']}")
        
        # Display IBN metrics
        ibn_metrics = summary['ai4_metrics']['ibn']
        print(f"\n🧠 IBN Metrics:")
        print(f"  Intents total: {ibn_metrics['intents_total']}")
        print(f"  Violations total: {ibn_metrics['violations_total']}")
        print(f"  Success rate: {ibn_metrics['success_rate']}")
        
        # Display ZTA metrics
        zta_metrics = summary['ai4_metrics']['zta']
        print(f"\n🤖 ZTA Metrics:")
        print(f"  Pipelines total: {zta_metrics['pipelines_total']}")
        print(f"  Deployments successful: {zta_metrics['deployments_successful']}")
        print(f"  Deployments failed: {zta_metrics['deployments_failed']}")
        print(f"  Rollbacks total: {zta_metrics['rollbacks_total']}")
        
        # Display Quantum-Safe Security metrics
        qs_metrics = summary['ai4_metrics']['quantum_safe']
        print(f"\n🔐 Quantum-Safe Security Metrics:")
        print(f"  Signatures total: {qs_metrics['signatures_total']}")
        print(f"  Encryptions total: {qs_metrics['encryptions_total']}")
        print(f"  Verification latency: {qs_metrics['verification_latency_ms']}ms")
        
        # Display Federation metrics
        fed_metrics = summary['ai4_metrics']['federation']
        print(f"\n🌐 Federation Metrics:")
        print(f"  Nodes total: {fed_metrics['nodes_total']}")
        print(f"  Model updates total: {fed_metrics['model_updates_total']}")
        print(f"  Communication latency: {fed_metrics['communication_latency_ms']}ms")
        
        # Display Self-Evolving metrics
        se_metrics = summary['ai4_metrics']['self_evolving']
        print(f"\n🧬 Self-Evolving Metrics:")
        print(f"  Evolutions total: {se_metrics['evolutions_total']}")
        print(f"  Improvements total: {se_metrics['improvements_total']}")
        
        # Test Prometheus metrics format
        print(f"\n📈 Testing Prometheus metrics format...")
        prometheus_metrics = metrics_collector.get_prometheus_metrics()
        print(f"Prometheus metrics length: {len(prometheus_metrics)} characters")
        
        # Stop metrics collection
        metrics_collector.stop_metrics_collection()
        
        print("\n✅ Enhanced Observability testing completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in observability test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_observability()
