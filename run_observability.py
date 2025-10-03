#!/usr/bin/env python3
"""
Run Enhanced Observability
"""

import sys
sys.path.append('.')

from monitoring.ai4_metrics import AI4MetricsCollector

def run_observability():
    print('📊 Starting Enhanced Observability...')
    
    metrics = AI4MetricsCollector()
    metrics.start_metrics_collection()
    
    metrics.record_ibn_intent('performance', 'active')
    metrics.record_zta_pipeline('model_update', 'success')
    metrics.record_pqc_signature('dilithium', 'level_3')
    metrics.update_federation_nodes('participant', 'active', 3)
    metrics.record_self_evolving_evolution('qos_agent', 'architecture', 'completed')
    
    summary = metrics.get_metrics_summary()
    
    print(f'✅ Observability: Metrics collection active')
    print(f'   IBN intents: {summary["ai4_metrics"]["ibn"]["intents_total"]}')
    print(f'   ZTA pipelines: {summary["ai4_metrics"]["zta"]["pipelines_total"]}')
    print(f'   PQC signatures: {summary["ai4_metrics"]["quantum_safe"]["signatures_total"]}')
    print(f'   Federation nodes: {summary["ai4_metrics"]["federation"]["nodes_total"]}')
    print(f'   Self-evolving: {summary["ai4_metrics"]["self_evolving"]["evolutions_total"]}')
    
    metrics.stop_metrics_collection()

if __name__ == "__main__":
    run_observability()
