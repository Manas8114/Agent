#!/usr/bin/env python3
"""
Test Self-Evolving AI Agents functionality
"""

import sys
sys.path.append('.')

from agents.self_evolving_agents import SelfEvolvingAgentManager, EvolutionType

def test_self_evolving_agents():
    print("🧬 Testing Self-Evolving AI Agents...")
    
    # Initialize Self-Evolving Agent Manager
    evolution_manager = SelfEvolvingAgentManager()
    
    # Start evolution mode
    evolution_manager.start_evolution_mode()
    
    # Create evolution tasks
    task1 = evolution_manager.create_evolution_task(
        agent_id="qos_agent",
        evolution_type=EvolutionType.ARCHITECTURE,
        current_performance=0.85,
        target_performance=0.90
    )
    
    task2 = evolution_manager.create_evolution_task(
        agent_id="energy_agent",
        evolution_type=EvolutionType.HYPERPARAMETERS,
        current_performance=0.80,
        target_performance=0.85
    )
    
    task3 = evolution_manager.create_evolution_task(
        agent_id="security_agent",
        evolution_type=EvolutionType.FEATURES,
        current_performance=0.88,
        target_performance=0.92
    )
    
    task4 = evolution_manager.create_evolution_task(
        agent_id="traffic_agent",
        evolution_type=EvolutionType.ALGORITHM,
        current_performance=0.82,
        target_performance=0.87
    )
    
    print("✅ Evolution tasks created successfully")
    print(f"Task 1: {task1.agent_id} - {task1.evolution_type.value}")
    print(f"Task 2: {task2.agent_id} - {task2.evolution_type.value}")
    print(f"Task 3: {task3.agent_id} - {task3.evolution_type.value}")
    print(f"Task 4: {task4.agent_id} - {task4.evolution_type.value}")
    
    # Execute evolution tasks
    print(f"\n🔄 Executing evolution tasks...")
    
    result1 = evolution_manager.execute_evolution_task(task1.task_id)
    print(f"Architecture evolution result: {result1}")
    
    result2 = evolution_manager.execute_evolution_task(task2.task_id)
    print(f"Hyperparameter evolution result: {result2}")
    
    result3 = evolution_manager.execute_evolution_task(task3.task_id)
    print(f"Feature evolution result: {result3}")
    
    result4 = evolution_manager.execute_evolution_task(task4.task_id)
    print(f"Algorithm evolution result: {result4}")
    
    # Test AutoML and NAS
    print(f"\n🤖 Testing AutoML and NAS...")
    
    # Test Neural Architecture Search
    from agents.self_evolving_agents import NeuralArchitectureSearchEngine
    nas_engine = NeuralArchitectureSearchEngine()
    
    architecture_candidates = nas_engine.generate_architecture_candidates(
        "qos_agent", 0.85, 0.90
    )
    
    print(f"Generated {len(architecture_candidates)} architecture candidates")
    for i, candidate in enumerate(architecture_candidates[:3]):
        print(f"Candidate {i+1}: {candidate.architecture}")
    
    # Test Hyperparameter Optimization
    from agents.self_evolving_agents import HyperparameterOptimizer
    hp_optimizer = HyperparameterOptimizer()
    
    hyperparameter_candidates = hp_optimizer.generate_hyperparameter_candidates(
        "energy_agent", 0.80, 0.85
    )
    
    print(f"Generated {len(hyperparameter_candidates)} hyperparameter candidates")
    for i, candidate in enumerate(hyperparameter_candidates[:3]):
        print(f"Candidate {i+1}: {candidate.hyperparameters}")
    
    # Test AutoML
    from agents.self_evolving_agents import AutoMLEngine
    automl_engine = AutoMLEngine()
    
    feature_candidates = automl_engine.generate_feature_candidates(
        "security_agent", 0.88, 0.92
    )
    
    algorithm_candidates = automl_engine.generate_algorithm_candidates(
        "traffic_agent", 0.82, 0.87
    )
    
    print(f"Generated {len(feature_candidates)} feature candidates")
    print(f"Generated {len(algorithm_candidates)} algorithm candidates")
    
    # Get evolution metrics
    metrics = evolution_manager.get_evolution_metrics()
    print(f"\n📊 Evolution metrics:")
    print(f"Total evolutions: {metrics['self_evolving_agents']['total_evolutions']}")
    print(f"Successful evolutions: {metrics['self_evolving_agents']['successful_evolutions']}")
    print(f"Success rate: {metrics['self_evolving_agents']['success_rate']}")
    print(f"Average improvement: {metrics['self_evolving_agents']['average_improvement']}")
    print(f"Best performance: {metrics['self_evolving_agents']['best_performance']}")
    
    # Test performance tracking
    print(f"\n📈 Testing performance tracking...")
    
    # Simulate performance improvements
    for agent_id in ["qos_agent", "energy_agent", "security_agent", "traffic_agent"]:
        performance_history = evolution_manager.get_agent_performance_history(agent_id)
        print(f"{agent_id} performance history: {len(performance_history)} entries")
    
    # Test MLflow tracking
    print(f"\n📊 Testing MLflow tracking...")
    
    import mlflow
    import mlflow.pytorch
    
    # Simulate MLflow tracking
    with mlflow.start_run(run_name="self_evolving_evolution_test"):
        mlflow.log_metrics({
            'evolution_type': 'architecture',
            'improvement': 0.05,
            'best_performance': 0.90,
            'agent_id': 'qos_agent'
        })
        
        mlflow.log_params({
            'evolution_algorithm': 'nas',
            'target_performance': 0.90,
            'current_performance': 0.85
        })
    
    print("✅ MLflow tracking test completed")
    
    # Test continuous improvement
    print(f"\n🔄 Testing continuous improvement...")
    
    # Simulate multiple evolution cycles
    for cycle in range(3):
        print(f"Evolution cycle {cycle + 1}:")
        
        # Create new evolution task
        new_task = evolution_manager.create_evolution_task(
            agent_id=f"agent_{cycle}",
            evolution_type=EvolutionType.ARCHITECTURE,
            current_performance=0.80 + (cycle * 0.02),
            target_performance=0.85 + (cycle * 0.02)
        )
        
        # Execute evolution
        result = evolution_manager.execute_evolution_task(new_task.task_id)
        print(f"  Cycle {cycle + 1} result: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"  Performance improved: {result.get('improvement', 0):.3f}")
    
    # Stop evolution mode
    evolution_manager.stop_evolution_mode()
    
    print("✅ Self-Evolving AI Agents testing completed successfully")
    return True

if __name__ == "__main__":
    test_self_evolving_agents()
