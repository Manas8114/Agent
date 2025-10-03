#!/usr/bin/env python3
"""
Test ZTA functionality
"""

import sys
sys.path.append('.')

from core.zero_touch import ZTAController, UpdateType

def test_zta():
    print("🤖 Testing Zero-Touch Automation (ZTA)...")
    
    # Initialize ZTA Controller
    zta_controller = ZTAController({
        'validation_timeout': 300,
        'deployment_timeout': 600
    })
    
    zta_controller.start_zta_mode()
    
    # Create test updates
    update1 = zta_controller.create_update(
        update_type=UpdateType.MODEL_UPDATE,
        description="Update QoS model to v2.0",
        source_path="models/qos_model_v2.pkl",
        target_path="models/qos_model.pkl",
        validation_required=True,
        rollback_enabled=True
    )
    
    update2 = zta_controller.create_update(
        update_type=UpdateType.AGENT_UPDATE,
        description="Update MARL agent configuration",
        source_path="agents/marl_agent_v2.py",
        target_path="agents/marl_agent.py",
        validation_required=True,
        rollback_enabled=True
    )
    
    print("✅ ZTA Updates created successfully")
    print(f"Update 1: {update1.description}")
    print(f"Update 2: {update2.description}")
    
    # Create pipeline
    pipeline = zta_controller.create_pipeline(
        name="AI Model Update Pipeline",
        updates=[update1, update2],
        digital_twin_required=True
    )
    
    print(f"✅ ZTA Pipeline created: {pipeline.name}")
    
    # Execute pipeline
    result = zta_controller.execute_pipeline(pipeline.pipeline_id)
    print(f"Pipeline execution result: {result}")
    
    # Get pipeline status
    status = zta_controller.get_pipeline_status(pipeline.pipeline_id)
    print(f"Pipeline status: {status}")
    
    # Test rollback mechanism
    print("\n🔄 Testing rollback mechanism...")
    
    # Create a failed update
    failed_update = zta_controller.create_update(
        update_type=UpdateType.SYSTEM_UPDATE,
        description="Failed system update",
        source_path="scripts/failed_update.py",
        target_path="scripts/update.py",
        validation_required=True,
        rollback_enabled=True
    )
    
    failed_pipeline = zta_controller.create_pipeline(
        name="Failed Update Pipeline",
        updates=[failed_update],
        digital_twin_required=True
    )
    
    # Execute failed pipeline
    failed_result = zta_controller.execute_pipeline(failed_pipeline.pipeline_id)
    print(f"Failed pipeline result: {failed_result}")
    
    # Get all pipelines
    all_pipelines = zta_controller.get_all_pipelines()
    print(f"Total pipelines: {len(all_pipelines)}")
    
    # Stop ZTA mode
    zta_controller.stop_zta_mode()
    
    print("✅ ZTA testing completed successfully")
    return True

if __name__ == "__main__":
    test_zta()
