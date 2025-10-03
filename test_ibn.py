#!/usr/bin/env python3
"""
Test IBN functionality
"""

import sys
sys.path.append('.')

from core.ibn_controller import IBNController, IntentType

def test_ibn():
    print("🧠 Testing Intent-Based Networking (IBN)...")
    
    # Initialize IBN Controller
    ibn = IBNController()
    ibn.start_ibn_mode()
    
    # Create test intents
    intent1 = ibn.create_intent(
        description="Maintain latency <10ms for AR traffic",
        intent_type=IntentType.PERFORMANCE,
        constraints={"max_latency": 10.0},
        priority=1
    )
    
    intent2 = ibn.create_intent(
        description="Optimize energy usage during off-peak hours",
        intent_type=IntentType.ENERGY,
        constraints={"energy_optimization": True},
        priority=2
    )
    
    print("✅ IBN Intents created successfully")
    print(f"Intent 1: {intent1.description}")
    print(f"Intent 2: {intent2.description}")
    
    # Check intent status
    status1 = ibn.get_intent_status(intent1.intent_id)
    status2 = ibn.get_intent_status(intent2.intent_id)
    
    print(f"Intent 1 Status: {status1['status']}")
    print(f"Intent 2 Status: {status2['status']}")
    
    # Get all intents
    all_intents = ibn.get_all_intents()
    print(f"Total active intents: {len(all_intents)}")
    
    # Stop IBN mode
    ibn.stop_ibn_mode()
    
    print("✅ IBN testing completed successfully")
    return True

if __name__ == "__main__":
    test_ibn()
