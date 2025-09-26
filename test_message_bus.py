#!/usr/bin/env python3
"""
Test script to verify Redis message bus integration
"""

import redis
import json
import time
import threading

def test_message_bus():
    """Test Redis message bus integration"""
    try:
        # Connect to Redis
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        
        # Test connection
        if not r.ping():
            print("❌ Redis connection failed")
            return False
        
        print("✅ Redis connected successfully")
        
        # Check active channels
        channels = r.pubsub_channels()
        print(f"📡 Active channels: {channels}")
        
        # Subscribe to anomalies.alerts channel
        pubsub = r.pubsub()
        pubsub.subscribe('anomalies.alerts')
        
        print("🎧 Listening for messages on 'anomalies.alerts' channel...")
        print("⏳ Waiting 10 seconds for messages...")
        
        # Listen for messages with timeout
        start_time = time.time()
        message_count = 0
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    message_count += 1
                    print(f"📨 Message {message_count}:")
                    print(f"   Channel: {data.get('channel')}")
                    print(f"   Action: {data.get('message', {}).get('action', 'unknown')}")
                    print(f"   Agent: {data.get('message', {}).get('agent_id', 'unknown')}")
                    print(f"   Timestamp: {data.get('timestamp')}")
                    print()
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
            
            # Stop after 10 seconds
            if time.time() - start_time > 10:
                break
        
        pubsub.close()
        
        if message_count > 0:
            print(f"✅ Successfully received {message_count} messages!")
            return True
        else:
            print("⚠️  No messages received in 10 seconds")
            return False
            
    except Exception as e:
        print(f"❌ Error testing message bus: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Redis Message Bus Integration")
    print("=" * 50)
    
    success = test_message_bus()
    
    if success:
        print("\n🎉 Message bus integration is working!")
    else:
        print("\n⚠️  Message bus integration needs attention")
