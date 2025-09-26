#!/usr/bin/env python3
"""
Redis Message Bus Integration for Enhanced Telecom System
Provides inter-agent communication and action publishing capabilities
"""

import redis
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import os

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisMessageBus:
    """Redis-based message bus for inter-agent communication"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.subscribers = {}  # Track active subscribers
        self.running = False
        
        logger.info("Redis Message Bus initialized")
    
    async def start(self):
        """Start the message bus"""
        self.running = True
        logger.info("Redis Message Bus started")
    
    async def stop(self):
        """Stop the message bus"""
        self.running = False
        logger.info("Redis Message Bus stopped")
    
    async def publish(self, channel: str, message: Dict[str, Any]):
        """Publish message to channel"""
        if not self.running:
            return
        
        try:
            message_data = {
                "channel": channel,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "publisher": "telecom_system"
            }
            
            # Publish to Redis
            self.redis_client.publish(channel, json.dumps(message_data))
            
            logger.debug(f"Published to {channel}: {message.get('action', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error publishing to {channel}: {e}")
    
    async def subscribe(self, channel: str, callback):
        """Subscribe to channel with callback"""
        if not self.running:
            return
        
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel)
            
            self.subscribers[channel] = pubsub
            
            logger.info(f"Subscribed to channel: {channel}")
            
            # Start listening in background
            asyncio.create_task(self._listen_to_channel(channel, pubsub, callback))
            
        except Exception as e:
            logger.error(f"Error subscribing to {channel}: {e}")
    
    async def _listen_to_channel(self, channel: str, pubsub, callback):
        """Listen to channel messages"""
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await callback(data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in {channel}: {message['data']}")
                    except Exception as e:
                        logger.error(f"Error processing message from {channel}: {e}")
                        
        except Exception as e:
            logger.error(f"Error listening to {channel}: {e}")
        finally:
            pubsub.close()

class EnhancedTelecomMessagePublisher:
    """Publisher for enhanced telecom system actions"""
    
    def __init__(self, message_bus: RedisMessageBus):
        self.message_bus = message_bus
        self.action_counter = 0
        
        logger.info("Enhanced Telecom Message Publisher initialized")
    
    async def publish_qos_anomaly(self, alert: Dict[str, Any]):
        """Publish QoS anomaly alert"""
        action_data = {
            "id": f"qos_action_{self.action_counter}",
            "action": "qos_anomaly_detected",
            "agent_id": alert.get("agent_id"),
            "confidence": alert.get("confidence", 0.8),
            "params": {
                "imsi": alert.get("imsi"),
                "cell_id": alert.get("cell_id"),
                "severity": alert.get("severity"),
                "root_cause": alert.get("root_cause_analysis", {}).get("primary_cause"),
                "user_impact": alert.get("user_impact", {}),
                "recommendations": alert.get("self_healing_recommendations", [])
            },
            "timestamp": alert.get("timestamp"),
            "explain": f"QoS anomaly detected: {alert.get('message', '')}"
        }
        
        self.action_counter += 1
        await self.message_bus.publish("anomalies.alerts", action_data)
    
    async def publish_failure_prediction(self, prediction: Dict[str, Any]):
        """Publish failure prediction"""
        action_data = {
            "id": f"failure_action_{self.action_counter}",
            "action": "failure_prediction",
            "agent_id": prediction.get("agent_id"),
            "confidence": prediction.get("confidence", 0.7),
            "params": {
                "imsi": prediction.get("imsi"),
                "cell_id": prediction.get("cell_id"),
                "failure_probability": prediction.get("failure_probability"),
                "risk_level": prediction.get("risk_level"),
                "component_health": prediction.get("component_health", {}),
                "predictive_alarm": prediction.get("predictive_alarm"),
                "explainability": prediction.get("explainability", {})
            },
            "timestamp": prediction.get("timestamp"),
            "explain": f"Failure prediction: {prediction.get('recommended_action', '')}"
        }
        
        self.action_counter += 1
        await self.message_bus.publish("optimization.commands", action_data)
    
    async def publish_energy_optimization(self, recommendation: Dict[str, Any]):
        """Publish energy optimization recommendation"""
        action_data = {
            "id": f"energy_action_{self.action_counter}",
            "action": recommendation.get("action", "energy_optimization"),
            "agent_id": "energy_optimization_004",
            "confidence": recommendation.get("confidence", 0.8),
            "params": {
                "cell_id": recommendation.get("cell_id"),
                "current_load": recommendation.get("current_load"),
                "energy_savings": recommendation.get("energy_savings"),
                "impact_assessment": recommendation.get("impact_assessment")
            },
            "timestamp": datetime.utcnow().isoformat(),
            "explain": f"Energy optimization: {recommendation.get('action', '')}"
        }
        
        self.action_counter += 1
        await self.message_bus.publish("optimization.commands", action_data)
    
    async def publish_security_threat(self, security_event: Dict[str, Any]):
        """Publish security threat"""
        action_data = {
            "id": f"security_action_{self.action_counter}",
            "action": "security_threat_detected",
            "agent_id": "security_intrusion_005",
            "confidence": 0.9,  # High confidence for security threats
            "params": {
                "imsi": security_event.get("imsi"),
                "event_type": security_event.get("event_type"),
                "threat_level": security_event.get("threat_level"),
                "location": security_event.get("location"),
                "details": security_event.get("details")
            },
            "timestamp": security_event.get("timestamp"),
            "explain": f"Security threat: {security_event.get('event_type', '')}"
        }
        
        self.action_counter += 1
        await self.message_bus.publish("anomalies.alerts", action_data)
    
    async def publish_traffic_forecast(self, forecast: Dict[str, Any]):
        """Publish traffic forecast"""
        action_data = {
            "id": f"traffic_action_{self.action_counter}",
            "action": "traffic_forecast",
            "agent_id": forecast.get("agent_id"),
            "confidence": forecast.get("confidence", 0.7),
            "params": {
                "cell_id": forecast.get("cell_id"),
                "current_throughput": forecast.get("current_throughput"),
                "forecasted_throughput": forecast.get("forecasted_throughput"),
                "current_active_ues": forecast.get("current_active_ues"),
                "forecasted_active_ues": forecast.get("forecasted_active_ues"),
                "capacity_utilization": forecast.get("capacity_utilization"),
                "recommendations": forecast.get("recommendations", [])
            },
            "timestamp": forecast.get("timestamp"),
            "explain": f"Traffic forecast: {forecast.get('trend', 'unknown')} trend"
        }
        
        self.action_counter += 1
        await self.message_bus.publish("optimization.commands", action_data)
    
    async def publish_data_quality_alert(self, quality_alert: Dict[str, Any]):
        """Publish data quality alert"""
        action_data = {
            "id": f"quality_action_{self.action_counter}",
            "action": "data_quality_issue",
            "agent_id": quality_alert.get("agent_id"),
            "confidence": 0.95,  # High confidence for data quality issues
            "params": {
                "event_id": quality_alert.get("event_id"),
                "issues": quality_alert.get("issues", []),
                "severity": quality_alert.get("severity"),
                "recommendations": quality_alert.get("recommendations", [])
            },
            "timestamp": quality_alert.get("timestamp"),
            "explain": f"Data quality issue: {', '.join(quality_alert.get('issues', []))}"
        }
        
        self.action_counter += 1
        await self.message_bus.publish("anomalies.alerts", action_data)

# Integration functions for the enhanced telecom system
async def integrate_message_bus_with_telecom_system(telecom_system):
    """Integrate message bus with the enhanced telecom system"""
    message_bus = RedisMessageBus()
    publisher = EnhancedTelecomMessagePublisher(message_bus)
    
    await message_bus.start()
    
    # Store references in telecom system
    telecom_system.message_bus = message_bus
    telecom_system.message_publisher = publisher
    
    logger.info("Message bus integrated with enhanced telecom system")
    
    return message_bus, publisher

async def publish_agent_output(telecom_system, agent_output: Dict[str, Any], agent_type: str):
    """Publish agent output to appropriate channels"""
    if not hasattr(telecom_system, 'message_publisher'):
        logger.warning("Message publisher not available")
        return
    
    publisher = telecom_system.message_publisher
    
    try:
        if agent_type == "qos_anomaly":
            await publisher.publish_qos_anomaly(agent_output)
        elif agent_type == "failure_prediction":
            await publisher.publish_failure_prediction(agent_output)
        elif agent_type == "energy_optimization":
            await publisher.publish_energy_optimization(agent_output)
        elif agent_type == "security_threat":
            await publisher.publish_security_threat(agent_output)
        elif agent_type == "traffic_forecast":
            await publisher.publish_traffic_forecast(agent_output)
        elif agent_type == "data_quality":
            await publisher.publish_data_quality_alert(agent_output)
        else:
            logger.warning(f"Unknown agent type: {agent_type}")
            
    except Exception as e:
        logger.error(f"Error publishing agent output: {e}")

# Test functions
async def test_message_bus():
    """Test the message bus functionality"""
    message_bus = RedisMessageBus()
    await message_bus.start()
    
    # Test publisher
    publisher = EnhancedTelecomMessagePublisher(message_bus)
    
    # Test QoS anomaly
    test_alert = {
        "agent_id": "test_qos_agent",
        "imsi": "001010000000001",
        "cell_id": "cell_001",
        "severity": "high",
        "confidence": 0.9,
        "message": "Test QoS anomaly",
        "root_cause_analysis": {"primary_cause": "congestion"},
        "user_impact": {"affected_users": 50},
        "self_healing_recommendations": [{"action": "bandwidth_reallocation"}],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await publisher.publish_qos_anomaly(test_alert)
    
    logger.info("Message bus test completed")
    await message_bus.stop()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_message_bus())
