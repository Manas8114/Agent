#!/usr/bin/env python3
"""
AI Agent Coordinator - Safety & Governance Framework
Handles automated action approval with safety checks, policy enforcement, and audit trails
"""

import redis
import json
import yaml
import time
import uuid
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import requests

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
POLICY_FILE = os.getenv("POLICY_FILE", "policy.yaml")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafetyGovernanceCoordinator:
    """Coordinator for AI agent actions with safety and governance controls"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.policy = self.load_policy()
        self.circuit_breakers = {}  # Track circuit breaker states
        self.rate_limiters = {}    # Track rate limiting
        self.audit_log = []        # In-memory audit log
        
        logger.info("Safety Governance Coordinator initialized")
    
    def load_policy(self) -> Dict[str, Any]:
        """Load policy configuration from YAML file"""
        try:
            with open(POLICY_FILE, 'r') as f:
                policy = yaml.safe_load(f)
            logger.info(f"Policy loaded from {POLICY_FILE}")
            return policy
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            # Return default policy
            return {
                'global': {
                    'auto_mode': False,
                    'default_confidence_threshold': 0.9,
                    'rate_limit': {'per_action': 1, 'per_window_sec': 3600}
                },
                'actions': {},
                'blackout_windows': [],
                'cost_limits': {},
                'validation_metrics': {},
                'operator_controls': {'emergency_stop': True}
            }
    
    def is_in_blackout_window(self) -> bool:
        """Check if current time is in a blackout window"""
        now = datetime.now()
        current_weekday = now.weekday() + 1  # Convert to ISO weekday (1=Monday)
        current_time = now.time()
        
        for window in self.policy.get('blackout_windows', []):
            if current_weekday in window.get('weekdays', []):
                start_time = datetime.strptime(window['start'], '%H:%M').time()
                end_time = datetime.strptime(window['end'], '%H:%M').time()
                
                if start_time <= current_time <= end_time:
                    logger.warning(f"In blackout window: {window['name']}")
                    return True
        
        return False
    
    def is_rate_limited(self, action_name: str) -> bool:
        """Check if action is rate limited"""
        if not action_name or action_name is None:
            return True  # Block None actions
        
        rate_key = f"rate_limit:{action_name}"
        window_sec = self.policy['global']['rate_limit']['per_window_sec']
        limit = self.policy['global']['rate_limit']['per_action']
        
        # Get current count
        current_count = self.redis_client.incr(rate_key)
        
        # Set expiration on first increment
        if current_count == 1:
            self.redis_client.expire(rate_key, window_sec)
        
        is_limited = current_count > limit
        if is_limited:
            logger.warning(f"Rate limit exceeded for action {action_name}: {current_count}/{limit}")
        
        return is_limited
    
    def is_circuit_breaker_open(self, action_name: str) -> bool:
        """Check if circuit breaker is open for an action"""
        if not action_name or action_name is None:
            return True  # Block None actions
            
        circuit_key = f"circuit_breaker:{action_name}"
        circuit_state = self.redis_client.get(circuit_key)
        
        if circuit_state == "open":
            logger.warning(f"Circuit breaker open for action {action_name}")
            return True
        
        return False
    
    def open_circuit_breaker(self, action_name: str):
        """Open circuit breaker for an action"""
        circuit_key = f"circuit_breaker:{action_name}"
        cooldown = self.policy['global']['circuit_breaker']['cooldown_seconds']
        
        self.redis_client.setex(circuit_key, cooldown, "open")
        logger.warning(f"Circuit breaker opened for action {action_name} for {cooldown} seconds")
    
    def increment_failure_count(self, action_name: str):
        """Increment failure count for circuit breaker"""
        failure_key = f"failure_count:{action_name}"
        threshold = self.policy['global']['circuit_breaker']['failure_threshold']
        
        current_failures = self.redis_client.incr(failure_key)
        
        if current_failures >= threshold:
            self.open_circuit_breaker(action_name)
            # Reset failure count
            self.redis_client.delete(failure_key)
    
    def validate_action(self, action_data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate action against policy and safety rules"""
        action_name = action_data.get('action')
        
        # Check if action_name is valid
        if not action_name or action_name is None:
            return False, "Invalid action: action name is None or empty"
        
        # Check global auto mode
        if not self.policy['global'].get('auto_mode', False):
            return False, "Global auto mode disabled"
        
        # Check if action is in blackout window
        if self.is_in_blackout_window():
            blocked_actions = []
            for window in self.policy.get('blackout_windows', []):
                blocked_actions.extend(window.get('actions_blocked', []))
            
            if action_name in blocked_actions:
                return False, f"Action blocked during blackout window"
        
        # Check rate limiting
        if self.is_rate_limited(action_name):
            return False, "Rate limit exceeded"
        
        # Check circuit breaker
        if self.is_circuit_breaker_open(action_name):
            return False, "Circuit breaker open"
        
        # Check action-specific policy
        action_policy = self.policy['actions'].get(action_name, {})
        if not action_policy.get('autofix', False):
            return False, f"Action {action_name} not enabled for auto-fix"
        
        # Check confidence threshold
        confidence = action_data.get('confidence', 0.0)
        required_confidence = action_policy.get('confidence_threshold', 
                                              self.policy['global']['default_confidence_threshold'])
        
        if confidence < required_confidence:
            return False, f"Confidence {confidence:.2f} below threshold {required_confidence:.2f}"
        
        # Check cost limits (if applicable)
        cost_limits = self.policy.get('cost_limits', {}).get(action_name)
        if cost_limits:
            # Simulate cost check (in real system, this would query actual costs)
            estimated_cost = action_data.get('estimated_cost', 0)
            if estimated_cost > cost_limits.get('max_cost_per_hour', float('inf')):
                return False, f"Estimated cost {estimated_cost} exceeds hourly limit"
        
        return True, "Action validated successfully"
    
    def create_execution_plan(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan with rollback strategy"""
        action_name = action_data['action']
        action_policy = self.policy['actions'].get(action_name, {})
        
        execution_id = str(uuid.uuid4())
        
        # Create execution plan
        execution_plan = {
            'id': execution_id,
            'action': action_name,
            'params': action_data.get('params', {}),
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': action_data.get('agent_id'),
            'confidence': action_data.get('confidence', 0.0),
            'canary': action_policy.get('canary', False),
            'canary_params': action_policy.get('canary_params', {}),
            'validation_metrics': self.get_validation_metrics(action_name),
            'rollback_plan': self.create_rollback_plan(action_name, action_data)
        }
        
        # Add cost estimation
        execution_plan['estimated_cost'] = self.estimate_action_cost(action_name, action_data)
        
        return execution_plan
    
    def create_rollback_plan(self, action_name: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create rollback plan for the action"""
        rollback_actions = self.policy.get('rollback', {}).get('rollback_actions', {})
        
        rollback_action = rollback_actions.get(action_name, f"rollback_{action_name}")
        
        return {
            'action': rollback_action,
            'params': action_data.get('params', {}),
            'timeout_seconds': self.policy.get('rollback', {}).get('rollback_timeout', 300),
            'triggers': self.policy.get('rollback', {}).get('rollback_triggers', []),
            'automatic': self.policy.get('rollback', {}).get('automatic_rollback', True)
        }
    
    def get_validation_metrics(self, action_name: str) -> List[str]:
        """Get validation metrics for the action"""
        action_policy = self.policy['actions'].get(action_name, {})
        canary_params = action_policy.get('canary_params', {})
        
        return [canary_params.get('validation_metric', 'success_rate')]
    
    def estimate_action_cost(self, action_name: str, action_data: Dict[str, Any]) -> float:
        """Estimate cost of the action (simplified)"""
        # In real system, this would query actual cost models
        cost_estimates = {
            'bandwidth_reallocation': 10.0,
            'load_balancing': 5.0,
            'qos_upgrade': 2.0,
            'restart_upf': 50.0,
            'scale_upf': 100.0,
            'sleep_mode': -20.0,  # Negative cost (savings)
            'power_adjustment': 5.0,
            'block_suspicious_ue': 1.0,
            'rate_limit_ue': 1.0,
            'reroute_slice': 15.0,
            'capacity_scaling': 200.0
        }
        
        return cost_estimates.get(action_name, 10.0)
    
    def log_audit_event(self, event_type: str, action_data: Dict[str, Any], 
                       result: str, details: str = ""):
        """Log audit event"""
        audit_event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'action': str(action_data.get('action', 'unknown')),
            'agent_id': str(action_data.get('agent_id', 'unknown')),
            'confidence': str(action_data.get('confidence', 0.0)),
            'result': str(result),
            'details': str(details),
            'operator': 'ai_coordinator'
        }
        
        # Store in Redis for persistence
        audit_key = f"audit:{uuid.uuid4()}"
        self.redis_client.hset(audit_key, mapping=audit_event)
        self.redis_client.expire(audit_key, 86400 * self.policy['global']['audit']['retention_days'])
        
        # Also store in memory for immediate access
        self.audit_log.append(audit_event)
        
        logger.info(f"Audit event: {event_type} - {action_data.get('action')} - {result}")
    
    def approve_action(self, action_data: Dict[str, Any]) -> tuple[bool, str, Optional[Dict[str, Any]]]:
        """Approve action with safety checks"""
        try:
            # Validate action
            is_valid, validation_message = self.validate_action(action_data)
            
            if not is_valid:
                self.log_audit_event('action_rejected', action_data, 'validation_failed', validation_message)
                return False, validation_message, None
            
            # Create execution plan
            execution_plan = self.create_execution_plan(action_data)
            
            # Log approval
            self.log_audit_event('action_approved', action_data, 'approved', 
                               f"Execution ID: {execution_plan['id']}")
            
            return True, "Action approved", execution_plan
            
        except Exception as e:
            error_msg = f"Error processing action: {str(e)}"
            logger.error(error_msg)
            self.log_audit_event('action_error', action_data, 'error', error_msg)
            return False, error_msg, None
    
    def handle_operator_command(self, command_data: Dict[str, Any]):
        """Handle operator commands (manual approval, emergency stop, etc.)"""
        command = command_data.get('cmd')
        
        if command == 'emergency_stop':
            # Disable global auto mode
            self.policy['global']['auto_mode'] = False
            logger.critical("EMERGENCY STOP activated - Auto mode disabled")
            self.log_audit_event('emergency_stop', command_data, 'executed', 'Auto mode disabled')
            
        elif command == 'enable_auto_mode':
            self.policy['global']['auto_mode'] = True
            logger.info("Auto mode enabled by operator")
            self.log_audit_event('auto_mode_enabled', command_data, 'executed', 'Auto mode enabled')
            
        elif command == 'manual_approve':
            # Manual approval of an action
            action_data = command_data.get('action_data', {})
            is_approved, message, execution_plan = self.approve_action(action_data)
            
            if is_approved and execution_plan:
                # Publish approved action
                self.redis_client.publish('actions.approved', json.dumps(execution_plan))
                logger.info(f"Manual approval: {action_data.get('action')}")
            
        elif command == 'reset_circuit_breaker':
            action_name = command_data.get('action_name')
            if action_name:
                circuit_key = f"circuit_breaker:{action_name}"
                self.redis_client.delete(circuit_key)
                logger.info(f"Circuit breaker reset for {action_name}")
    
    def run(self):
        """Main coordinator loop"""
        pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe([
            'anomalies.alerts',
            'optimization.commands',
            'operator.commands',
            'actions.feedback'
        ])
        
        logger.info("Safety Governance Coordinator started - listening for actions...")
        
        try:
            for message in pubsub.listen():
                try:
                    data = json.loads(message['data'])
                    channel = message['channel']
                    
                    if channel == 'operator.commands':
                        self.handle_operator_command(data)
                        continue
                    
                    elif channel == 'actions.feedback':
                        # Handle feedback from executed actions
                        self.handle_action_feedback(data)
                        continue
                    
                    # Extract the actual message content
                    action_data = data.get('message', {})
                    
                    # Process action requests
                    is_approved, message_text, execution_plan = self.approve_action(action_data)
                    
                    if is_approved and execution_plan:
                        # Publish approved action
                        self.redis_client.publish('actions.approved', json.dumps(execution_plan))
                        logger.info(f"Action approved: {action_data.get('action')} - {message_text}")
                    else:
                        logger.warning(f"Action rejected: {action_data.get('action')} - {message_text}")
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in message: {message['data']}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Coordinator shutting down...")
        finally:
            pubsub.close()
    
    def handle_action_feedback(self, feedback_data: Dict[str, Any]):
        """Handle feedback from executed actions"""
        action_id = feedback_data.get('action_id')
        success = feedback_data.get('success', False)
        action_name = feedback_data.get('action')
        
        if not success:
            # Increment failure count for circuit breaker
            self.increment_failure_count(action_name)
            logger.warning(f"Action failed: {action_name} (ID: {action_id})")
        else:
            # Reset failure count on success
            failure_key = f"failure_count:{action_name}"
            self.redis_client.delete(failure_key)
        
        # Log feedback
        self.log_audit_event('action_feedback', feedback_data, 
                           'success' if success else 'failure', 
                           f"Action ID: {action_id}")

def main():
    """Main entry point"""
    coordinator = SafetyGovernanceCoordinator()
    coordinator.run()

if __name__ == "__main__":
    main()
