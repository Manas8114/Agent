#!/usr/bin/env python3
"""
AI Agent Executor - Safe Action Execution with Rollback
Executes approved actions with canary deployments, validation, and automatic rollback
"""

import redis
import json
import os
import subprocess
import time
import datetime
import logging
import requests
import random
from typing import Dict, Any, Optional, List
import threading
import queue

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9090")
CONTROL_API_URL = os.getenv("CONTROL_API_URL", "http://127.0.0.1:5001")
CONTROL_API_KEY = os.getenv("CONTROL_API_KEY", "secret")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafeActionExecutor:
    """Safe executor for AI agent actions with rollback capabilities"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.execution_queue = queue.Queue()
        self.active_executions = {}  # Track active executions
        self.rollback_queue = queue.Queue()
        self.metric_baselines = {}  # Store baseline metrics for validation
        
        logger.info("Safe Action Executor initialized")
    
    def run_command(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        """Execute command safely with timeout"""
        try:
            logger.info(f"Executing command: {command}")
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + "\n" + result.stderr
            
            if success:
                logger.info(f"Command succeeded: {command}")
            else:
                logger.error(f"Command failed: {command} - {output}")
            
            return success, output
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {command}")
            return False, "Command execution timeout"
        except Exception as e:
            logger.error(f"Command error: {command} - {str(e)}")
            return False, str(e)
    
    def call_control_api(self, endpoint: str, data: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Call control API for safe command execution"""
        try:
            headers = {
                'X-API-Key': CONTROL_API_KEY,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{CONTROL_API_URL}/{endpoint}",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {'error': f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Control API error: {e}")
            return False, {'error': str(e)}
    
    def get_metric_baseline(self, metric_name: str) -> float:
        """Get baseline metric value"""
        if metric_name not in self.metric_baselines:
            # Simulate baseline (in real system, query Prometheus)
            self.metric_baselines[metric_name] = {
                'qos_compliance': 0.95,
                'latency': 50.0,
                'cpu_utilization': 0.6,
                'energy_savings': 0.0,
                'signal_quality': -80.0,
                'attack_mitigation': 0.0,
                'throughput': 100.0,
                'capacity_utilization': 0.7
            }.get(metric_name, 0.0)
        
        return self.metric_baselines[metric_name]
    
    def validate_metric(self, metric_name: str, expected_improvement: str = 'increase') -> tuple[bool, float]:
        """Validate metric after action execution"""
        try:
            # Simulate metric query (in real system, query Prometheus)
            current_value = self.get_metric_baseline(metric_name)
            
            # Simulate improvement/degradation
            if expected_improvement == 'increase':
                current_value *= random.uniform(1.1, 1.3)  # 10-30% improvement
            elif expected_improvement == 'decrease':
                current_value *= random.uniform(0.7, 0.9)  # 10-30% improvement (decrease is good for latency)
            
            # Define validation thresholds
            thresholds = {
                'qos_compliance': 0.95,
                'latency': 100.0,
                'cpu_utilization': 0.8,
                'energy_savings': 0.1,
                'signal_quality': -90.0,
                'attack_mitigation': 0.8,
                'throughput': 90.0,
                'capacity_utilization': 0.7
            }
            
            threshold = thresholds.get(metric_name, 0.0)
            
            if expected_improvement == 'increase':
                is_valid = current_value >= threshold
            else:
                is_valid = current_value <= threshold
            
            logger.info(f"Metric validation: {metric_name} = {current_value:.2f}, threshold = {threshold:.2f}, valid = {is_valid}")
            
            return is_valid, current_value
            
        except Exception as e:
            logger.error(f"Metric validation error: {e}")
            return False, 0.0
    
    def execute_canary_action(self, execution_plan: Dict[str, Any]) -> tuple[bool, str]:
        """Execute canary version of the action"""
        action_name = execution_plan['action']
        canary_params = execution_plan.get('canary_params', {})
        
        logger.info(f"Executing canary action: {action_name}")
        
        if action_name == 'scale_upf':
            # Scale by smaller step
            replicas_step = canary_params.get('replicas_step', 1)
            deployment = execution_plan['params'].get('deployment', 'open5gs-upf')
            current_replicas = execution_plan['params'].get('replicas', 3)
            
            canary_replicas = current_replicas + replicas_step
            command = f"kubectl scale deployment {deployment} --replicas={canary_replicas}"
            
            success, output = self.run_command(command)
            if success:
                # Store original state for rollback
                self.active_executions[execution_plan['id']] = {
                    'action': action_name,
                    'original_replicas': current_replicas,
                    'canary_replicas': canary_replicas,
                    'deployment': deployment
                }
            
            return success, output
        
        elif action_name == 'bandwidth_reallocation':
            # Allocate smaller percentage
            bandwidth_pct = canary_params.get('bandwidth_percentage', 0.1)
            command = f"echo 'Canary bandwidth reallocation: {bandwidth_pct * 100}%'"
            return self.run_command(command)
        
        elif action_name == 'load_balancing':
            # Test on small sample
            sample_pct = canary_params.get('sample_pct', 5)
            command = f"echo 'Canary load balancing: {sample_pct}% of traffic'"
            return self.run_command(command)
        
        elif action_name == 'sleep_mode':
            # Short sleep duration
            duration_minutes = canary_params.get('duration_minutes', 5)
            command = f"echo 'Canary sleep mode: {duration_minutes} minutes'"
            return self.run_command(command)
        
        elif action_name == 'power_adjustment':
            # Small power adjustment
            power_step_db = canary_params.get('power_step_db', 1)
            command = f"echo 'Canary power adjustment: {power_step_db} dB'"
            return self.run_command(command)
        
        elif action_name == 'rate_limit_ue':
            # Limit small percentage
            limit_pct = canary_params.get('limit_percentage', 0.5)
            command = f"echo 'Canary rate limiting: {limit_pct * 100}%'"
            return self.run_command(command)
        
        elif action_name == 'reroute_slice':
            # Reroute small sample
            sample_pct = canary_params.get('sample_pct', 10)
            command = f"echo 'Canary slice rerouting: {sample_pct}%'"
            return self.run_command(command)
        
        elif action_name == 'capacity_scaling':
            # Small scale factor
            scale_factor = canary_params.get('scale_factor', 1.2)
            command = f"echo 'Canary capacity scaling: {scale_factor}x'"
            return self.run_command(command)
        
        else:
            # Default canary execution
            command = f"echo 'Canary execution: {action_name}'"
            return self.run_command(command)
    
    def execute_full_action(self, execution_plan: Dict[str, Any]) -> tuple[bool, str]:
        """Execute full action after canary validation"""
        action_name = execution_plan['action']
        
        logger.info(f"Executing full action: {action_name}")
        
        if action_name == 'scale_upf':
            deployment = execution_plan['params'].get('deployment', 'open5gs-upf')
            target_replicas = execution_plan['params'].get('replicas', 3)
            
            command = f"kubectl scale deployment {deployment} --replicas={target_replicas}"
            success, output = self.run_command(command)
            
            if success:
                # Update stored state
                if execution_plan['id'] in self.active_executions:
                    self.active_executions[execution_plan['id']]['target_replicas'] = target_replicas
            
            return success, output
        
        elif action_name == 'restart_upf':
            command = "sudo systemctl restart open5gs-upfd"
            return self.run_command(command)
        
        elif action_name == 'bandwidth_reallocation':
            bandwidth_pct = execution_plan['params'].get('bandwidth_percentage', 0.2)
            command = f"echo 'Full bandwidth reallocation: {bandwidth_pct * 100}%'"
            return self.run_command(command)
        
        elif action_name == 'load_balancing':
            command = "echo 'Full load balancing enabled'"
            return self.run_command(command)
        
        elif action_name == 'sleep_mode':
            command = "echo 'Cell sleep mode activated'"
            return self.run_command(command)
        
        elif action_name == 'power_adjustment':
            power_db = execution_plan['params'].get('power_adjustment_db', 3)
            command = f"echo 'Power adjustment: {power_db} dB'"
            return self.run_command(command)
        
        elif action_name == 'block_suspicious_ue':
            imsi = execution_plan['params'].get('imsi', 'unknown')
            command = f"echo 'Blocking suspicious UE: {imsi}'"
            return self.run_command(command)
        
        elif action_name == 'rate_limit_ue':
            imsi = execution_plan['params'].get('imsi', 'unknown')
            command = f"echo 'Rate limiting UE: {imsi}'"
            return self.run_command(command)
        
        elif action_name == 'reroute_slice':
            slice_id = execution_plan['params'].get('slice_id', 'unknown')
            command = f"echo 'Rerouting slice: {slice_id}'"
            return self.run_command(command)
        
        elif action_name == 'capacity_scaling':
            scale_factor = execution_plan['params'].get('scale_factor', 2.0)
            command = f"echo 'Capacity scaling: {scale_factor}x'"
            return self.run_command(command)
        
        else:
            command = f"echo 'Executing action: {action_name}'"
            return self.run_command(command)
    
    def execute_rollback(self, execution_plan: Dict[str, Any]) -> tuple[bool, str]:
        """Execute rollback for the action"""
        action_name = execution_plan['action']
        rollback_plan = execution_plan.get('rollback_plan', {})
        
        logger.warning(f"Executing rollback for action: {action_name}")
        
        if action_name == 'scale_upf':
            execution_id = execution_plan['id']
            if execution_id in self.active_executions:
                stored_state = self.active_executions[execution_id]
                original_replicas = stored_state.get('original_replicas', 3)
                deployment = stored_state.get('deployment', 'open5gs-upf')
                
                command = f"kubectl scale deployment {deployment} --replicas={original_replicas}"
                success, output = self.run_command(command)
                
                if success:
                    del self.active_executions[execution_id]
                
                return success, output
        
        elif action_name == 'restart_upf':
            command = "sudo systemctl start open5gs-upfd"
            return self.run_command(command)
        
        elif action_name == 'bandwidth_reallocation':
            command = "echo 'Rolling back bandwidth reallocation'"
            return self.run_command(command)
        
        elif action_name == 'load_balancing':
            command = "echo 'Disabling load balancing'"
            return self.run_command(command)
        
        elif action_name == 'sleep_mode':
            command = "echo 'Waking up cell from sleep mode'"
            return self.run_command(command)
        
        elif action_name == 'power_adjustment':
            command = "echo 'Restoring original power settings'"
            return self.run_command(command)
        
        elif action_name == 'block_suspicious_ue':
            imsi = execution_plan['params'].get('imsi', 'unknown')
            command = f"echo 'Unblocking UE: {imsi}'"
            return self.run_command(command)
        
        elif action_name == 'rate_limit_ue':
            imsi = execution_plan['params'].get('imsi', 'unknown')
            command = f"echo 'Removing rate limit for UE: {imsi}'"
            return self.run_command(command)
        
        elif action_name == 'reroute_slice':
            slice_id = execution_plan['params'].get('slice_id', 'unknown')
            command = f"echo 'Restoring original routing for slice: {slice_id}'"
            return self.run_command(command)
        
        elif action_name == 'capacity_scaling':
            command = "echo 'Scaling back to original capacity'"
            return self.run_command(command)
        
        else:
            command = f"echo 'Rolling back action: {action_name}'"
            return self.run_command(command)
    
    def execute_action(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with canary deployment and validation"""
        execution_id = execution_plan['id']
        action_name = execution_plan['action']
        
        logger.info(f"Starting execution: {action_name} (ID: {execution_id})")
        
        try:
            # Phase 1: Canary execution (if enabled)
            if execution_plan.get('canary', False):
                logger.info(f"Executing canary for {action_name}")
                
                canary_success, canary_output = self.execute_canary_action(execution_plan)
                
                if not canary_success:
                    logger.error(f"Canary execution failed: {canary_output}")
                    return {
                        'execution_id': execution_id,
                        'action': action_name,
                        'success': False,
                        'phase': 'canary',
                        'error': canary_output,
                        'timestamp': datetime.datetime.utcnow().isoformat()
                    }
                
                # Wait for canary validation
                hold_seconds = execution_plan.get('canary_params', {}).get('hold_seconds', 30)
                logger.info(f"Waiting {hold_seconds} seconds for canary validation...")
                time.sleep(hold_seconds)
                
                # Validate canary metrics
                validation_metrics = execution_plan.get('validation_metrics', ['success_rate'])
                canary_valid = True
                
                for metric in validation_metrics:
                    is_valid, value = self.validate_metric(metric)
                    if not is_valid:
                        logger.warning(f"Canary validation failed for metric: {metric}")
                        canary_valid = False
                        break
                
                if not canary_valid:
                    logger.error(f"Canary validation failed, rolling back")
                    rollback_success, rollback_output = self.execute_rollback(execution_plan)
                    
                    return {
                        'execution_id': execution_id,
                        'action': action_name,
                        'success': False,
                        'phase': 'canary_validation',
                        'error': 'Canary validation failed',
                        'rollback_success': rollback_success,
                        'rollback_output': rollback_output,
                        'timestamp': datetime.datetime.utcnow().isoformat()
                    }
                
                logger.info(f"Canary validation successful, proceeding to full execution")
            
            # Phase 2: Full execution
            full_success, full_output = self.execute_full_action(execution_plan)
            
            if not full_success:
                logger.error(f"Full execution failed: {full_output}")
                
                # Attempt rollback
                rollback_success, rollback_output = self.execute_rollback(execution_plan)
                
                return {
                    'execution_id': execution_id,
                    'action': action_name,
                    'success': False,
                    'phase': 'full_execution',
                    'error': full_output,
                    'rollback_success': rollback_success,
                    'rollback_output': rollback_output,
                    'timestamp': datetime.datetime.utcnow().isoformat()
                }
            
            # Phase 3: Post-execution validation
            logger.info(f"Validating post-execution metrics for {action_name}")
            validation_metrics = execution_plan.get('validation_metrics', ['success_rate'])
            post_execution_valid = True
            
            for metric in validation_metrics:
                is_valid, value = self.validate_metric(metric)
                if not is_valid:
                    logger.warning(f"Post-execution validation failed for metric: {metric}")
                    post_execution_valid = False
                    break
            
            if not post_execution_valid:
                logger.warning(f"Post-execution validation failed, but action completed")
            
            # Success
            logger.info(f"Action executed successfully: {action_name}")
            
            return {
                'execution_id': execution_id,
                'action': action_name,
                'success': True,
                'phase': 'completed',
                'output': full_output,
                'validation_passed': post_execution_valid,
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            
            # Attempt rollback on error
            try:
                rollback_success, rollback_output = self.execute_rollback(execution_plan)
            except Exception as rollback_error:
                rollback_success = False
                rollback_output = str(rollback_error)
            
            return {
                'execution_id': execution_id,
                'action': action_name,
                'success': False,
                'phase': 'error',
                'error': str(e),
                'rollback_success': rollback_success,
                'rollback_output': rollback_output,
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
    
    def run(self):
        """Main executor loop"""
        pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(['actions.approved'])
        
        logger.info("Safe Action Executor started - listening for approved actions...")
        
        try:
            for message in pubsub.listen():
                try:
                    execution_plan = json.loads(message['data'])
                    
                    # Execute action in separate thread to avoid blocking
                    execution_thread = threading.Thread(
                        target=self._execute_action_thread,
                        args=(execution_plan,)
                    )
                    execution_thread.start()
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in message: {message['data']}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Executor shutting down...")
        finally:
            pubsub.close()
    
    def _execute_action_thread(self, execution_plan: Dict[str, Any]):
        """Execute action in separate thread"""
        try:
            # Execute the action
            result = self.execute_action(execution_plan)
            
            # Publish execution result
            self.redis_client.publish('actions.executed', json.dumps(result))
            
            # Send feedback to coordinator
            feedback = {
                'action_id': execution_plan['id'],
                'action': execution_plan['action'],
                'success': result['success'],
                'timestamp': result['timestamp']
            }
            self.redis_client.publish('actions.feedback', json.dumps(feedback))
            
            logger.info(f"Execution completed: {execution_plan['action']} - Success: {result['success']}")
            
        except Exception as e:
            logger.error(f"Execution thread error: {e}")
            
            # Send error feedback
            error_feedback = {
                'action_id': execution_plan['id'],
                'action': execution_plan['action'],
                'success': False,
                'error': str(e),
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
            self.redis_client.publish('actions.feedback', json.dumps(error_feedback))

def main():
    """Main entry point"""
    executor = SafeActionExecutor()
    executor.run()

if __name__ == "__main__":
    main()
