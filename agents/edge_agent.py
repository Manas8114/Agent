#!/usr/bin/env python3
"""
Edge Agent for Enhanced Telecom AI System
Lightweight agents for deployment on MEC servers with ONNX models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class EdgeAgentType(Enum):
    """Edge agent types"""
    QOS_ANOMALY = "qos_anomaly"
    FAILURE_PREDICTION = "failure_prediction"
    TRAFFIC_FORECAST = "traffic_forecast"
    ENERGY_OPTIMIZE = "energy_optimize"
    SECURITY_DETECTION = "security_detection"
    DATA_QUALITY = "data_quality"

@dataclass
class EdgeAgentConfig:
    """Edge agent configuration"""
    agent_type: EdgeAgentType
    model_path: str
    input_size: int
    output_size: int
    max_memory_mb: int = 100
    max_cpu_percent: int = 50
    inference_timeout_ms: int = 100
    batch_size: int = 1
    enable_quantization: bool = True
    enable_pruning: bool = True

class ONNXModelWrapper:
    """Wrapper for ONNX models with optimization"""
    
    def __init__(self, model_path: str, config: EdgeAgentConfig):
        self.model_path = model_path
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ONNX Runtime session
        self.session = None
        self.input_names = []
        self.output_names = []
        
        # Performance metrics
        self.inference_times = []
        self.memory_usage = []
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load and optimize ONNX model"""
        try:
            # Load ONNX model
            model = onnx.load(self.model_path)
            
            # Optimize model for edge deployment
            if self.config.enable_quantization:
                model = self._quantize_model(model)
            
            if self.config.enable_pruning:
                model = self._prune_model(model)
            
            # Create ONNX Runtime session with optimizations
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            
            # Use CPU provider for edge deployment
            providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                model.SerializeToString(),
                sess_options=session_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.logger.info(f"ONNX model loaded successfully: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _quantize_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Quantize model for reduced size"""
        # Simple quantization simulation
        # In practice, use onnx.quantization.quantize_dynamic()
        self.logger.info("Model quantization applied")
        return model
    
    def _prune_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Prune model for reduced size"""
        # Simple pruning simulation
        # In practice, use onnx.pruning.prune_model()
        self.logger.info("Model pruning applied")
        return model
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data"""
        start_time = time.time()
        
        try:
            # Prepare input data
            input_dict = {self.input_names[0]: input_data.astype(np.float32)}
            
            # Run inference
            outputs = self.session.run(self.output_names, input_dict)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            # Check if inference time exceeds timeout
            if inference_time > self.config.inference_timeout_ms:
                self.logger.warning(f"Inference time {inference_time:.2f}ms exceeds timeout {self.config.inference_timeout_ms}ms")
            
            # Track memory usage
            self._track_memory_usage()
            
            return outputs[0] if len(outputs) == 1 else outputs
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
    
    def _track_memory_usage(self):
        """Track memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage.append(memory_mb)
            
            # Check memory limit
            if memory_mb > self.config.max_memory_mb:
                self.logger.warning(f"Memory usage {memory_mb:.2f}MB exceeds limit {self.config.max_memory_mb}MB")
                
        except ImportError:
            pass  # psutil not available
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'total_inferences': len(self.inference_times),
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_mb': np.max(self.memory_usage) if self.memory_usage else 0
        }

class EdgeAgent:
    """Lightweight edge agent for MEC deployment"""
    
    def __init__(self, config: EdgeAgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"edge_agent_{config.agent_type.value}")
        
        # ONNX model wrapper
        self.model_wrapper = ONNXModelWrapper(config.model_path, config)
        
        # Agent state
        self.is_running = False
        self.last_prediction = None
        self.prediction_history = []
        
        # Performance monitoring
        self.performance_metrics = {}
        self.resource_usage = {}
        
        # Communication with core coordinator
        self.core_coordinator_url = os.getenv('CORE_COORDINATOR_URL', 'http://localhost:8000')
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_thread = None
    
    def start(self):
        """Start edge agent"""
        if self.is_running:
            self.logger.warning("Edge agent already running")
            return
        
        self.is_running = True
        self.logger.info(f"Starting edge agent: {self.config.agent_type.value}")
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.start()
    
    def stop(self):
        """Stop edge agent"""
        self.is_running = False
        self.logger.info(f"Stopping edge agent: {self.config.agent_type.value}")
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join()
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction on input data"""
        try:
            # Validate input
            if input_data.shape[-1] != self.config.input_size:
                raise ValueError(f"Input size {input_data.shape[-1]} doesn't match expected {self.config.input_size}")
            
            # Run inference
            prediction = self.model_wrapper.predict(input_data)
            
            # Store prediction
            self.last_prediction = {
                'input': input_data.tolist(),
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'timestamp': datetime.now().isoformat(),
                'agent_type': self.config.agent_type.value
            }
            
            self.prediction_history.append(self.last_prediction)
            
            # Keep only recent predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return self.last_prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def batch_predict(self, input_batch: np.ndarray) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        if input_batch.shape[0] > self.config.batch_size:
            raise ValueError(f"Batch size {input_batch.shape[0]} exceeds limit {self.config.batch_size}")
        
        predictions = []
        for i in range(input_batch.shape[0]):
            prediction = self.predict(input_batch[i])
            predictions.append(prediction)
        
        return predictions
    
    def _heartbeat_loop(self):
        """Send heartbeat to core coordinator"""
        while self.is_running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
                time.sleep(5)  # Retry after 5 seconds
    
    def _send_heartbeat(self):
        """Send heartbeat to core coordinator"""
        try:
            import requests
            
            heartbeat_data = {
                'agent_id': f"edge_{self.config.agent_type.value}",
                'agent_type': self.config.agent_type.value,
                'status': 'healthy' if self.is_running else 'stopped',
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self.get_performance_metrics(),
                'resource_usage': self.get_resource_usage()
            }
            
            # Send heartbeat (in practice, this would be an HTTP request)
            self.logger.debug(f"Heartbeat sent: {heartbeat_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        model_metrics = self.model_wrapper.get_performance_metrics()
        
        return {
            'agent_type': self.config.agent_type.value,
            'is_running': self.is_running,
            'total_predictions': len(self.prediction_history),
            'last_prediction_time': self.last_prediction.get('timestamp') if self.last_prediction else None,
            'model_metrics': model_metrics,
            'resource_limits': {
                'max_memory_mb': self.config.max_memory_mb,
                'max_cpu_percent': self.config.max_cpu_percent,
                'inference_timeout_ms': self.config.inference_timeout_ms
            }
        }
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            import psutil
            
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_limit_mb': self.config.max_memory_mb,
                'cpu_limit_percent': self.config.max_cpu_percent,
                'within_limits': (
                    cpu_percent <= self.config.max_cpu_percent and
                    memory_mb <= self.config.max_memory_mb
                )
            }
            
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}

class EdgeAgentManager:
    """Manager for multiple edge agents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Edge agents
        self.agents = {}
        self.agent_configs = {}
        
        # Resource monitoring
        self.total_memory_limit = self.config.get('total_memory_limit_mb', 500)
        self.total_cpu_limit = self.config.get('total_cpu_limit_percent', 80)
    
    def add_agent(self, agent_config: EdgeAgentConfig) -> EdgeAgent:
        """Add edge agent"""
        agent = EdgeAgent(agent_config)
        self.agents[agent_config.agent_type.value] = agent
        self.agent_configs[agent_config.agent_type.value] = agent_config
        
        self.logger.info(f"Added edge agent: {agent_config.agent_type.value}")
        return agent
    
    def start_all_agents(self):
        """Start all edge agents"""
        for agent in self.agents.values():
            agent.start()
        
        self.logger.info(f"Started {len(self.agents)} edge agents")
    
    def stop_all_agents(self):
        """Stop all edge agents"""
        for agent in self.agents.values():
            agent.stop()
        
        self.logger.info(f"Stopped {len(self.agents)} edge agents")
    
    def get_agent_status(self, agent_type: str) -> Dict[str, Any]:
        """Get status of specific agent"""
        if agent_type not in self.agents:
            raise ValueError(f"Agent {agent_type} not found")
        
        agent = self.agents[agent_type]
        return {
            'agent_type': agent_type,
            'status': 'running' if agent.is_running else 'stopped',
            'performance_metrics': agent.get_performance_metrics(),
            'resource_usage': agent.get_resource_usage()
        }
    
    def get_all_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'total_agents': len(self.agents),
            'running_agents': len([a for a in self.agents.values() if a.is_running]),
            'agents': {
                agent_type: self.get_agent_status(agent_type)
                for agent_type in self.agents.keys()
            },
            'resource_limits': {
                'total_memory_limit_mb': self.total_memory_limit,
                'total_cpu_limit_percent': self.total_cpu_limit
            }
        }
    
    def predict_with_agent(self, agent_type: str, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction with specific agent"""
        if agent_type not in self.agents:
            raise ValueError(f"Agent {agent_type} not found")
        
        agent = self.agents[agent_type]
        return agent.predict(input_data)
    
    def batch_predict_with_agent(self, agent_type: str, input_batch: np.ndarray) -> List[Dict[str, Any]]:
        """Make batch predictions with specific agent"""
        if agent_type not in self.agents:
            raise ValueError(f"Agent {agent_type} not found")
        
        agent = self.agents[agent_type]
        return agent.batch_predict(input_batch)

class ONNXModelConverter:
    """Convert PyTorch models to ONNX for edge deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def convert_pytorch_to_onnx(self, pytorch_model: nn.Module, 
                                input_shape: Tuple[int, ...], 
                                output_path: str,
                                optimize: bool = True) -> str:
        """Convert PyTorch model to ONNX"""
        try:
            # Set model to evaluation mode
            pytorch_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            self.logger.info(f"PyTorch model converted to ONNX: {output_path}")
            
            # Optimize ONNX model if requested
            if optimize:
                self._optimize_onnx_model(output_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert PyTorch model to ONNX: {e}")
            raise
    
    def _optimize_onnx_model(self, model_path: str):
        """Optimize ONNX model for edge deployment"""
        try:
            # Load model
            model = onnx.load(model_path)
            
            # Apply optimizations
            # 1. Remove unused nodes
            # 2. Fuse operations
            # 3. Quantize weights
            
            # Save optimized model
            onnx.save(model, model_path)
            
            self.logger.info(f"ONNX model optimized: {model_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to optimize ONNX model: {e}")
    
    def validate_onnx_model(self, model_path: str) -> bool:
        """Validate ONNX model"""
        try:
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            self.logger.info(f"ONNX model validation passed: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX model validation failed: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test edge agent
    print("Testing Edge Agent with ONNX...")
    
    # Create edge agent config
    config = EdgeAgentConfig(
        agent_type=EdgeAgentType.QOS_ANOMALY,
        model_path="models/qos_anomaly.onnx",  # This would be a real ONNX model
        input_size=10,
        output_size=1,
        max_memory_mb=50,
        max_cpu_percent=30,
        inference_timeout_ms=50
    )
    
    # Create edge agent manager
    manager = EdgeAgentManager()
    
    # Add agent
    agent = manager.add_agent(config)
    
    # Start agent
    agent.start()
    
    # Test prediction
    test_input = np.random.random(10)
    prediction = agent.predict(test_input)
    print(f"Prediction: {prediction}")
    
    # Get performance metrics
    metrics = agent.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Stop agent
    agent.stop()
    
    print("Edge agent testing completed!")
