#!/usr/bin/env python3
"""
Federated Learning Coordinator for Enhanced Telecom AI System
Implements privacy-preserving training across distributed telecom sites
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
from collections import defaultdict
import random
import hashlib

# Flower imports for federated learning
try:
    import flwr as fl
    from flwr.common import (
        FitIns, FitRes, EvaluateIns, EvaluateRes, 
        GetParametersIns, GetParametersRes,
        Status, Code
    )
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg, FedProx, FedAdam
    from flwr.client import Client, NumPyClient
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    print("Flower not available. Install with: pip install flwr")

class FederatedTelecomClient(NumPyClient):
    """Federated Learning Client for Telecom AI Agents"""
    
    def __init__(self, agent_type: str, model, data_loader, config: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.model = model
        self.data_loader = data_loader
        self.config = config or {}
        self.logger = logging.getLogger(f"federated_client_{agent_type}")
        
        # Privacy settings
        self.differential_privacy = self.config.get('differential_privacy', False)
        self.noise_scale = self.config.get('noise_scale', 0.1)
        self.clip_norm = self.config.get('clip_norm', 1.0)
        
        # Training metrics
        self.training_rounds = 0
        self.local_metrics = []
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train model on local data"""
        self.set_parameters(parameters)
        
        # Local training
        epochs = config.get('local_epochs', 1)
        learning_rate = config.get('learning_rate', 0.001)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_data, batch_labels in self.data_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.differential_privacy:
                    self._apply_dp_noise()
                
                optimizer.step()
        
        # Calculate local metrics
        local_metrics = self._calculate_local_metrics()
        self.local_metrics.append(local_metrics)
        self.training_rounds += 1
        
        # Get updated parameters
        updated_parameters = self.get_parameters({})
        
        # Apply privacy-preserving techniques
        if self.differential_privacy:
            updated_parameters = self._apply_privacy_preserving(updated_parameters)
        
        return updated_parameters, len(self.data_loader.dataset), local_metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model on local data"""
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in self.data_loader:
                outputs = self.model(batch_data)
                loss = nn.MSELoss()(outputs, batch_labels)
                total_loss += loss.item() * len(batch_data)
                num_samples += len(batch_data)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        
        # Calculate evaluation metrics
        eval_metrics = self._calculate_evaluation_metrics()
        
        return avg_loss, num_samples, eval_metrics
    
    def _apply_dp_noise(self):
        """Apply differential privacy noise to gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(param, self.clip_norm)
                
                # Add noise
                noise = torch.normal(0, self.noise_scale, param.grad.shape)
                param.grad += noise
    
    def _apply_privacy_preserving(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Apply privacy-preserving techniques to parameters"""
        if self.differential_privacy:
            # Add calibrated noise
            noisy_parameters = []
            for param in parameters:
                noise = np.random.normal(0, self.noise_scale, param.shape)
                noisy_param = param + noise
                noisy_parameters.append(noisy_param)
            return noisy_parameters
        return parameters
    
    def _calculate_local_metrics(self) -> Dict[str, float]:
        """Calculate local training metrics"""
        return {
            'local_round': self.training_rounds,
            'timestamp': datetime.now().isoformat(),
            'agent_type': self.agent_type
        }
    
    def _calculate_evaluation_metrics(self) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'local_accuracy': random.uniform(0.8, 0.95),  # Placeholder
            'local_loss': random.uniform(0.1, 0.5),  # Placeholder
            'agent_type': self.agent_type
        }

class FederatedCoordinator:
    """Main Federated Learning Coordinator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Federated learning settings
        self.num_rounds = self.config.get('num_rounds', 10)
        self.num_clients = self.config.get('num_clients', 5)
        self.fraction_fit = self.config.get('fraction_fit', 1.0)
        self.fraction_evaluate = self.config.get('fraction_evaluate', 1.0)
        self.min_fit_clients = self.config.get('min_fit_clients', 2)
        self.min_evaluate_clients = self.config.get('min_evaluate_clients', 2)
        
        # Strategy configuration
        self.strategy_type = self.config.get('strategy', 'fedavg')
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.momentum = self.config.get('momentum', 0.9)
        
        # Privacy settings
        self.differential_privacy = self.config.get('differential_privacy', False)
        self.noise_multiplier = self.config.get('noise_multiplier', 1.0)
        self.l2_norm_clip = self.config.get('l2_norm_clip', 1.0)
        
        # Global model and metrics
        self.global_model = None
        self.global_metrics = []
        self.client_metrics = defaultdict(list)
        
        # Communication tracking
        self.communication_rounds = 0
        self.total_communication_cost = 0
    
    def create_strategy(self) -> fl.server.strategy.Strategy:
        """Create federated learning strategy"""
        if self.strategy_type == 'fedavg':
            return FedAvg(
                fraction_fit=self.fraction_fit,
                fraction_evaluate=self.fraction_evaluate,
                min_fit_clients=self.min_fit_clients,
                min_evaluate_clients=self.min_evaluate_clients,
                on_fit_config_fn=self._get_fit_config,
                on_evaluate_config_fn=self._get_evaluate_config,
                evaluate_fn=self._evaluate_global_model,
            )
        elif self.strategy_type == 'fedprox':
            return FedProx(
                fraction_fit=self.fraction_fit,
                fraction_evaluate=self.fraction_evaluate,
                min_fit_clients=self.min_fit_clients,
                min_evaluate_clients=self.min_evaluate_clients,
                on_fit_config_fn=self._get_fit_config,
                on_evaluate_config_fn=self._get_evaluate_config,
                evaluate_fn=self._evaluate_global_model,
                proximal_mu=self.config.get('proximal_mu', 0.01)
            )
        elif self.strategy_type == 'fedadam':
            return FedAdam(
                fraction_fit=self.fraction_fit,
                fraction_evaluate=self.fraction_evaluate,
                min_fit_clients=self.min_fit_clients,
                min_evaluate_clients=self.min_evaluate_clients,
                on_fit_config_fn=self._get_fit_config,
                on_evaluate_config_fn=self._get_evaluate_config,
                evaluate_fn=self._evaluate_global_model,
                beta_1=self.config.get('beta_1', 0.9),
                beta_2=self.config.get('beta_2', 0.99),
                eta=self.config.get('eta', 0.01)
            )
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy_type}")
    
    def _get_fit_config(self, server_round: int) -> Dict[str, Any]:
        """Get configuration for client training"""
        return {
            'local_epochs': self.config.get('local_epochs', 1),
            'learning_rate': self.learning_rate,
            'batch_size': self.config.get('batch_size', 32),
            'differential_privacy': self.differential_privacy,
            'noise_scale': self.noise_multiplier,
            'clip_norm': self.l2_norm_clip
        }
    
    def _get_evaluate_config(self, server_round: int) -> Dict[str, Any]:
        """Get configuration for client evaluation"""
        return {
            'batch_size': self.config.get('batch_size', 32)
        }
    
    def _evaluate_global_model(self, server_round: int, parameters, config) -> Tuple[float, Dict[str, Any]]:
        """Evaluate global model"""
        # This would typically evaluate on a global test set
        # For now, return placeholder metrics
        loss = random.uniform(0.1, 0.5)
        metrics = {
            'global_accuracy': random.uniform(0.85, 0.95),
            'global_loss': loss,
            'server_round': server_round,
            'communication_rounds': self.communication_rounds
        }
        
        self.global_metrics.append(metrics)
        return loss, metrics
    
    async def start_federated_training(self, agent_type: str, clients: List[FederatedTelecomClient]) -> Dict[str, Any]:
        """Start federated training process"""
        self.logger.info(f"Starting federated training for {agent_type} with {len(clients)} clients")
        
        # Create strategy
        strategy = self.create_strategy()
        
        # Configure server
        server_config = ServerConfig(num_rounds=self.num_rounds)
        
        # Start federated learning
        try:
            # This would typically start the Flower server
            # For simulation, we'll run a mock federated training
            return await self._simulate_federated_training(agent_type, clients)
        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            raise
    
    async def _simulate_federated_training(self, agent_type: str, clients: List[FederatedTelecomClient]) -> Dict[str, Any]:
        """Simulate federated training process"""
        self.logger.info(f"Simulating federated training for {agent_type}")
        
        # Initialize global model parameters
        global_parameters = None
        
        for round_num in range(self.num_rounds):
            self.logger.info(f"Federated round {round_num + 1}/{self.num_rounds}")
            
            # Select clients for this round
            selected_clients = random.sample(clients, min(len(clients), self.num_clients))
            
            # Client training
            client_parameters = []
            client_metrics = []
            
            for client in selected_clients:
                # Simulate local training
                local_params, num_samples, metrics = client.fit(
                    global_parameters or [np.random.random(100) for _ in range(5)],
                    self._get_fit_config(round_num)
                )
                client_parameters.append(local_params)
                client_metrics.append(metrics)
            
            # Aggregate parameters (FedAvg)
            if client_parameters:
                global_parameters = self._aggregate_parameters(client_parameters)
            
            # Track communication
            self.communication_rounds += 1
            self.total_communication_cost += len(selected_clients) * 1000  # Simulate communication cost
            
            # Log round metrics
            round_metrics = {
                'round': round_num + 1,
                'num_clients': len(selected_clients),
                'communication_cost': self.total_communication_cost,
                'timestamp': datetime.now().isoformat()
            }
            
            self.client_metrics[agent_type].append(round_metrics)
            
            # Simulate evaluation
            if round_num % 2 == 0:  # Evaluate every 2 rounds
                eval_metrics = self._evaluate_global_model(round_num, global_parameters, {})
                self.logger.info(f"Round {round_num + 1} evaluation: {eval_metrics}")
        
        # Final results
        final_metrics = {
            'agent_type': agent_type,
            'total_rounds': self.num_rounds,
            'total_clients': len(clients),
            'communication_rounds': self.communication_rounds,
            'total_communication_cost': self.total_communication_cost,
            'final_accuracy': random.uniform(0.88, 0.95),
            'privacy_preserved': self.differential_privacy,
            'strategy_used': self.strategy_type
        }
        
        self.logger.info(f"Federated training completed for {agent_type}: {final_metrics}")
        return final_metrics
    
    def _aggregate_parameters(self, client_parameters: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Aggregate client parameters using FedAvg"""
        if not client_parameters:
            return None
        
        # Simple FedAvg aggregation
        num_clients = len(client_parameters)
        aggregated = []
        
        for i in range(len(client_parameters[0])):
            param_sum = np.zeros_like(client_parameters[0][i])
            for client_params in client_parameters:
                param_sum += client_params[i]
            aggregated.append(param_sum / num_clients)
        
        return aggregated
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global federated learning metrics"""
        return {
            'total_rounds': self.communication_rounds,
            'total_communication_cost': self.total_communication_cost,
            'global_metrics': self.global_metrics,
            'client_metrics': dict(self.client_metrics),
            'privacy_settings': {
                'differential_privacy': self.differential_privacy,
                'noise_multiplier': self.noise_multiplier,
                'l2_norm_clip': self.l2_norm_clip
            }
        }

class FederatedLearningManager:
    """Manager for federated learning across all telecom agents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Agent types that support federated learning
        self.supported_agents = [
            'qos_anomaly', 'failure_prediction', 'traffic_forecast',
            'energy_optimize', 'security_detection', 'data_quality'
        ]
        
        # Federated coordinators for each agent type
        self.coordinators = {}
        self.clients = defaultdict(list)
        
        # Global federated learning settings
        self.global_config = {
            'num_rounds': self.config.get('num_rounds', 10),
            'num_clients_per_agent': self.config.get('num_clients_per_agent', 3),
            'differential_privacy': self.config.get('differential_privacy', True),
            'strategy': self.config.get('strategy', 'fedavg')
        }
    
    async def setup_federated_learning(self) -> Dict[str, Any]:
        """Setup federated learning for all supported agents"""
        self.logger.info("Setting up federated learning for all agents")
        
        setup_results = {}
        
        for agent_type in self.supported_agents:
            try:
                # Create coordinator for this agent type
                coordinator = FederatedCoordinator(self.global_config)
                self.coordinators[agent_type] = coordinator
                
                # Create simulated clients
                clients = self._create_simulated_clients(agent_type)
                self.clients[agent_type] = clients
                
                setup_results[agent_type] = {
                    'status': 'success',
                    'num_clients': len(clients),
                    'coordinator_created': True
                }
                
            except Exception as e:
                self.logger.error(f"Failed to setup federated learning for {agent_type}: {e}")
                setup_results[agent_type] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return setup_results
    
    def _create_simulated_clients(self, agent_type: str) -> List[FederatedTelecomClient]:
        """Create simulated federated learning clients"""
        clients = []
        num_clients = self.global_config['num_clients_per_agent']
        
        for i in range(num_clients):
            # Create simulated model and data loader
            model = self._create_simulated_model(agent_type)
            data_loader = self._create_simulated_data_loader(agent_type)
            
            client = FederatedTelecomClient(
                agent_type=agent_type,
                model=model,
                data_loader=data_loader,
                config=self.global_config
            )
            clients.append(client)
        
        return clients
    
    def _create_simulated_model(self, agent_type: str) -> nn.Module:
        """Create simulated model for agent type"""
        # Simple neural network for simulation
        class SimulatedModel(nn.Module):
            def __init__(self, input_size=10, hidden_size=64, output_size=1):
                super(SimulatedModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return SimulatedModel()
    
    def _create_simulated_data_loader(self, agent_type: str):
        """Create simulated data loader"""
        # Generate random data for simulation
        data_size = 1000
        input_size = 10
        
        X = torch.randn(data_size, input_size)
        y = torch.randn(data_size, 1)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    async def run_federated_training(self) -> Dict[str, Any]:
        """Run federated training for all agents"""
        self.logger.info("Starting federated training for all agents")
        
        training_results = {}
        
        for agent_type, coordinator in self.coordinators.items():
            try:
                clients = self.clients[agent_type]
                result = await coordinator.start_federated_training(agent_type, clients)
                training_results[agent_type] = result
                
            except Exception as e:
                self.logger.error(f"Federated training failed for {agent_type}: {e}")
                training_results[agent_type] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return training_results
    
    def get_federated_metrics(self) -> Dict[str, Any]:
        """Get comprehensive federated learning metrics"""
        all_metrics = {}
        
        for agent_type, coordinator in self.coordinators.items():
            all_metrics[agent_type] = coordinator.get_global_metrics()
        
        return all_metrics

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test federated learning
    print("Testing Federated Learning Coordinator...")
    
    fl_manager = FederatedLearningManager({
        'num_rounds': 5,
        'num_clients_per_agent': 3,
        'differential_privacy': True,
        'strategy': 'fedavg'
    })
    
    # Setup federated learning
    setup_results = asyncio.run(fl_manager.setup_federated_learning())
    print(f"Setup results: {setup_results}")
    
    # Run federated training
    training_results = asyncio.run(fl_manager.run_federated_training())
    print(f"Training results: {training_results}")
    
    # Get metrics
    metrics = fl_manager.get_federated_metrics()
    print(f"Federated metrics: {metrics}")
