#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning (MARL) Manager for Telecom AI 3.0
Implements QMIX, MADDPG, and other MARL algorithms for coordinated optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from collections import deque, namedtuple
import random
import uuid
import mlflow
import mlflow.pytorch

# MARL algorithm imports
try:
    from marl_agents import QMIXAgent, MADDPGAgent, IQLAgent
    MARL_AGENTS_AVAILABLE = True
except ImportError:
    MARL_AGENTS_AVAILABLE = False
    print("MARL agents not available. Install with: pip install marl-agents")

class MARLAlgorithm(Enum):
    """MARL algorithm types"""
    QMIX = "qmix"
    MADDPG = "maddpg"
    IQL = "iql"
    VDN = "vdn"
    COMA = "coma"

class AgentRole(Enum):
    """Agent roles in MARL"""
    EDGE = "edge"
    CORE = "core"
    COORDINATOR = "coordinator"

@dataclass
class MARLAgent:
    """MARL agent configuration"""
    agent_id: str
    role: AgentRole
    algorithm: MARLAlgorithm
    state_size: int
    action_size: int
    learning_rate: float = 0.001
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 10000
    batch_size: int = 64
    target_update: int = 10

@dataclass
class MARLExperience:
    """MARL experience tuple"""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    agent_ids: List[str]

class QMIXNetwork(nn.Module):
    """QMIX mixing network for value decomposition"""
    
    def __init__(self, state_size: int, agent_count: int, hidden_size: int = 64):
        super(QMIXNetwork, self).__init__()
        self.state_size = state_size
        self.agent_count = agent_count
        self.hidden_size = hidden_size
        
        # Hypernetworks for mixing weights
        self.hyper_w1 = nn.Linear(state_size, hidden_size * agent_count)
        self.hyper_w2 = nn.Linear(state_size, hidden_size)
        self.hyper_b1 = nn.Linear(state_size, hidden_size)
        self.hyper_b2 = nn.Linear(state_size, 1)
    
    def forward(self, q_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Forward pass through QMIX network"""
        batch_size = q_values.size(0)
        
        # Get mixing weights from hypernetworks
        w1 = torch.abs(self.hyper_w1(states))
        w2 = torch.abs(self.hyper_w2(states))
        b1 = self.hyper_b1(states)
        b2 = self.hyper_b2(states)
        
        # Reshape weights
        w1 = w1.view(batch_size, self.agent_count, self.hidden_size)
        w2 = w2.view(batch_size, self.hidden_size, 1)
        
        # First layer
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1.unsqueeze(1))
        
        # Second layer
        q_total = torch.bmm(hidden, w2) + b2.unsqueeze(1)
        
        return q_total.squeeze(1)

class MADDPGCritic(nn.Module):
    """MADDPG Critic network"""
    
    def __init__(self, state_size: int, action_size: int, agent_count: int, hidden_size: int = 128):
        super(MADDPGCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.agent_count = agent_count
        
        # Input: all states + all actions
        input_size = state_size * agent_count + action_size * agent_count
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network"""
        # Flatten states and actions
        states_flat = states.view(states.size(0), -1)
        actions_flat = actions.view(actions.size(0), -1)
        
        # Concatenate states and actions
        x = torch.cat([states_flat, actions_flat], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class MARLManager:
    """Multi-Agent Reinforcement Learning Manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # MARL configuration
        self.algorithm = MARLAlgorithm(self.config.get('algorithm', 'qmix'))
        self.agent_count = self.config.get('agent_count', 6)
        self.state_size = self.config.get('state_size', 20)
        self.action_size = self.config.get('action_size', 10)
        
        # MARL agents
        self.agents = {}
        self.mixing_network = None
        self.critic_networks = {}
        
        # Training configuration
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.01)
        self.batch_size = self.config.get('batch_size', 64)
        self.memory_size = self.config.get('memory_size', 10000)
        
        # Experience replay
        self.memory = deque(maxlen=self.memory_size)
        
        # Training metrics
        self.training_episodes = 0
        self.training_rewards = []
        self.coordination_scores = []
        
        # MLflow tracking
        self.mlflow_experiment = f"marl_telecom_{self.algorithm.value}"
        mlflow.set_experiment(self.mlflow_experiment)
        
        # Initialize MARL components
        self._initialize_marl_components()
    
    def _initialize_marl_components(self):
        """Initialize MARL components based on algorithm"""
        if self.algorithm == MARLAlgorithm.QMIX:
            self._initialize_qmix()
        elif self.algorithm == MARLAlgorithm.MADDPG:
            self._initialize_maddpg()
        elif self.algorithm == MARLAlgorithm.IQL:
            self._initialize_iql()
        else:
            self.logger.warning(f"Algorithm {self.algorithm.value} not implemented, using QMIX")
            self._initialize_qmix()
    
    def _initialize_qmix(self):
        """Initialize QMIX components"""
        self.logger.info("Initializing QMIX MARL system")
        
        # Create individual agents
        for i in range(self.agent_count):
            agent_id = f"agent_{i}"
            role = AgentRole.EDGE if i < self.agent_count // 2 else AgentRole.CORE
            
            self.agents[agent_id] = {
                'role': role,
                'q_network': self._create_q_network(),
                'target_network': self._create_q_network(),
                'optimizer': None,
                'epsilon': 1.0
            }
            
            # Initialize optimizer
            self.agents[agent_id]['optimizer'] = optim.Adam(
                self.agents[agent_id]['q_network'].parameters(),
                lr=self.learning_rate
            )
        
        # Create mixing network
        self.mixing_network = QMIXNetwork(
            state_size=self.state_size,
            agent_count=self.agent_count
        )
        
        self.mixing_optimizer = optim.Adam(
            self.mixing_network.parameters(),
            lr=self.learning_rate
        )
    
    def _initialize_maddpg(self):
        """Initialize MADDPG components"""
        self.logger.info("Initializing MADDPG MARL system")
        
        # Create individual agents with actor-critic
        for i in range(self.agent_count):
            agent_id = f"agent_{i}"
            role = AgentRole.EDGE if i < self.agent_count // 2 else AgentRole.CORE
            
            self.agents[agent_id] = {
                'role': role,
                'actor': self._create_actor_network(),
                'critic': MADDPGCritic(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    agent_count=self.agent_count
                ),
                'target_actor': self._create_actor_network(),
                'target_critic': MADDPGCritic(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    agent_count=self.agent_count
                ),
                'actor_optimizer': None,
                'critic_optimizer': None,
                'noise': 0.1
            }
            
            # Initialize optimizers
            self.agents[agent_id]['actor_optimizer'] = optim.Adam(
                self.agents[agent_id]['actor'].parameters(),
                lr=self.learning_rate
            )
            self.agents[agent_id]['critic_optimizer'] = optim.Adam(
                self.agents[agent_id]['critic'].parameters(),
                lr=self.learning_rate
            )
    
    def _initialize_iql(self):
        """Initialize Independent Q-Learning"""
        self.logger.info("Initializing IQL MARL system")
        
        # Create independent agents
        for i in range(self.agent_count):
            agent_id = f"agent_{i}"
            role = AgentRole.EDGE if i < self.agent_count // 2 else AgentRole.CORE
            
            self.agents[agent_id] = {
                'role': role,
                'q_network': self._create_q_network(),
                'target_network': self._create_q_network(),
                'optimizer': None,
                'epsilon': 1.0
            }
            
            # Initialize optimizer
            self.agents[agent_id]['optimizer'] = optim.Adam(
                self.agents[agent_id]['q_network'].parameters(),
                lr=self.learning_rate
            )
    
    def _create_q_network(self) -> nn.Module:
        """Create Q-network for individual agents"""
        class QNetwork(nn.Module):
            def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, action_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        return QNetwork(self.state_size, self.action_size)
    
    def _create_actor_network(self) -> nn.Module:
        """Create actor network for MADDPG"""
        class ActorNetwork(nn.Module):
            def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
                super(ActorNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, action_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.tanh(self.fc3(x))  # Tanh for continuous actions
                return x
        
        return ActorNetwork(self.state_size, self.action_size)
    
    def train_episode(self, episode: int) -> Dict[str, float]:
        """Train MARL agents for one episode"""
        # Generate episode data
        episode_data = self._generate_episode_data()
        
        # Store experience
        self._store_experience(episode_data)
        
        # Train agents
        if len(self.memory) >= self.batch_size:
            training_metrics = self._train_agents()
        else:
            training_metrics = {'loss': 0.0, 'coordination_score': 0.0}
        
        # Update metrics
        self.training_episodes += 1
        episode_reward = np.sum(episode_data['rewards'])
        self.training_rewards.append(episode_reward)
        
        # Calculate coordination score
        coordination_score = self._calculate_coordination_score(episode_data)
        self.coordination_scores.append(coordination_score)
        
        # Log to MLflow
        if episode % 10 == 0:
            self._log_mlflow_metrics(episode, training_metrics, coordination_score)
        
        return {
            'episode_reward': episode_reward,
            'coordination_score': coordination_score,
            'training_loss': training_metrics.get('loss', 0.0),
            'episode': episode
        }
    
    def _generate_episode_data(self) -> Dict[str, Any]:
        """Generate episode data for training"""
        episode_length = 100
        states = np.random.random((episode_length, self.agent_count, self.state_size))
        actions = np.random.randint(0, self.action_size, (episode_length, self.agent_count))
        rewards = np.random.random((episode_length, self.agent_count))
        next_states = np.random.random((episode_length, self.agent_count, self.state_size))
        dones = np.random.choice([True, False], (episode_length, self.agent_count), p=[0.1, 0.9])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'agent_ids': [f"agent_{i}" for i in range(self.agent_count)]
        }
    
    def _store_experience(self, episode_data: Dict[str, Any]):
        """Store episode experience in replay buffer"""
        for t in range(len(episode_data['states'])):
            experience = MARLExperience(
                states=episode_data['states'][t],
                actions=episode_data['actions'][t],
                rewards=episode_data['rewards'][t],
                next_states=episode_data['next_states'][t],
                dones=episode_data['dones'][t],
                agent_ids=episode_data['agent_ids']
            )
            self.memory.append(experience)
    
    def _train_agents(self) -> Dict[str, float]:
        """Train MARL agents"""
        if self.algorithm == MARLAlgorithm.QMIX:
            return self._train_qmix()
        elif self.algorithm == MARLAlgorithm.MADDPG:
            return self._train_maddpg()
        elif self.algorithm == MARLAlgorithm.IQL:
            return self._train_iql()
        else:
            return {'loss': 0.0}
    
    def _train_qmix(self) -> Dict[str, float]:
        """Train QMIX agents"""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([exp.states for exp in batch])
        actions = torch.LongTensor([exp.actions for exp in batch])
        rewards = torch.FloatTensor([exp.rewards for exp in batch])
        next_states = torch.FloatTensor([exp.next_states for exp in batch])
        dones = torch.BoolTensor([exp.dones for exp in batch])
        
        # Get Q-values for current states
        q_values = []
        for i, agent_id in enumerate(self.agents.keys()):
            agent_states = states[:, i]
            agent_actions = actions[:, i]
            q_vals = self.agents[agent_id]['q_network'](agent_states)
            q_vals = q_vals.gather(1, agent_actions.unsqueeze(1)).squeeze(1)
            q_values.append(q_vals)
        
        q_values = torch.stack(q_values, dim=1)
        
        # Get target Q-values
        target_q_values = []
        for i, agent_id in enumerate(self.agents.keys()):
            agent_next_states = next_states[:, i]
            target_q_vals = self.agents[agent_id]['target_network'](agent_next_states)
            target_q_vals = target_q_vals.max(1)[0].detach()
            target_q_values.append(target_q_vals)
        
        target_q_values = torch.stack(target_q_values, dim=1)
        target_q_total = target_q_values.sum(dim=1)
        
        # Calculate total reward
        total_rewards = rewards.sum(dim=1)
        
        # Mix Q-values
        global_state = states.mean(dim=1)  # Average state across agents
        q_total = self.mixing_network(q_values, global_state)
        
        # Calculate loss
        target = total_rewards + self.gamma * target_q_total * (~dones.any(dim=1)).float()
        loss = F.mse_loss(q_total, target)
        
        # Backward pass
        self.mixing_optimizer.zero_grad()
        for agent_id in self.agents.keys():
            self.agents[agent_id]['optimizer'].zero_grad()
        
        loss.backward()
        
        self.mixing_optimizer.step()
        for agent_id in self.agents.keys():
            self.agents[agent_id]['optimizer'].step()
        
        # Update target networks
        if self.training_episodes % 10 == 0:
            for agent_id in self.agents.keys():
                self._soft_update(
                    self.agents[agent_id]['q_network'],
                    self.agents[agent_id]['target_network']
                )
        
        return {'loss': loss.item()}
    
    def _train_maddpg(self) -> Dict[str, float]:
        """Train MADDPG agents"""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([exp.states for exp in batch])
        actions = torch.FloatTensor([exp.actions for exp in batch])
        rewards = torch.FloatTensor([exp.rewards for exp in batch])
        next_states = torch.FloatTensor([exp.next_states for exp in batch])
        dones = torch.BoolTensor([exp.dones for exp in batch])
        
        total_loss = 0.0
        
        # Train each agent
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            # Get agent-specific data
            agent_states = states[:, i]
            agent_actions = actions[:, i]
            agent_rewards = rewards[:, i]
            agent_next_states = next_states[:, i]
            agent_dones = dones[:, i]
            
            # Train critic
            target_actions = agent['target_actor'](agent_next_states)
            target_q = agent['target_critic'](next_states, target_actions)
            target_q = agent_rewards + self.gamma * target_q.squeeze() * (~agent_dones).float()
            
            current_q = agent['critic'](states, actions)
            critic_loss = F.mse_loss(current_q.squeeze(), target_q.detach())
            
            agent['critic_optimizer'].zero_grad()
            critic_loss.backward()
            agent['critic_optimizer'].step()
            
            # Train actor
            predicted_actions = agent['actor'](agent_states)
            actor_loss = -agent['critic'](states, predicted_actions).mean()
            
            agent['actor_optimizer'].zero_grad()
            actor_loss.backward()
            agent['actor_optimizer'].step()
            
            total_loss += critic_loss.item() + actor_loss.item()
            
            # Update target networks
            if self.training_episodes % 10 == 0:
                self._soft_update(agent['actor'], agent['target_actor'])
                self._soft_update(agent['critic'], agent['target_critic'])
        
        return {'loss': total_loss / len(self.agents)}
    
    def _train_iql(self) -> Dict[str, float]:
        """Train Independent Q-Learning agents"""
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([exp.states for exp in batch])
        actions = torch.LongTensor([exp.actions for exp in batch])
        rewards = torch.FloatTensor([exp.rewards for exp in batch])
        next_states = torch.FloatTensor([exp.next_states for exp in batch])
        dones = torch.BoolTensor([exp.dones for exp in batch])
        
        total_loss = 0.0
        
        # Train each agent independently
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            # Get agent-specific data
            agent_states = states[:, i]
            agent_actions = actions[:, i]
            agent_rewards = rewards[:, i]
            agent_next_states = next_states[:, i]
            agent_dones = dones[:, i]
            
            # Get current Q-values
            current_q = agent['q_network'](agent_states)
            current_q = current_q.gather(1, agent_actions.unsqueeze(1)).squeeze(1)
            
            # Get target Q-values
            target_q = agent['target_network'](agent_next_states)
            target_q = target_q.max(1)[0].detach()
            target_q = agent_rewards + self.gamma * target_q * (~agent_dones).float()
            
            # Calculate loss
            loss = F.mse_loss(current_q, target_q)
            
            # Backward pass
            agent['optimizer'].zero_grad()
            loss.backward()
            agent['optimizer'].step()
            
            total_loss += loss.item()
            
            # Update target network
            if self.training_episodes % 10 == 0:
                self._soft_update(agent['q_network'], agent['target_network'])
        
        return {'loss': total_loss / len(self.agents)}
    
    def _soft_update(self, local_model: nn.Module, target_model: nn.Module):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def _calculate_coordination_score(self, episode_data: Dict[str, Any]) -> float:
        """Calculate coordination score between agents"""
        # Simple coordination score based on action similarity
        actions = episode_data['actions']
        coordination_scores = []
        
        for t in range(len(actions)):
            agent_actions = actions[t]
            # Calculate similarity between agent actions
            similarity = np.corrcoef(agent_actions)[0, 1] if len(agent_actions) > 1 else 0.0
            coordination_scores.append(similarity)
        
        return np.mean(coordination_scores)
    
    def _log_mlflow_metrics(self, episode: int, training_metrics: Dict[str, float], coordination_score: float):
        """Log metrics to MLflow"""
        with mlflow.start_run(run_name=f"marl_episode_{episode}"):
            mlflow.log_metrics({
                'episode': episode,
                'training_loss': training_metrics.get('loss', 0.0),
                'coordination_score': coordination_score,
                'avg_reward': np.mean(self.training_rewards[-10:]) if self.training_rewards else 0.0,
                'algorithm': self.algorithm.value
            })
    
    def get_agent_actions(self, states: np.ndarray) -> np.ndarray:
        """Get actions from all agents for given states"""
        actions = []
        
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            agent_state = states[i]
            
            if self.algorithm == MARLAlgorithm.QMIX or self.algorithm == MARLAlgorithm.IQL:
                # Q-learning agents
                q_values = agent['q_network'](torch.FloatTensor(agent_state))
                action = q_values.argmax().item()
            elif self.algorithm == MARLAlgorithm.MADDPG:
                # Actor-critic agents
                action = agent['actor'](torch.FloatTensor(agent_state))
                action = action.detach().numpy()
            else:
                action = np.random.randint(0, self.action_size)
            
            actions.append(action)
        
        return np.array(actions)
    
    def get_marL_status(self) -> Dict[str, Any]:
        """Get MARL system status"""
        return {
            'algorithm': self.algorithm.value,
            'agent_count': len(self.agents),
            'training_episodes': self.training_episodes,
            'avg_reward': np.mean(self.training_rewards[-100:]) if self.training_rewards else 0.0,
            'avg_coordination_score': np.mean(self.coordination_scores[-100:]) if self.coordination_scores else 0.0,
            'memory_size': len(self.memory),
            'agents': {
                agent_id: {
                    'role': agent['role'].value,
                    'epsilon': agent.get('epsilon', 0.0)
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    def save_models(self, filepath: str):
        """Save MARL models"""
        checkpoint = {
            'algorithm': self.algorithm.value,
            'agents': {},
            'mixing_network': self.mixing_network.state_dict() if self.mixing_network else None,
            'training_episodes': self.training_episodes,
            'config': self.config
        }
        
        for agent_id, agent in self.agents.items():
            checkpoint['agents'][agent_id] = {
                'q_network': agent['q_network'].state_dict(),
                'target_network': agent['target_network'].state_dict(),
                'role': agent['role'].value
            }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"MARL models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load MARL models"""
        checkpoint = torch.load(filepath)
        
        # Load agent models
        for agent_id, agent_data in checkpoint['agents'].items():
            if agent_id in self.agents:
                self.agents[agent_id]['q_network'].load_state_dict(agent_data['q_network'])
                self.agents[agent_id]['target_network'].load_state_dict(agent_data['target_network'])
        
        # Load mixing network if available
        if checkpoint['mixing_network'] and self.mixing_network:
            self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        
        self.training_episodes = checkpoint['training_episodes']
        self.logger.info(f"MARL models loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test MARL Manager
    print("Testing Multi-Agent Reinforcement Learning (MARL) Manager...")
    
    marl_manager = MARLManager({
        'algorithm': 'qmix',
        'agent_count': 6,
        'state_size': 20,
        'action_size': 10,
        'learning_rate': 0.001
    })
    
    # Train for several episodes
    for episode in range(100):
        metrics = marl_manager.train_episode(episode)
        if episode % 20 == 0:
            print(f"Episode {episode}: Reward={metrics['episode_reward']:.2f}, "
                  f"Coordination={metrics['coordination_score']:.2f}")
    
    # Get status
    status = marl_manager.get_marL_status()
    print(f"MARL Status: {status}")
    
    print("MARL Manager testing completed!")
