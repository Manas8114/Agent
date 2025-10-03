#!/usr/bin/env python3
"""
Deep Reinforcement Learning Agent for Energy Optimization
Implements DQN and PPO algorithms for dynamic power-saving strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import mlflow
import mlflow.pytorch
from collections import deque, namedtuple
import random
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import gym
from gym import spaces

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """Deep Q-Network for energy optimization"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
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

class PPOActor(nn.Module):
    """PPO Actor network for policy"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class PPOCritic(nn.Module):
    """PPO Critic network for value estimation"""
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Experience(*zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class EnergyOptimizationEnv(gym.Env):
    """Custom Gym environment for energy optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super(EnergyOptimizationEnv, self).__init__()
        
        self.config = config or {}
        self.num_gnbs = self.config.get('num_gnbs', 10)
        self.max_users_per_gnb = self.config.get('max_users_per_gnb', 100)
        
        # State space: [gnb_power, gnb_users, gnb_load, time_of_day, energy_price]
        self.state_size = self.num_gnbs * 5
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_size,), dtype=np.float32
        )
        
        # Action space: [sleep, low_power, normal, high_power] for each gNB
        self.action_size = self.num_gnbs * 4
        self.action_space = spaces.MultiDiscrete([4] * self.num_gnbs)
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Initialize random state
        self.state = np.random.random(self.state_size).astype(np.float32)
        self.episode_step = 0
        self.episode_reward = 0
        return self.state
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        self.episode_step += 1
        
        # Simulate environment dynamics
        self.state = self._simulate_environment(action)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward
        
        # Check if episode is done
        done = self.episode_step >= 1000  # Max episode length
        
        info = {
            'energy_saved': self._calculate_energy_saved(action),
            'qos_penalty': self._calculate_qos_penalty(action),
            'episode_reward': self.episode_reward
        }
        
        return self.state, reward, done, info
    
    def _simulate_environment(self, action):
        """Simulate environment response to action"""
        # Add some noise and dynamics
        noise = np.random.normal(0, 0.1, self.state_size)
        new_state = self.state + noise
        new_state = np.clip(new_state, 0, 1)
        return new_state.astype(np.float32)
    
    def _calculate_reward(self, action):
        """Calculate reward based on energy savings vs QoS"""
        energy_saved = self._calculate_energy_saved(action)
        qos_penalty = self._calculate_qos_penalty(action)
        
        # Reward = energy_saved - qos_penalty
        reward = energy_saved - qos_penalty
        return reward
    
    def _calculate_energy_saved(self, action):
        """Calculate energy savings from action"""
        # Convert action to power levels
        power_levels = action / 3.0  # Normalize to [0, 1]
        energy_saved = np.sum(power_levels) * 0.1  # Scale factor
        return energy_saved
    
    def _calculate_qos_penalty(self, action):
        """Calculate QoS penalty from action"""
        # Higher power reduction = higher QoS penalty
        power_reduction = 1.0 - (action / 3.0)
        qos_penalty = np.sum(power_reduction) * 0.05  # Scale factor
        return qos_penalty

class DQNAgent:
    """Deep Q-Network Agent for energy optimization"""
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = self.config.get('lr', 0.001)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.batch_size = self.config.get('batch_size', 64)
        self.memory_size = self.config.get('memory_size', 10000)
        self.target_update = self.config.get('target_update', 10)
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        self.update_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class PPOAgent:
    """Proximal Policy Optimization Agent for energy optimization"""
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.lr = self.config.get('lr', 0.0003)
        self.gamma = self.config.get('gamma', 0.99)
        self.eps_clip = self.config.get('eps_clip', 0.2)
        self.k_epochs = self.config.get('k_epochs', 4)
        self.batch_size = self.config.get('batch_size', 64)
        
        # Networks
        self.actor = PPOActor(state_size, action_size).to(self.device)
        self.critic = PPOCritic(state_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.logger = logging.getLogger(__name__)
    
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action and log probability"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def update(self, states, actions, rewards, log_probs, values, advantages):
        """Update actor and critic networks"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(rewards).to(self.device)
        
        for _ in range(self.k_epochs):
            # Actor update
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

class RLEnergyOptimizer:
    """Main RL Energy Optimizer with MLflow tracking"""
    
    def __init__(self, algorithm: str = 'dqn', config: Dict[str, Any] = None):
        self.algorithm = algorithm.lower()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment
        self.env = EnergyOptimizationEnv(self.config)
        state_size = self.env.state_size
        action_size = self.env.action_size
        
        # Initialize agent
        if self.algorithm == 'dqn':
            self.agent = DQNAgent(state_size, action_size, self.config)
        elif self.algorithm == 'ppo':
            self.agent = PPOAgent(state_size, action_size, self.config)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Training metrics
        self.episode_rewards = []
        self.energy_savings = []
        self.qos_penalties = []
    
    def train(self, episodes: int = 1000, save_interval: int = 100) -> Dict[str, Any]:
        """Train the RL agent"""
        self.logger.info(f"Starting RL training with {self.algorithm.upper()} for {episodes} episodes")
        
        with mlflow.start_run(run_name=f"rl_energy_optimization_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log hyperparameters
            mlflow.log_params({
                'algorithm': self.algorithm,
                'episodes': episodes,
                'state_size': self.env.state_size,
                'action_size': self.env.action_size,
                **self.config
            })
            
            for episode in range(episodes):
                state = self.env.reset()
                episode_reward = 0
                episode_energy_saved = 0
                episode_qos_penalty = 0
                
                for step in range(1000):  # Max steps per episode
                    if self.algorithm == 'dqn':
                        action = self.agent.act(state, training=True)
                        next_state, reward, done, info = self.env.step(action)
                        self.agent.remember(state, action, reward, next_state, done)
                        self.agent.replay()
                    elif self.algorithm == 'ppo':
                        action, log_prob = self.agent.act(state)
                        next_state, reward, done, info = self.env.step(action)
                        # Store for batch update
                        # (In practice, you'd collect a batch before updating)
                    
                    episode_reward += reward
                    episode_energy_saved += info.get('energy_saved', 0)
                    episode_qos_penalty += info.get('qos_penalty', 0)
                    
                    state = next_state
                    if done:
                        break
                
                # Log episode metrics
                self.episode_rewards.append(episode_reward)
                self.energy_savings.append(episode_energy_saved)
                self.qos_penalties.append(episode_qos_penalty)
                
                # Log to MLflow every 10 episodes
                if episode % 10 == 0:
                    mlflow.log_metrics({
                        'episode_reward': episode_reward,
                        'energy_saved': episode_energy_saved,
                        'qos_penalty': episode_qos_penalty,
                        'episode': episode
                    })
                
                # Save model periodically
                if episode % save_interval == 0:
                    model_path = f"models/rl_energy_{self.algorithm}_episode_{episode}.pth"
                    self.agent.save_model(model_path)
                    mlflow.pytorch.log_model(self.agent, f"rl_model_episode_{episode}")
            
            # Final metrics
            final_metrics = {
                'final_avg_reward': np.mean(self.episode_rewards[-100:]),
                'final_avg_energy_saved': np.mean(self.energy_savings[-100:]),
                'final_avg_qos_penalty': np.mean(self.qos_penalties[-100:]),
                'total_episodes': episodes
            }
            
            mlflow.log_metrics(final_metrics)
            mlflow.pytorch.log_model(self.agent, "final_rl_model")
            
            self.logger.info(f"Training completed. Final metrics: {final_metrics}")
            return final_metrics
    
    def predict(self, state: np.ndarray) -> int:
        """Predict optimal action for given state"""
        if self.algorithm == 'dqn':
            return self.agent.act(state, training=False)
        elif self.algorithm == 'ppo':
            action, _ = self.agent.act(state)
            return action
    
    def evaluate(self, test_episodes: int = 100) -> Dict[str, float]:
        """Evaluate the trained agent"""
        test_rewards = []
        test_energy_savings = []
        test_qos_penalties = []
        
        for episode in range(test_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_energy_saved = 0
            episode_qos_penalty = 0
            
            for step in range(1000):
                action = self.predict(state)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_energy_saved += info.get('energy_saved', 0)
                episode_qos_penalty += info.get('qos_penalty', 0)
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
            test_energy_savings.append(episode_energy_saved)
            test_qos_penalties.append(episode_qos_penalty)
        
        evaluation_metrics = {
            'avg_reward': np.mean(test_rewards),
            'std_reward': np.std(test_rewards),
            'avg_energy_saved': np.mean(test_energy_savings),
            'avg_qos_penalty': np.mean(test_qos_penalties),
            'energy_efficiency': np.mean(test_energy_savings) / (np.mean(test_qos_penalties) + 1e-8)
        }
        
        self.logger.info(f"Evaluation metrics: {evaluation_metrics}")
        return evaluation_metrics

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test DQN agent
    print("Testing DQN Energy Optimizer...")
    dqn_optimizer = RLEnergyOptimizer('dqn', {'lr': 0.001, 'gamma': 0.99})
    dqn_metrics = dqn_optimizer.train(episodes=100)
    print(f"DQN Results: {dqn_metrics}")
    
    # Test PPO agent
    print("\nTesting PPO Energy Optimizer...")
    ppo_optimizer = RLEnergyOptimizer('ppo', {'lr': 0.0003, 'gamma': 0.99})
    ppo_metrics = ppo_optimizer.train(episodes=100)
    print(f"PPO Results: {ppo_metrics}")
