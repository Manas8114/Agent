"""
AI Coordinator for Enhanced Telecom AI System

This module provides the central coordinator that orchestrates all AI agents,
manages their interactions, and provides unified decision-making capabilities.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents import (
    QoSAnomalyAgent, FailurePredictionAgent, TrafficForecastAgent,
    EnergyOptimizeAgent, SecurityDetectionAgent, DataQualityAgent
)

logger = logging.getLogger(__name__)

class AICoordinator:
    """
    Central coordinator for all AI agents in the Enhanced Telecom AI System.
    
    Manages agent lifecycle, coordinates predictions, and provides unified
    decision-making capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the AI Coordinator.
        
        Args:
            config: Configuration dictionary for the coordinator
        """
        self.config = config or {}
        self.agents = {}
        self.agent_weights = {
            'qos_anomaly': 0.2,
            'failure_prediction': 0.2,
            'traffic_forecast': 0.15,
            'energy_optimize': 0.15,
            'security_detection': 0.2,
            'data_quality': 0.1
        }
        self.coordination_history = []
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info("AI Coordinator initialized with 6 agents")
    
    def _initialize_agents(self):
        """Initialize all AI agents."""
        try:
            self.agents['qos_anomaly'] = QoSAnomalyAgent()
            self.agents['failure_prediction'] = FailurePredictionAgent()
            self.agents['traffic_forecast'] = TrafficForecastAgent()
            self.agents['energy_optimize'] = EnergyOptimizeAgent()
            self.agents['security_detection'] = SecurityDetectionAgent()
            self.agents['data_quality'] = DataQualityAgent()
            
            logger.info("All AI agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def train_all_agents(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Train all agents with their respective data.
        
        Args:
            training_data: Dictionary mapping agent names to training data
            
        Returns:
            Training results for all agents
        """
        logger.info("Starting training for all agents...")
        
        training_results = {}
        training_tasks = []
        
        # Create training tasks for each agent
        for agent_name, agent in self.agents.items():
            if agent_name in training_data:
                task = asyncio.create_task(
                    self._train_agent_async(agent_name, agent, training_data[agent_name])
                )
                training_tasks.append((agent_name, task))
        
        # Wait for all training to complete
        for agent_name, task in training_tasks:
            try:
                result = await task
                training_results[agent_name] = result
                logger.info(f"Agent {agent_name} training completed")
            except Exception as e:
                logger.error(f"Agent {agent_name} training failed: {e}")
                training_results[agent_name] = {'error': str(e)}
        
        logger.info("All agents training completed")
        return training_results
    
    async def _train_agent_async(self, agent_name: str, agent: Any, data: pd.DataFrame) -> Dict[str, float]:
        """
        Train a single agent asynchronously.
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            data: Training data
            
        Returns:
            Training metrics
        """
        def train_agent():
            return agent.train(data)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, train_agent)
        return result
    
    async def coordinate_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Coordinate predictions from all agents and provide unified recommendations.
        
        Args:
            data: Input data for predictions
            
        Returns:
            Coordinated predictions and recommendations
        """
        logger.info("Starting coordinated predictions...")
        
        # Get predictions from all agents
        agent_predictions = {}
        agent_recommendations = {}
        
        prediction_tasks = []
        for agent_name, agent in self.agents.items():
            if agent.is_trained:
                task = asyncio.create_task(
                    self._predict_agent_async(agent_name, agent, data)
                )
                prediction_tasks.append((agent_name, task))
        
        # Collect predictions
        for agent_name, task in prediction_tasks:
            try:
                predictions, recommendations = await task
                agent_predictions[agent_name] = predictions
                agent_recommendations[agent_name] = recommendations
            except Exception as e:
                logger.error(f"Agent {agent_name} prediction failed: {e}")
                agent_predictions[agent_name] = None
                agent_recommendations[agent_name] = []
        
        # Generate coordinated recommendations
        coordinated_recommendations = self._generate_coordinated_recommendations(
            agent_predictions, agent_recommendations
        )
        
        # Calculate coordination score
        coordination_score = self._calculate_coordination_score(agent_predictions)
        
        # Store coordination history
        coordination_event = {
            'timestamp': datetime.now().isoformat(),
            'agent_predictions': {k: v.tolist() if v is not None else None 
                                for k, v in agent_predictions.items()},
            'coordination_score': coordination_score,
            'recommendations': coordinated_recommendations
        }
        self.coordination_history.append(coordination_event)
        
        return {
            'agent_predictions': agent_predictions,
            'agent_recommendations': agent_recommendations,
            'coordinated_recommendations': coordinated_recommendations,
            'coordination_score': coordination_score,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _predict_agent_async(self, agent_name: str, agent: Any, data: pd.DataFrame) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get predictions from a single agent asynchronously.
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            data: Input data
            
        Returns:
            Tuple of (predictions, recommendations)
        """
        def predict_agent():
            predictions = agent.predict(data)
            recommendations = agent.get_recommendations(predictions)
            return predictions, recommendations
        
        loop = asyncio.get_event_loop()
        predictions, recommendations = await loop.run_in_executor(self.executor, predict_agent)
        return predictions, recommendations
    
    def _generate_coordinated_recommendations(self, agent_predictions: Dict[str, np.ndarray], 
                                            agent_recommendations: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Generate coordinated recommendations from all agents.
        
        Args:
            agent_predictions: Predictions from all agents
            agent_recommendations: Recommendations from all agents
            
        Returns:
            Coordinated recommendations
        """
        coordinated_recommendations = []
        
        # Analyze agent consensus
        consensus_analysis = self._analyze_agent_consensus(agent_predictions)
        
        # Generate priority-based recommendations
        all_recommendations = []
        for agent_name, recommendations in agent_recommendations.items():
            for rec in recommendations:
                rec['source_agent'] = agent_name
                rec['weight'] = self.agent_weights.get(agent_name, 0.1)
                all_recommendations.append(rec)
        
        # Sort by priority and weight
        all_recommendations.sort(key=lambda x: (
            x.get('priority', 'low') == 'high',
            x.get('weight', 0),
            x.get('confidence', 0)
        ), reverse=True)
        
        # Group similar recommendations
        grouped_recommendations = self._group_similar_recommendations(all_recommendations)
        
        # Generate coordinated actions
        for group in grouped_recommendations:
            coordinated_recommendations.append({
                'type': 'coordinated_action',
                'priority': self._determine_priority(group),
                'actions': self._merge_actions(group),
                'source_agents': [rec['source_agent'] for rec in group],
                'confidence': np.mean([rec.get('confidence', 0.5) for rec in group]),
                'estimated_impact': self._estimate_impact(group),
                'implementation_order': self._determine_implementation_order(group)
            })
        
        return coordinated_recommendations
    
    def _analyze_agent_consensus(self, agent_predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze consensus between agents.
        
        Args:
            agent_predictions: Predictions from all agents
            
        Returns:
            Consensus analysis
        """
        consensus_analysis = {
            'high_consensus': [],
            'medium_consensus': [],
            'low_consensus': [],
            'conflicting_predictions': []
        }
        
        # Compare predictions between agents
        agent_names = list(agent_predictions.keys())
        for i, agent1 in enumerate(agent_names):
            for agent2 in agent_names[i+1:]:
                if (agent_predictions[agent1] is not None and 
                    agent_predictions[agent2] is not None):
                    
                    # Calculate correlation
                    correlation = np.corrcoef(
                        agent_predictions[agent1].flatten(),
                        agent_predictions[agent2].flatten()
                    )[0, 1]
                    
                    if correlation > 0.8:
                        consensus_analysis['high_consensus'].append((agent1, agent2))
                    elif correlation > 0.5:
                        consensus_analysis['medium_consensus'].append((agent1, agent2))
                    else:
                        consensus_analysis['low_consensus'].append((agent1, agent2))
        
        return consensus_analysis
    
    def _group_similar_recommendations(self, recommendations: List[Dict]) -> List[List[Dict]]:
        """
        Group similar recommendations together.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Grouped recommendations
        """
        groups = []
        used_indices = set()
        
        for i, rec1 in enumerate(recommendations):
            if i in used_indices:
                continue
            
            group = [rec1]
            used_indices.add(i)
            
            for j, rec2 in enumerate(recommendations[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Check if recommendations are similar
                if self._are_recommendations_similar(rec1, rec2):
                    group.append(rec2)
                    used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    def _are_recommendations_similar(self, rec1: Dict, rec2: Dict) -> bool:
        """
        Check if two recommendations are similar.
        
        Args:
            rec1: First recommendation
            rec2: Second recommendation
            
        Returns:
            True if similar, False otherwise
        """
        # Check type similarity
        if rec1.get('type') != rec2.get('type'):
            return False
        
        # Check priority similarity
        if rec1.get('priority') != rec2.get('priority'):
            return False
        
        # Check action similarity (simplified)
        actions1 = set(rec1.get('actions', []))
        actions2 = set(rec2.get('actions', []))
        
        if len(actions1) == 0 or len(actions2) == 0:
            return False
        
        similarity = len(actions1.intersection(actions2)) / len(actions1.union(actions2))
        return similarity > 0.5
    
    def _determine_priority(self, group: List[Dict]) -> str:
        """
        Determine priority for a group of recommendations.
        
        Args:
            group: Group of recommendations
            
        Returns:
            Priority level
        """
        priorities = [rec.get('priority', 'low') for rec in group]
        
        if 'high' in priorities:
            return 'high'
        elif 'medium' in priorities:
            return 'medium'
        else:
            return 'low'
    
    def _merge_actions(self, group: List[Dict]) -> List[str]:
        """
        Merge actions from a group of recommendations.
        
        Args:
            group: Group of recommendations
            
        Returns:
            Merged actions list
        """
        all_actions = []
        for rec in group:
            all_actions.extend(rec.get('actions', []))
        
        # Remove duplicates while preserving order
        unique_actions = []
        for action in all_actions:
            if action not in unique_actions:
                unique_actions.append(action)
        
        return unique_actions
    
    def _estimate_impact(self, group: List[Dict]) -> str:
        """
        Estimate impact of a group of recommendations.
        
        Args:
            group: Group of recommendations
            
        Returns:
            Impact estimation
        """
        impacts = [rec.get('estimated_impact', 'Unknown') for rec in group]
        
        if any('critical' in impact.lower() for impact in impacts):
            return 'Critical system impact'
        elif any('high' in impact.lower() for impact in impacts):
            return 'High impact on operations'
        elif any('medium' in impact.lower() for impact in impacts):
            return 'Medium impact on performance'
        else:
            return 'Low impact on system'
    
    def _determine_implementation_order(self, group: List[Dict]) -> List[str]:
        """
        Determine implementation order for recommendations.
        
        Args:
            group: Group of recommendations
            
        Returns:
            Implementation order
        """
        # Simple ordering based on agent weights and priorities
        ordered_agents = sorted(
            [(rec['source_agent'], rec.get('priority', 'low'), rec.get('weight', 0)) 
            for rec in group],
            key=lambda x: (x[1] == 'high', x[2]),
            reverse=True
        )
        
        return [agent for agent, _, _ in ordered_agents]
    
    def _calculate_coordination_score(self, agent_predictions: Dict[str, np.ndarray]) -> float:
        """
        Calculate coordination score based on agent predictions.
        
        Args:
            agent_predictions: Predictions from all agents
            
        Returns:
            Coordination score (0-1)
        """
        valid_predictions = {k: v for k, v in agent_predictions.items() if v is not None}
        
        if len(valid_predictions) < 2:
            return 0.5
        
        # Calculate average correlation between agents
        correlations = []
        agent_names = list(valid_predictions.keys())
        
        for i, agent1 in enumerate(agent_names):
            for agent2 in agent_names[i+1:]:
                pred1 = valid_predictions[agent1].flatten()
                pred2 = valid_predictions[agent2].flatten()
                
                if len(pred1) > 1 and len(pred2) > 1:
                    correlation = np.corrcoef(pred1, pred2)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
        
        if correlations:
            return np.mean(correlations)
        else:
            return 0.5
    
    def get_coordination_analytics(self) -> Dict[str, Any]:
        """
        Get coordination analytics and insights.
        
        Returns:
            Coordination analytics
        """
        if not self.coordination_history:
            return {'message': 'No coordination history available'}
        
        # Analyze coordination history
        recent_events = self.coordination_history[-100:]  # Last 100 events
        
        coordination_scores = [event['coordination_score'] for event in recent_events]
        avg_coordination_score = np.mean(coordination_scores)
        
        # Agent participation analysis
        agent_participation = {}
        for event in recent_events:
            for agent_name in event['agent_predictions'].keys():
                if event['agent_predictions'][agent_name] is not None:
                    agent_participation[agent_name] = agent_participation.get(agent_name, 0) + 1
        
        # Recommendation analysis
        all_recommendations = []
        for event in recent_events:
            all_recommendations.extend(event.get('recommendations', []))
        
        priority_distribution = {}
        for rec in all_recommendations:
            priority = rec.get('priority', 'unknown')
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        return {
            'avg_coordination_score': avg_coordination_score,
            'total_coordination_events': len(self.coordination_history),
            'agent_participation': agent_participation,
            'priority_distribution': priority_distribution,
            'recent_trend': 'improving' if len(coordination_scores) > 1 and 
                           coordination_scores[-1] > coordination_scores[0] else 'stable'
        }
    
    def start_coordination(self):
        """Start the coordination process."""
        self.is_running = True
        logger.info("AI Coordination started")
    
    def stop_coordination(self):
        """Stop the coordination process."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("AI Coordination stopped")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents.
        
        Returns:
            Status of all agents
        """
        agent_status = {}
        
        for agent_name, agent in self.agents.items():
            try:
                status = agent.get_agent_status()
                agent_status[agent_name] = {
                    'status': 'healthy',
                    'is_trained': status['is_trained'],
                    'metrics': status['metrics'],
                    'feature_importance_available': status['feature_importance_available']
                }
            except Exception as e:
                agent_status[agent_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return agent_status
