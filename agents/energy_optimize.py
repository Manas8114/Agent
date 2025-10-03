"""
Energy Optimization Agent

This agent optimizes energy consumption in telecom networks by:
- Predicting energy demand patterns
- Optimizing base station power management
- Load balancing across cells
- Dynamic power scaling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class EnergyOptimizeAgent(BaseAgent):
    """
    AI Agent for optimizing energy consumption in telecom networks.
    
    Uses ML models to predict energy demand and optimization algorithms
    to minimize energy consumption while maintaining service quality.
    """
    
    def __init__(self):
        super().__init__("energy_optimize", "sklearn")
        self.scaler = StandardScaler()
        self.feature_columns = [
            'traffic_load', 'user_count', 'data_volume', 'time_of_day',
            'day_of_week', 'temperature', 'humidity', 'wind_speed',
            'base_station_id', 'cell_load', 'neighbor_load', 'distance_to_users'
        ]
        self.optimization_target = 'energy_consumption_kwh'
        self.constraints = {
            'min_power': 0.1,  # Minimum 10% power
            'max_power': 1.0,  # Maximum 100% power
            'min_coverage': 0.95,  # Minimum 95% coverage
            'max_latency': 50  # Maximum 50ms latency
        }
    
    def train(self, data: pd.DataFrame, target_column: str = None, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the energy optimization model.
        
        Args:
            data: Historical energy and traffic data
            target_column: Energy consumption target column
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        logger.info("Training Energy Optimization Agent...")
        
        target_col = target_column or self.optimization_target
        
        # Prepare features and target
        features = self._prepare_features(data)
        target = data[target_col].values
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = target[:split_idx], target[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train energy prediction model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        # Calculate energy efficiency metrics
        energy_savings_potential = self._calculate_energy_savings_potential(y_val, y_pred)
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'energy_savings_potential': energy_savings_potential,
            'avg_energy_consumption': np.mean(y_val),
            'predicted_avg_consumption': np.mean(y_pred)
        }
        
        self.metrics = metrics
        self.is_trained = True
        
        # Log to MLflow
        self.log_metrics_to_mlflow(metrics)
        
        logger.info(f"Energy Optimization Agent training completed. Metrics: {metrics}")
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict energy consumption for given conditions.
        
        Args:
            data: Network conditions data
            
        Returns:
            Energy consumption predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Evaluate the energy optimization model.
        
        Args:
            test_data: Test data with energy consumption
            target_column: Energy consumption target column
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        target_col = target_column or self.optimization_target
        features = self._prepare_features(test_data)
        target = test_data[target_col].values
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(target, predictions)
        mse = mean_squared_error(target, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(target, predictions)
        
        # Energy-specific metrics
        energy_savings = self._calculate_energy_savings(target, predictions)
        efficiency_score = self._calculate_efficiency_score(target, predictions)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'energy_savings_percent': energy_savings,
            'efficiency_score': efficiency_score
        }
    
    def optimize_energy_consumption(self, current_conditions: pd.DataFrame, 
                                  optimization_horizon: int = 24) -> Dict[str, Any]:
        """
        Optimize energy consumption for given conditions.
        
        Args:
            current_conditions: Current network conditions
            optimization_horizon: Hours to optimize ahead
            
        Returns:
            Optimization results with recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before optimization")
        
        # Predict baseline energy consumption
        baseline_consumption = self.predict(current_conditions)
        
        # Generate optimization scenarios
        optimization_scenarios = self._generate_optimization_scenarios(
            current_conditions, optimization_horizon
        )
        
        # Find optimal configuration
        optimal_config = self._find_optimal_configuration(
            current_conditions, optimization_scenarios
        )
        
        # Calculate potential savings
        optimized_consumption = self.predict(optimal_config)
        energy_savings = baseline_consumption - optimized_consumption
        savings_percent = (energy_savings / baseline_consumption) * 100
        
        return {
            'baseline_consumption': baseline_consumption.tolist(),
            'optimized_consumption': optimized_consumption.tolist(),
            'energy_savings': energy_savings.tolist(),
            'savings_percent': savings_percent.tolist(),
            'optimal_configuration': optimal_config.to_dict('records'),
            'recommendations': self._generate_optimization_recommendations(
                energy_savings, optimal_config
            )
        }
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for energy optimization.
        
        Args:
            data: Raw network data
            
        Returns:
            Processed feature array
        """
        features = data[self.feature_columns].copy()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Add derived features
        features['load_efficiency'] = features['traffic_load'] / (features['user_count'] + 1e-6)
        features['environmental_factor'] = (
            features['temperature'] * features['humidity'] / 100
        )
        features['network_density'] = features['user_count'] / (features['distance_to_users'] + 1e-6)
        
        # Time-based features
        if 'time_of_day' in features.columns:
            features['is_peak_hour'] = ((features['time_of_day'] >= 8) & 
                                      (features['time_of_day'] <= 20)).astype(int)
            features['is_night'] = ((features['time_of_day'] >= 22) | 
                                  (features['time_of_day'] <= 6)).astype(int)
        
        return features.values
    
    def _calculate_energy_savings_potential(self, actual: np.ndarray, 
                                          predicted: np.ndarray) -> float:
        """
        Calculate potential energy savings.
        
        Args:
            actual: Actual energy consumption
            predicted: Predicted energy consumption
            
        Returns:
            Energy savings potential percentage
        """
        baseline_consumption = np.mean(actual)
        optimized_consumption = np.mean(predicted)
        savings_potential = ((baseline_consumption - optimized_consumption) / 
                           baseline_consumption) * 100
        
        return max(0, savings_potential)
    
    def _calculate_energy_savings(self, actual: np.ndarray, 
                                predicted: np.ndarray) -> float:
        """
        Calculate actual energy savings.
        
        Args:
            actual: Actual energy consumption
            predicted: Predicted energy consumption
            
        Returns:
            Energy savings percentage
        """
        savings = np.mean(actual - predicted)
        savings_percent = (savings / np.mean(actual)) * 100
        
        return max(0, savings_percent)
    
    def _calculate_efficiency_score(self, actual: np.ndarray, 
                                  predicted: np.ndarray) -> float:
        """
        Calculate energy efficiency score.
        
        Args:
            actual: Actual energy consumption
            predicted: Predicted energy consumption
            
        Returns:
            Efficiency score (0-1)
        """
        # Efficiency based on how close predictions are to optimal
        optimal_consumption = np.percentile(actual, 25)  # 25th percentile as optimal
        efficiency = 1 - (np.mean(predicted) - optimal_consumption) / optimal_consumption
        
        return max(0, min(1, efficiency))
    
    def _generate_optimization_scenarios(self, conditions: pd.DataFrame, 
                                       horizon: int) -> List[pd.DataFrame]:
        """
        Generate optimization scenarios.
        
        Args:
            conditions: Current conditions
            horizon: Optimization horizon in hours
            
        Returns:
            List of optimization scenarios
        """
        scenarios = []
        
        for hour in range(horizon):
            scenario = conditions.copy()
            
            # Adjust traffic load based on time
            time_of_day = (conditions['time_of_day'].iloc[0] + hour) % 24
            if 8 <= time_of_day <= 20:  # Peak hours
                scenario['traffic_load'] *= 1.2
            else:  # Off-peak hours
                scenario['traffic_load'] *= 0.8
            
            # Adjust power settings
            scenario['power_level'] = self._calculate_optimal_power_level(scenario)
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_optimal_power_level(self, conditions: pd.DataFrame) -> float:
        """
        Calculate optimal power level for given conditions.
        
        Args:
            conditions: Network conditions
            
        Returns:
            Optimal power level (0-1)
        """
        # Base power calculation
        traffic_factor = conditions['traffic_load'].iloc[0]
        user_factor = conditions['user_count'].iloc[0]
        environmental_factor = conditions['temperature'].iloc[0] / 30.0  # Normalize temperature
        
        # Calculate optimal power level
        optimal_power = (traffic_factor * 0.4 + user_factor * 0.3 + 
                        environmental_factor * 0.3)
        
        # Apply constraints
        optimal_power = max(self.constraints['min_power'], 
                          min(self.constraints['max_power'], optimal_power))
        
        return optimal_power
    
    def _find_optimal_configuration(self, conditions: pd.DataFrame, 
                                  scenarios: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Find optimal configuration from scenarios.
        
        Args:
            conditions: Current conditions
            scenarios: Optimization scenarios
            
        Returns:
            Optimal configuration
        """
        best_scenario = None
        best_score = float('inf')
        
        for scenario in scenarios:
            # Predict energy consumption for scenario
            energy_consumption = self.predict(scenario)
            
            # Calculate optimization score (lower is better)
            energy_score = np.mean(energy_consumption)
            coverage_score = self._calculate_coverage_score(scenario)
            latency_score = self._calculate_latency_score(scenario)
            
            total_score = (energy_score * 0.5 + 
                          (1 - coverage_score) * 0.3 + 
                          latency_score * 0.2)
            
            if total_score < best_score:
                best_score = total_score
                best_scenario = scenario
        
        return best_scenario
    
    def _calculate_coverage_score(self, scenario: pd.DataFrame) -> float:
        """
        Calculate coverage score for scenario.
        
        Args:
            scenario: Network scenario
            
        Returns:
            Coverage score (0-1)
        """
        # Simplified coverage calculation
        power_level = scenario['power_level'].iloc[0] if 'power_level' in scenario.columns else 0.8
        user_count = scenario['user_count'].iloc[0]
        
        # Coverage decreases with more users and lower power
        coverage = power_level * (1 - (user_count / 1000))  # Assume max 1000 users
        
        return max(0, min(1, coverage))
    
    def _calculate_latency_score(self, scenario: pd.DataFrame) -> float:
        """
        Calculate latency score for scenario.
        
        Args:
            scenario: Network scenario
            
        Returns:
            Latency score (0-1, lower is better)
        """
        # Simplified latency calculation
        traffic_load = scenario['traffic_load'].iloc[0]
        power_level = scenario['power_level'].iloc[0] if 'power_level' in scenario.columns else 0.8
        
        # Latency increases with traffic load and decreases with power
        latency = (traffic_load / power_level) * 10  # Base latency calculation
        
        # Normalize to 0-1 scale
        latency_score = min(1, latency / 100)
        
        return latency_score
    
    def _generate_optimization_recommendations(self, energy_savings: np.ndarray, 
                                             optimal_config: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations.
        
        Args:
            energy_savings: Calculated energy savings
            optimal_config: Optimal configuration
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        avg_savings = np.mean(energy_savings)
        
        if avg_savings > 0.1:  # 10% savings
            recommendations.append({
                'type': 'power_optimization',
                'priority': 'high',
                'actions': [
                    'Implement dynamic power scaling',
                    'Activate sleep mode for low-traffic cells',
                    'Optimize antenna tilt and power distribution',
                    'Schedule maintenance during low-traffic hours'
                ],
                'expected_savings': f"{avg_savings:.1%}",
                'implementation_time': '1-2 weeks'
            })
        
        if 'power_level' in optimal_config.columns:
            power_levels = optimal_config['power_level'].values
            if np.min(power_levels) < 0.5:
                recommendations.append({
                    'type': 'load_balancing',
                    'priority': 'medium',
                    'actions': [
                        'Redistribute traffic across cells',
                        'Implement intelligent handover',
                        'Optimize cell coverage areas',
                        'Use energy-efficient modulation schemes'
                    ],
                    'expected_savings': '5-15%',
                    'implementation_time': '2-4 weeks'
                })
        
        return recommendations
    
    def get_recommendations(self, predictions: np.ndarray, 
                          confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate energy optimization recommendations.
        
        Args:
            predictions: Energy consumption predictions
            confidence_threshold: Confidence threshold
            
        Returns:
            List of energy optimization recommendations
        """
        recommendations = []
        
        avg_consumption = np.mean(predictions)
        max_consumption = np.max(predictions)
        
        # High consumption recommendations
        if max_consumption > avg_consumption * 1.5:
            recommendations.append({
                'index': -1,
                'type': 'energy_optimization',
                'severity': 'high',
                'predicted_consumption': max_consumption,
                'actions': [
                    'Activate power saving mode',
                    'Implement load balancing',
                    'Schedule traffic offloading',
                    'Review equipment efficiency'
                ],
                'priority': 'high',
                'expected_savings': '15-25%'
            })
        
        # General optimization recommendations
        if avg_consumption > np.percentile(predictions, 50):
            recommendations.append({
                'index': -1,
                'type': 'general_optimization',
                'severity': 'medium',
                'predicted_consumption': avg_consumption,
                'actions': [
                    'Optimize base station configurations',
                    'Implement smart cooling systems',
                    'Use renewable energy sources',
                    'Monitor equipment efficiency'
                ],
                'priority': 'medium',
                'expected_savings': '5-15%'
            })
        
        return recommendations
