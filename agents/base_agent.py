"""
Base Agent Class for Enhanced Telecom AI System

This module provides the abstract base class that all AI agents must implement.
Each agent must provide train, predict, and evaluate methods with MLflow tracking.
"""

import logging
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import joblib
import os

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in the Enhanced Telecom AI System.
    
    Each agent must implement:
    - train(): Train the model on provided data
    - predict(): Make predictions on new data
    - evaluate(): Evaluate model performance on test data
    """
    
    def __init__(self, agent_name: str, model_type: str = "sklearn"):
        """
        Initialize the base agent.
        
        Args:
            agent_name: Name of the agent (e.g., "qos_anomaly")
            model_type: Type of ML framework ("sklearn", "pytorch", "tensorflow")
        """
        self.agent_name = agent_name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.metrics = {}
        self.feature_importance = None
        
        # MLflow experiment name
        self.experiment_name = f"telecom_ai_{agent_name}"
        
        # Initialize MLflow experiment
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
    
    @abstractmethod
    def train(self, data: pd.DataFrame, target_column: str = None, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the agent's model on provided data.
        
        Args:
            data: Training data as pandas DataFrame
            target_column: Name of target column (for supervised learning)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data as pandas DataFrame
            
        Returns:
            Predictions as numpy array
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data as pandas DataFrame
            target_column: Name of target column
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance array or None
        """
        return self.feature_importance
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.model_type == "sklearn":
            joblib.dump(self.model, path)
        elif self.model_type == "pytorch":
            import torch
            torch.save(self.model.state_dict(), path)
        else:
            # For other frameworks, use joblib as fallback
            joblib.dump(self.model, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a pre-trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if self.model_type == "sklearn":
            self.model = joblib.load(path)
        elif self.model_type == "pytorch":
            import torch
            self.model.load_state_dict(torch.load(path))
        else:
            self.model = joblib.load(path)
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    def log_metrics_to_mlflow(self, metrics: Dict[str, float], step: int = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        try:
            with mlflow.start_run(run_name=f"{self.agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value, step=step)
                
                # Log model if available
                if self.model is not None:
                    if self.model_type == "sklearn":
                        mlflow.sklearn.log_model(self.model, "model")
                    elif self.model_type == "pytorch":
                        mlflow.pytorch.log_model(self.model, "model")
                
                logger.info(f"Logged metrics to MLflow: {metrics}")
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
    
    def get_recommendations(self, predictions: np.ndarray, 
                          confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on predictions.
        
        Args:
            predictions: Model predictions
            confidence_threshold: Minimum confidence for recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # This is a base implementation - each agent should override
        # with domain-specific recommendations
        for i, pred in enumerate(predictions):
            if isinstance(pred, (list, np.ndarray)) and len(pred) > 1:
                confidence = max(pred)
                prediction_class = np.argmax(pred)
            else:
                confidence = abs(pred)
                prediction_class = 1 if pred > 0.5 else 0
            
            if confidence >= confidence_threshold:
                recommendations.append({
                    'index': i,
                    'prediction': prediction_class,
                    'confidence': confidence,
                    'action': self._get_action_for_prediction(prediction_class),
                    'priority': 'high' if confidence > 0.9 else 'medium'
                })
        
        return recommendations
    
    def _get_action_for_prediction(self, prediction: int) -> str:
        """
        Get recommended action for a prediction.
        Override in subclasses for domain-specific actions.
        
        Args:
            prediction: Model prediction
            
        Returns:
            Recommended action string
        """
        return "Monitor closely" if prediction == 1 else "Continue normal operations"
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status of the agent.
        
        Returns:
            Dictionary with agent status information
        """
        return {
            'agent_name': self.agent_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'feature_importance_available': self.feature_importance is not None
        }
