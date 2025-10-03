"""
Failure Prediction Agent

This agent predicts network equipment failures using:
- Historical failure data
- Equipment health metrics
- Environmental factors
- Usage patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class FailurePredictionAgent(BaseAgent):
    """
    AI Agent for predicting network equipment failures.
    
    Uses ensemble methods to predict failures with high accuracy.
    """
    
    def __init__(self):
        super().__init__("failure_prediction", "sklearn")
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = [
            'equipment_age_days', 'temperature_celsius', 'humidity_percent',
            'cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent',
            'network_load_percent', 'uptime_hours', 'restart_count',
            'error_log_count', 'warning_log_count', 'equipment_type'
        ]
        self.target_column = 'failure_occurred'
    
    def train(self, data: pd.DataFrame, target_column: str = None, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the failure prediction model.
        
        Args:
            data: Historical equipment data with failure labels
            target_column: Name of failure target column
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        logger.info("Training Failure Prediction Agent...")
        
        target_col = target_column or self.target_column
        
        # Prepare features and target
        features = self._prepare_features(data)
        target = data[target_col].astype(int)
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = target[:split_idx], target[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_val)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'precision': classification_report(y_val, y_pred, output_dict=True)['1']['precision'],
            'recall': classification_report(y_val, y_pred, output_dict=True)['1']['recall'],
            'f1_score': classification_report(y_val, y_pred, output_dict=True)['1']['f1-score']
        }
        
        self.metrics = metrics
        self.is_trained = True
        
        # Log to MLflow
        self.log_metrics_to_mlflow(metrics)
        
        logger.info(f"Failure Prediction Agent training completed. Metrics: {metrics}")
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict equipment failures.
        
        Args:
            data: Equipment health data
            
        Returns:
            Failure probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict_proba(features_scaled)[:, 1]
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Evaluate the failure prediction model.
        
        Args:
            test_data: Test data with known failures
            target_column: Name of failure target column
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        target_col = target_column or self.target_column
        features = self._prepare_features(test_data)
        target = test_data[target_col].astype(int)
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        predictions_proba = self.model.predict_proba(features_scaled)[:, 1]
        
        # Calculate comprehensive metrics
        report = classification_report(target, predictions, output_dict=True)
        auc_score = roc_auc_score(target, predictions_proba)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_score': auc_score,
            'true_positive_rate': report['1']['recall'],
            'false_positive_rate': 1 - report['0']['recall']
        }
        
        return metrics
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for failure prediction.
        
        Args:
            data: Raw equipment data
            
        Returns:
            Processed feature array
        """
        features = data[self.feature_columns].copy()
        
        # Handle categorical variables
        if 'equipment_type' in features.columns:
            if 'equipment_type' not in self.label_encoders:
                self.label_encoders['equipment_type'] = LabelEncoder()
                features['equipment_type'] = self.label_encoders['equipment_type'].fit_transform(
                    features['equipment_type'].astype(str)
                )
            else:
                features['equipment_type'] = self.label_encoders['equipment_type'].transform(
                    features['equipment_type'].astype(str)
                )
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Add derived features
        features['cpu_memory_ratio'] = features['cpu_usage_percent'] / (features['memory_usage_percent'] + 1e-6)
        features['load_stress_score'] = (
            features['cpu_usage_percent'] * features['memory_usage_percent'] * 
            features['network_load_percent'] / 10000
        )
        features['age_usage_factor'] = features['equipment_age_days'] * features['cpu_usage_percent'] / 100
        
        return features.values
    
    def get_recommendations(self, predictions: np.ndarray, 
                          confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate failure-specific recommendations.
        
        Args:
            predictions: Failure probability predictions
            confidence_threshold: Confidence threshold for recommendations
            
        Returns:
            List of failure prevention recommendations
        """
        recommendations = []
        
        for i, prob in enumerate(predictions):
            if prob >= confidence_threshold:
                severity = 'critical' if prob > 0.9 else 'high' if prob > 0.8 else 'medium'
                
                recommendations.append({
                    'index': i,
                    'type': 'failure_prediction',
                    'failure_probability': prob,
                    'severity': severity,
                    'actions': self._get_preventive_actions(prob),
                    'priority': 'high' if prob > 0.8 else 'medium',
                    'time_to_failure_estimate': self._estimate_time_to_failure(prob)
                })
        
        return recommendations
    
    def _get_preventive_actions(self, failure_prob: float) -> List[str]:
        """
        Get preventive actions based on failure probability.
        
        Args:
            failure_prob: Predicted failure probability
            
        Returns:
            List of recommended actions
        """
        if failure_prob > 0.9:
            return [
                'Immediate equipment replacement',
                'Schedule emergency maintenance',
                'Activate backup systems',
                'Notify operations team'
            ]
        elif failure_prob > 0.7:
            return [
                'Schedule preventive maintenance',
                'Increase monitoring frequency',
                'Prepare replacement equipment',
                'Review maintenance logs'
            ]
        else:
            return [
                'Continue monitoring',
                'Schedule routine maintenance',
                'Update maintenance schedule'
            ]
    
    def _estimate_time_to_failure(self, failure_prob: float) -> str:
        """
        Estimate time to failure based on probability.
        
        Args:
            failure_prob: Predicted failure probability
            
        Returns:
            Time estimate string
        """
        if failure_prob > 0.9:
            return "0-24 hours"
        elif failure_prob > 0.7:
            return "1-7 days"
        elif failure_prob > 0.5:
            return "1-4 weeks"
        else:
            return "1-3 months"
    
    def predict_equipment_lifetime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict remaining equipment lifetime.
        
        Args:
            data: Equipment health data
            
        Returns:
            Lifetime prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        failure_probs = self.predict(data)
        
        # Calculate risk scores
        risk_scores = []
        for prob in failure_probs:
            if prob > 0.8:
                risk_scores.append('critical')
            elif prob > 0.6:
                risk_scores.append('high')
            elif prob > 0.4:
                risk_scores.append('medium')
            else:
                risk_scores.append('low')
        
        return {
            'failure_probabilities': failure_probs.tolist(),
            'risk_scores': risk_scores,
            'high_risk_count': sum(1 for score in risk_scores if score in ['critical', 'high']),
            'recommended_actions': self.get_recommendations(failure_probs)
        }
