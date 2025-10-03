"""
QoS Anomaly Detection Agent

This agent detects anomalies in Quality of Service metrics including:
- Latency spikes
- Throughput drops
- Jitter anomalies
- Packet loss increases
- Connection failures
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any
import logging

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class QoSAnomalyAgent(BaseAgent):
    """
    AI Agent for detecting QoS anomalies in telecom networks.
    
    Uses Isolation Forest for unsupervised anomaly detection on QoS metrics.
    """
    
    def __init__(self):
        super().__init__("qos_anomaly", "sklearn")
        self.scaler = StandardScaler()
        self.anomaly_threshold = 0.1  # 10% of data expected to be anomalous
        self.feature_columns = [
            'latency_ms', 'throughput_mbps', 'jitter_ms', 
            'packet_loss_rate', 'connection_quality', 'signal_strength'
        ]
    
    def train(self, data: pd.DataFrame, target_column: str = None, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the QoS anomaly detection model.
        
        Args:
            data: QoS metrics data
            target_column: Not used for unsupervised learning
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        logger.info("Training QoS Anomaly Detection Agent...")
        
        # Prepare features
        features = self._prepare_features(data)
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        train_features = features[:split_idx]
        val_features = features[split_idx:]
        
        # Initialize and train model
        self.model = IsolationForest(
            contamination=self.anomaly_threshold,
            random_state=42,
            n_estimators=100
        )
        
        # Fit scaler and model
        train_scaled = self.scaler.fit_transform(train_features)
        self.model.fit(train_scaled)
        
        # Evaluate on validation set
        val_scaled = self.scaler.transform(val_features)
        predictions = self.model.predict(val_scaled)
        anomaly_scores = self.model.score_samples(val_scaled)
        
        # Calculate metrics
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)
        
        metrics = {
            'anomaly_rate': anomaly_rate,
            'mean_anomaly_score': np.mean(anomaly_scores),
            'std_anomaly_score': np.std(anomaly_scores),
            'n_anomalies_detected': n_anomalies,
            'total_samples': len(predictions)
        }
        
        self.metrics = metrics
        self.is_trained = True
        
        # Log to MLflow
        self.log_metrics_to_mlflow(metrics)
        
        logger.info(f"QoS Anomaly Agent training completed. Metrics: {metrics}")
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies in QoS data.
        
        Args:
            data: QoS metrics data
            
        Returns:
            Anomaly predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Evaluate the anomaly detection model.
        
        Args:
            test_data: Test data with known anomalies
            target_column: Column indicating true anomaly labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        features = self._prepare_features(test_data)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        anomaly_scores = self.model.score_samples(features_scaled)
        
        metrics = {
            'anomaly_detection_rate': np.sum(predictions == -1) / len(predictions),
            'mean_anomaly_score': np.mean(anomaly_scores),
            'min_anomaly_score': np.min(anomaly_scores),
            'max_anomaly_score': np.max(anomaly_scores)
        }
        
        # If ground truth is available, calculate precision/recall
        if target_column and target_column in test_data.columns:
            true_labels = (test_data[target_column] == 1).astype(int)  # 1 for anomaly
            pred_labels = (predictions == -1).astype(int)
            
            report = classification_report(true_labels, pred_labels, output_dict=True)
            metrics.update({
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score']
            })
        
        return metrics
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection.
        
        Args:
            data: Raw QoS data
            
        Returns:
            Processed feature array
        """
        # Select and clean features
        features = data[self.feature_columns].copy()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Add derived features
        features['latency_throughput_ratio'] = features['latency_ms'] / (features['throughput_mbps'] + 1e-6)
        features['jitter_latency_ratio'] = features['jitter_ms'] / (features['latency_ms'] + 1e-6)
        features['quality_score'] = (
            features['connection_quality'] * features['signal_strength'] / 100
        )
        
        return features.values
    
    def get_recommendations(self, predictions: np.ndarray, 
                          confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate QoS-specific recommendations for detected anomalies.
        
        Args:
            predictions: Anomaly predictions
            confidence_threshold: Confidence threshold for recommendations
            
        Returns:
            List of QoS recommendations
        """
        recommendations = []
        
        for i, pred in enumerate(predictions):
            if pred == -1:  # Anomaly detected
                recommendations.append({
                    'index': i,
                    'type': 'qos_anomaly',
                    'severity': 'high',
                    'actions': [
                        'Check network congestion',
                        'Verify routing configuration',
                        'Monitor adjacent cells',
                        'Review traffic patterns'
                    ],
                    'priority': 'high',
                    'estimated_impact': 'Service degradation possible'
                })
        
        return recommendations
    
    def detect_latency_spikes(self, data: pd.DataFrame, threshold_percentile: float = 95) -> List[int]:
        """
        Detect specific latency spikes in the data.
        
        Args:
            data: QoS data
            threshold_percentile: Percentile threshold for spike detection
            
        Returns:
            List of indices with latency spikes
        """
        latency_threshold = np.percentile(data['latency_ms'], threshold_percentile)
        spike_indices = data[data['latency_ms'] > latency_threshold].index.tolist()
        
        return spike_indices
    
    def detect_throughput_drops(self, data: pd.DataFrame, threshold_percentile: float = 5) -> List[int]:
        """
        Detect throughput drops in the data.
        
        Args:
            data: QoS data
            threshold_percentile: Percentile threshold for drop detection
            
        Returns:
            List of indices with throughput drops
        """
        throughput_threshold = np.percentile(data['throughput_mbps'], threshold_percentile)
        drop_indices = data[data['throughput_mbps'] < throughput_threshold].index.tolist()
        
        return drop_indices
