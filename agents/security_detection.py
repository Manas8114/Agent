"""
Security Detection Agent

This agent detects security threats in telecom networks including:
- Intrusion attempts
- DDoS attacks
- Malicious traffic patterns
- Suspicious user behavior
- Network anomalies
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SecurityDetectionAgent(BaseAgent):
    """
    AI Agent for detecting security threats in telecom networks.
    
    Uses multiple ML techniques to detect various types of security threats.
    """
    
    def __init__(self):
        super().__init__("security_detection", "sklearn")
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = [
            'source_ip', 'dest_ip', 'source_port', 'dest_port', 'protocol',
            'packet_size', 'packet_count', 'flow_duration', 'bytes_sent',
            'bytes_received', 'packet_rate', 'byte_rate', 'connection_state',
            'flag_count', 'urgent_packets', 'user_agent', 'request_type'
        ]
        self.threat_types = [
            'normal', 'ddos', 'intrusion', 'malware', 'botnet', 
            'scanning', 'brute_force', 'data_exfiltration'
        ]
        self.anomaly_threshold = 0.1
    
    def train(self, data: pd.DataFrame, target_column: str = None, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the security detection model.
        
        Args:
            data: Network traffic data with security labels
            target_column: Security threat target column
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        logger.info("Training Security Detection Agent...")
        
        target_col = target_column or 'threat_type'
        
        # Prepare features and target
        features = self._prepare_features(data)
        
        if target_col in data.columns:
            # Supervised learning with known threats
            target = data[target_col]
            metrics = self._train_supervised_model(features, target, validation_split)
        else:
            # Unsupervised learning for anomaly detection
            metrics = self._train_unsupervised_model(features, validation_split)
        
        self.metrics = metrics
        self.is_trained = True
        
        # Log to MLflow
        self.log_metrics_to_mlflow(metrics)
        
        logger.info(f"Security Detection Agent training completed. Metrics: {metrics}")
        return metrics
    
    def _train_supervised_model(self, features: np.ndarray, target: pd.Series, 
                              validation_split: float) -> Dict[str, float]:
        """
        Train supervised security detection model.
        
        Args:
            features: Feature matrix
            target: Target labels
            validation_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = target[:split_idx], target[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_val_scaled)
        y_pred_proba = self.model.predict_proba(X_val_scaled)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_val)
        auc_score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        
        # Get classification report
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'threat_detection_rate': self._calculate_threat_detection_rate(y_val, y_pred)
        }
        
        return metrics
    
    def _train_unsupervised_model(self, features: np.ndarray, 
                                validation_split: float) -> Dict[str, float]:
        """
        Train unsupervised anomaly detection model.
        
        Args:
            features: Feature matrix
            validation_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.anomaly_threshold,
            random_state=42,
            n_estimators=200
        )
        
        self.model.fit(X_train_scaled)
        
        # Make predictions
        y_pred = self.model.predict(X_val_scaled)
        anomaly_scores = self.model.score_samples(X_val_scaled)
        
        # Calculate metrics
        n_anomalies = np.sum(y_pred == -1)
        anomaly_rate = n_anomalies / len(y_pred)
        
        metrics = {
            'anomaly_rate': anomaly_rate,
            'mean_anomaly_score': np.mean(anomaly_scores),
            'std_anomaly_score': np.std(anomaly_scores),
            'n_anomalies_detected': n_anomalies,
            'total_samples': len(y_pred)
        }
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict security threats in network data.
        
        Args:
            data: Network traffic data
            
        Returns:
            Threat predictions or anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        
        if hasattr(self.model, 'predict_proba'):
            # Supervised model
            predictions = self.model.predict_proba(features_scaled)
            return predictions
        else:
            # Unsupervised model
            predictions = self.model.predict(features_scaled)
            anomaly_scores = self.model.score_samples(features_scaled)
            return np.column_stack([predictions, anomaly_scores])
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Evaluate the security detection model.
        
        Args:
            test_data: Test data with known threats
            target_column: Threat target column
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        features = self._prepare_features(test_data)
        features_scaled = self.scaler.transform(features)
        
        if hasattr(self.model, 'predict_proba'):
            # Supervised evaluation
            target_col = target_column or 'threat_type'
            target = test_data[target_col]
            
            predictions = self.model.predict(features_scaled)
            predictions_proba = self.model.predict_proba(features_scaled)
            
            # Calculate metrics
            accuracy = np.mean(predictions == target)
            auc_score = roc_auc_score(target, predictions_proba, multi_class='ovr')
            
            report = classification_report(target, predictions, output_dict=True)
            
            return {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'precision_macro': report['macro avg']['precision'],
                'recall_macro': report['macro avg']['recall'],
                'f1_macro': report['macro avg']['f1-score']
            }
        else:
            # Unsupervised evaluation
            predictions = self.model.predict(features_scaled)
            anomaly_scores = self.model.score_samples(features_scaled)
            
            return {
                'anomaly_detection_rate': np.sum(predictions == -1) / len(predictions),
                'mean_anomaly_score': np.mean(anomaly_scores),
                'min_anomaly_score': np.min(anomaly_scores),
                'max_anomaly_score': np.max(anomaly_scores)
            }
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for security detection.
        
        Args:
            data: Raw network data
            
        Returns:
            Processed feature array
        """
        features = data[self.feature_columns].copy()
        
        # Handle categorical variables
        categorical_cols = ['protocol', 'connection_state', 'user_agent', 'request_type']
        for col in categorical_cols:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features[col] = self.label_encoders[col].fit_transform(
                        features[col].astype(str)
                    )
                else:
                    features[col] = self.label_encoders[col].transform(
                        features[col].astype(str)
                    )
        
        # Handle IP addresses (simplified encoding)
        if 'source_ip' in features.columns:
            features['source_ip_encoded'] = features['source_ip'].apply(
                lambda x: hash(str(x)) % 10000
            )
        if 'dest_ip' in features.columns:
            features['dest_ip_encoded'] = features['dest_ip'].apply(
                lambda x: hash(str(x)) % 10000
            )
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Add derived security features
        features['packet_size_ratio'] = features['packet_size'] / (features['packet_count'] + 1e-6)
        features['byte_rate_per_packet'] = features['byte_rate'] / (features['packet_rate'] + 1e-6)
        features['flow_efficiency'] = features['bytes_sent'] / (features['bytes_received'] + 1e-6)
        features['urgent_ratio'] = features['urgent_packets'] / (features['packet_count'] + 1e-6)
        
        # Remove original IP columns if encoded versions exist
        if 'source_ip_encoded' in features.columns:
            features = features.drop('source_ip', axis=1)
        if 'dest_ip_encoded' in features.columns:
            features = features.drop('dest_ip', axis=1)
        
        return features.values
    
    def _calculate_threat_detection_rate(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> float:
        """
        Calculate threat detection rate.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Threat detection rate
        """
        # Count correctly detected threats (non-normal predictions)
        true_threats = y_true != 'normal'
        predicted_threats = y_pred != 'normal'
        
        if np.sum(true_threats) == 0:
            return 0.0
        
        detection_rate = np.sum(true_threats & predicted_threats) / np.sum(true_threats)
        return detection_rate
    
    def detect_ddos_attacks(self, data: pd.DataFrame, 
                          threshold_packets: int = 1000) -> List[Dict[str, Any]]:
        """
        Detect DDoS attacks in network data.
        
        Args:
            data: Network traffic data
            threshold_packets: Packet count threshold for DDoS detection
            
        Returns:
            List of detected DDoS attacks
        """
        ddos_attacks = []
        
        # Group by source IP and time window
        data['timestamp'] = pd.to_datetime(data.get('timestamp', pd.Timestamp.now()))
        data['time_window'] = data['timestamp'].dt.floor('1min')
        
        # Calculate packet rates per source IP
        ip_stats = data.groupby(['source_ip', 'time_window']).agg({
            'packet_count': 'sum',
            'packet_rate': 'mean',
            'byte_rate': 'sum'
        }).reset_index()
        
        # Detect high packet rate sources
        high_rate_sources = ip_stats[ip_stats['packet_count'] > threshold_packets]
        
        for _, row in high_rate_sources.iterrows():
            ddos_attacks.append({
                'source_ip': row['source_ip'],
                'time_window': row['time_window'],
                'packet_count': row['packet_count'],
                'packet_rate': row['packet_rate'],
                'byte_rate': row['byte_rate'],
                'severity': 'high' if row['packet_count'] > threshold_packets * 2 else 'medium',
                'recommended_actions': [
                    'Block source IP',
                    'Rate limit traffic',
                    'Activate DDoS protection',
                    'Notify security team'
                ]
            })
        
        return ddos_attacks
    
    def detect_intrusion_attempts(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect intrusion attempts in network data.
        
        Args:
            data: Network traffic data
            
        Returns:
            List of detected intrusion attempts
        """
        intrusions = []
        
        # Analyze suspicious patterns
        suspicious_patterns = data[
            (data['packet_size'] > data['packet_size'].quantile(0.95)) |
            (data['urgent_packets'] > 0) |
            (data['flag_count'] > data['flag_count'].quantile(0.9))
        ]
        
        for _, row in suspicious_patterns.iterrows():
            intrusions.append({
                'source_ip': row.get('source_ip', 'unknown'),
                'dest_ip': row.get('dest_ip', 'unknown'),
                'protocol': row.get('protocol', 'unknown'),
                'packet_size': row['packet_size'],
                'urgent_packets': row['urgent_packets'],
                'flag_count': row['flag_count'],
                'severity': 'high',
                'threat_type': 'intrusion_attempt',
                'recommended_actions': [
                    'Block suspicious IPs',
                    'Monitor network traffic',
                    'Check for data exfiltration',
                    'Update firewall rules'
                ]
            })
        
        return intrusions
    
    def get_recommendations(self, predictions: np.ndarray, 
                          confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate security-specific recommendations.
        
        Args:
            predictions: Security predictions
            confidence_threshold: Confidence threshold
            
        Returns:
            List of security recommendations
        """
        recommendations = []
        
        if len(predictions.shape) == 2:  # Supervised model with probabilities
            threat_probs = predictions
            max_probs = np.max(threat_probs, axis=1)
            threat_classes = np.argmax(threat_probs, axis=1)
            
            for i, (max_prob, threat_class) in enumerate(zip(max_probs, threat_classes)):
                if max_prob >= confidence_threshold and threat_class != 0:  # Not normal
                    threat_type = self.threat_types[threat_class]
                    
                    recommendations.append({
                        'index': i,
                        'type': 'security_threat',
                        'threat_type': threat_type,
                        'confidence': max_prob,
                        'severity': 'critical' if max_prob > 0.9 else 'high' if max_prob > 0.8 else 'medium',
                        'actions': self._get_security_actions(threat_type),
                        'priority': 'high' if max_prob > 0.8 else 'medium'
                    })
        
        else:  # Unsupervised model with anomaly scores
            anomaly_scores = predictions[:, 1] if predictions.shape[1] > 1 else predictions
            
            for i, score in enumerate(anomaly_scores):
                if score < -0.5:  # Low score indicates anomaly
                    recommendations.append({
                        'index': i,
                        'type': 'security_anomaly',
                        'anomaly_score': score,
                        'severity': 'high' if score < -1.0 else 'medium',
                        'actions': [
                            'Investigate network traffic',
                            'Check for unusual patterns',
                            'Monitor user behavior',
                            'Review security logs'
                        ],
                        'priority': 'high' if score < -1.0 else 'medium'
                    })
        
        return recommendations
    
    def _get_security_actions(self, threat_type: str) -> List[str]:
        """
        Get security actions for specific threat type.
        
        Args:
            threat_type: Type of security threat
            
        Returns:
            List of recommended actions
        """
        action_map = {
            'ddos': [
                'Activate DDoS protection',
                'Block malicious IPs',
                'Rate limit traffic',
                'Scale up resources'
            ],
            'intrusion': [
                'Block suspicious IPs',
                'Monitor network access',
                'Check authentication logs',
                'Update firewall rules'
            ],
            'malware': [
                'Quarantine infected systems',
                'Scan for malware',
                'Update antivirus signatures',
                'Isolate network segments'
            ],
            'botnet': [
                'Block botnet IPs',
                'Monitor command channels',
                'Update security policies',
                'Notify law enforcement'
            ],
            'scanning': [
                'Block scanning IPs',
                'Monitor port access',
                'Implement honeypots',
                'Update intrusion detection'
            ],
            'brute_force': [
                'Block attacking IPs',
                'Implement account lockout',
                'Monitor login attempts',
                'Enable two-factor authentication'
            ],
            'data_exfiltration': [
                'Block data transfer',
                'Monitor data access',
                'Check user permissions',
                'Implement data loss prevention'
            ]
        }
        
        return action_map.get(threat_type, [
            'Investigate security incident',
            'Monitor network traffic',
            'Update security policies',
            'Notify security team'
        ])
    
    def generate_security_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive security report.
        
        Args:
            data: Network traffic data
            
        Returns:
            Security report with threats and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating report")
        
        # Get predictions
        predictions = self.predict(data)
        
        # Detect specific threats
        ddos_attacks = self.detect_ddos_attacks(data)
        intrusions = self.detect_intrusion_attempts(data)
        
        # Generate recommendations
        recommendations = self.get_recommendations(predictions)
        
        # Calculate security metrics
        total_threats = len(ddos_attacks) + len(intrusions)
        high_severity_threats = sum(1 for r in recommendations if r.get('severity') == 'high')
        
        return {
            'total_threats_detected': total_threats,
            'ddos_attacks': len(ddos_attacks),
            'intrusion_attempts': len(intrusions),
            'high_severity_threats': high_severity_threats,
            'security_score': max(0, 100 - (total_threats * 10)),
            'threats': {
                'ddos_attacks': ddos_attacks,
                'intrusions': intrusions
            },
            'recommendations': recommendations,
            'next_actions': [
                'Review and implement security recommendations',
                'Update firewall and security policies',
                'Monitor network traffic continuously',
                'Schedule security team review'
            ]
        }
