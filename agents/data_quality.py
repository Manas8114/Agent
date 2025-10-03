"""
Data Quality Monitoring Agent

This agent monitors and ensures data quality in telecom networks by:
- Detecting missing or corrupted data
- Identifying data inconsistencies
- Monitoring data completeness
- Validating data formats and ranges
- Detecting data drift
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class DataQualityAgent(BaseAgent):
    """
    AI Agent for monitoring data quality in telecom networks.
    
    Detects data quality issues and provides recommendations for improvement.
    """
    
    def __init__(self):
        super().__init__("data_quality", "sklearn")
        self.scaler = StandardScaler()
        self.feature_columns = [
            'timestamp', 'latency_ms', 'throughput_mbps', 'jitter_ms',
            'packet_loss_rate', 'connection_quality', 'signal_strength',
            'user_count', 'data_volume', 'error_count', 'warning_count'
        ]
        self.quality_thresholds = {
            'latency_ms': {'min': 0, 'max': 1000},
            'throughput_mbps': {'min': 0, 'max': 10000},
            'jitter_ms': {'min': 0, 'max': 100},
            'packet_loss_rate': {'min': 0, 'max': 1},
            'connection_quality': {'min': 0, 'max': 100},
            'signal_strength': {'min': -120, 'max': 0}
        }
        self.anomaly_threshold = 0.05  # 5% of data expected to be anomalous
    
    def train(self, data: pd.DataFrame, target_column: str = None, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the data quality monitoring model.
        
        Args:
            data: Historical data quality metrics
            target_column: Data quality target column
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        logger.info("Training Data Quality Monitoring Agent...")
        
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
        
        # Calculate quality metrics
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(data)
        
        metrics = {
            'anomaly_rate': anomaly_rate,
            'mean_anomaly_score': np.mean(anomaly_scores),
            'std_anomaly_score': np.std(anomaly_scores),
            'n_anomalies_detected': n_anomalies,
            'total_samples': len(predictions),
            'data_quality_score': quality_score
        }
        
        self.metrics = metrics
        self.is_trained = True
        
        # Log to MLflow
        self.log_metrics_to_mlflow(metrics)
        
        logger.info(f"Data Quality Agent training completed. Metrics: {metrics}")
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict data quality issues.
        
        Args:
            data: Data to analyze
            
        Returns:
            Quality anomaly predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Evaluate the data quality monitoring model.
        
        Args:
            test_data: Test data with known quality issues
            target_column: Quality target column
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        features = self._prepare_features(test_data)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        anomaly_scores = self.model.score_samples(features_scaled)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(test_data)
        completeness_score = self._calculate_completeness_score(test_data)
        consistency_score = self._calculate_consistency_score(test_data)
        
        metrics = {
            'anomaly_detection_rate': np.sum(predictions == -1) / len(predictions),
            'mean_anomaly_score': np.mean(anomaly_scores),
            'min_anomaly_score': np.min(anomaly_scores),
            'max_anomaly_score': np.max(anomaly_scores),
            'data_quality_score': quality_score,
            'completeness_score': completeness_score,
            'consistency_score': consistency_score
        }
        
        return metrics
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for data quality analysis.
        
        Args:
            data: Raw data
            
        Returns:
            Processed feature array
        """
        features = data[self.feature_columns].copy()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Add derived quality features
        features['latency_throughput_ratio'] = features['latency_ms'] / (features['throughput_mbps'] + 1e-6)
        features['jitter_latency_ratio'] = features['jitter_ms'] / (features['latency_ms'] + 1e-6)
        features['error_rate'] = features['error_count'] / (features['user_count'] + 1e-6)
        features['warning_rate'] = features['warning_count'] / (features['user_count'] + 1e-6)
        
        # Add statistical features
        features['latency_zscore'] = np.abs((features['latency_ms'] - features['latency_ms'].mean()) / 
                                          (features['latency_ms'].std() + 1e-6))
        features['throughput_zscore'] = np.abs((features['throughput_mbps'] - features['throughput_mbps'].mean()) / 
                                             (features['throughput_mbps'].std() + 1e-6))
        
        return features.values
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            data: Data to analyze
            
        Returns:
            Quality score (0-100)
        """
        score = 100.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 30
        
        # Check for outliers
        outlier_penalty = 0
        for col, thresholds in self.quality_thresholds.items():
            if col in data.columns:
                outliers = ((data[col] < thresholds['min']) | 
                           (data[col] > thresholds['max'])).sum()
                outlier_ratio = outliers / len(data)
                outlier_penalty += outlier_ratio * 10
        
        score -= outlier_penalty
        
        # Check for data consistency
        consistency_penalty = self._calculate_consistency_penalty(data)
        score -= consistency_penalty
        
        return max(0, score)
    
    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """
        Calculate data completeness score.
        
        Args:
            data: Data to analyze
            
        Returns:
            Completeness score (0-100)
        """
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        return completeness
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """
        Calculate data consistency score.
        
        Args:
            data: Data to analyze
            
        Returns:
            Consistency score (0-100)
        """
        consistency_score = 100.0
        
        # Check for logical inconsistencies
        if 'latency_ms' in data.columns and 'throughput_mbps' in data.columns:
            # High latency should correlate with low throughput
            correlation = data['latency_ms'].corr(data['throughput_mbps'])
            if correlation > 0.5:  # Unexpected positive correlation
                consistency_score -= 20
        
        # Check for impossible values
        for col, thresholds in self.quality_thresholds.items():
            if col in data.columns:
                invalid_count = ((data[col] < thresholds['min']) | 
                               (data[col] > thresholds['max'])).sum()
                invalid_ratio = invalid_count / len(data)
                consistency_score -= invalid_ratio * 50
        
        return max(0, consistency_score)
    
    def _calculate_consistency_penalty(self, data: pd.DataFrame) -> float:
        """
        Calculate consistency penalty for quality score.
        
        Args:
            data: Data to analyze
            
        Returns:
            Consistency penalty (0-100)
        """
        penalty = 0
        
        # Check for data type inconsistencies
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if numeric data is stored as strings
                try:
                    pd.to_numeric(data[col], errors='raise')
                except:
                    penalty += 5
        
        # Check for duplicate records
        duplicate_ratio = data.duplicated().sum() / len(data)
        penalty += duplicate_ratio * 20
        
        return penalty
    
    def detect_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect missing data patterns.
        
        Args:
            data: Data to analyze
            
        Returns:
            Missing data analysis
        """
        missing_analysis = {}
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_ratio = missing_count / len(data)
            
            if missing_ratio > 0:
                missing_analysis[col] = {
                    'missing_count': missing_count,
                    'missing_ratio': missing_ratio,
                    'severity': 'high' if missing_ratio > 0.1 else 'medium' if missing_ratio > 0.05 else 'low'
                }
        
        return missing_analysis
    
    def detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Outlier analysis
        """
        outlier_analysis = {}
        
        for col, thresholds in self.quality_thresholds.items():
            if col in data.columns:
                outliers = ((data[col] < thresholds['min']) | 
                           (data[col] > thresholds['max']))
                outlier_count = outliers.sum()
                outlier_ratio = outlier_count / len(data)
                
                if outlier_ratio > 0:
                    outlier_analysis[col] = {
                        'outlier_count': outlier_count,
                        'outlier_ratio': outlier_ratio,
                        'min_value': data[col].min(),
                        'max_value': data[col].max(),
                        'severity': 'high' if outlier_ratio > 0.05 else 'medium'
                    }
        
        return outlier_analysis
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Data drift analysis
        """
        drift_analysis = {}
        
        for col in reference_data.columns:
            if col in current_data.columns and reference_data[col].dtype in ['int64', 'float64']:
                ref_mean = reference_data[col].mean()
                ref_std = reference_data[col].std()
                curr_mean = current_data[col].mean()
                curr_std = current_data[col].std()
                
                # Calculate drift metrics
                mean_drift = abs(curr_mean - ref_mean) / (ref_std + 1e-6)
                std_drift = abs(curr_std - ref_std) / (ref_std + 1e-6)
                
                drift_analysis[col] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'drift_severity': 'high' if mean_drift > 2 or std_drift > 2 else 'medium' if mean_drift > 1 or std_drift > 1 else 'low',
                    'reference_mean': ref_mean,
                    'current_mean': curr_mean,
                    'reference_std': ref_std,
                    'current_std': curr_std
                }
        
        return drift_analysis
    
    def get_recommendations(self, predictions: np.ndarray, 
                          confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate data quality recommendations.
        
        Args:
            predictions: Quality predictions
            confidence_threshold: Confidence threshold
            
        Returns:
            List of data quality recommendations
        """
        recommendations = []
        
        for i, pred in enumerate(predictions):
            if pred == -1:  # Quality issue detected
                recommendations.append({
                    'index': i,
                    'type': 'data_quality_issue',
                    'severity': 'high',
                    'actions': [
                        'Review data collection process',
                        'Check data validation rules',
                        'Verify data sources',
                        'Implement data quality checks'
                    ],
                    'priority': 'high',
                    'estimated_impact': 'Data reliability compromised'
                })
        
        return recommendations
    
    def generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: Data to analyze
            
        Returns:
            Data quality report
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating report")
        
        # Get predictions
        predictions = self.predict(data)
        
        # Analyze data quality issues
        missing_analysis = self.detect_missing_data(data)
        outlier_analysis = self.detect_outliers(data)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(data)
        completeness_score = self._calculate_completeness_score(data)
        consistency_score = self._calculate_consistency_score(data)
        
        # Generate recommendations
        recommendations = self.get_recommendations(predictions)
        
        # Count quality issues
        n_quality_issues = np.sum(predictions == -1)
        n_missing_columns = len(missing_analysis)
        n_outlier_columns = len(outlier_analysis)
        
        return {
            'overall_quality_score': quality_score,
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'quality_issues_detected': n_quality_issues,
            'missing_data_columns': n_missing_columns,
            'outlier_columns': n_outlier_columns,
            'missing_data_analysis': missing_analysis,
            'outlier_analysis': outlier_analysis,
            'recommendations': recommendations,
            'quality_status': 'good' if quality_score > 80 else 'fair' if quality_score > 60 else 'poor',
            'next_actions': [
                'Address missing data issues',
                'Clean outlier data',
                'Implement data validation',
                'Monitor data quality continuously'
            ]
        }
    
    def auto_correct_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically correct common data quality issues.
        
        Args:
            data: Data to correct
            
        Returns:
            Corrected data
        """
        corrected_data = data.copy()
        
        # Fill missing values with median
        for col in corrected_data.columns:
            if corrected_data[col].dtype in ['int64', 'float64']:
                corrected_data[col] = corrected_data[col].fillna(corrected_data[col].median())
        
        # Cap outliers to reasonable ranges
        for col, thresholds in self.quality_thresholds.items():
            if col in corrected_data.columns:
                corrected_data[col] = corrected_data[col].clip(
                    lower=thresholds['min'], 
                    upper=thresholds['max']
                )
        
        return corrected_data
