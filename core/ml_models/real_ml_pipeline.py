#!/usr/bin/env python3
"""
Real Machine Learning Pipeline with MLflow Tracking
Implements actual ML models using scikit-learn and PyTorch
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import logging
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime

class MLflowTracker:
    """MLflow experiment tracking for reproducibility"""
    
    def __init__(self, experiment_name: str = "telecom_ai_agents"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.logger = logging.getLogger(__name__)
    
    def start_run(self, agent_name: str, model_type: str):
        """Start MLflow run"""
        return mlflow.start_run(run_name=f"{agent_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow"""
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        for param_name, value in params.items():
            mlflow.log_param(param_name, value)
    
    def log_model(self, model, model_name: str):
        """Log model to MLflow"""
        if hasattr(model, 'predict'):
            mlflow.sklearn.log_model(model, model_name)
        else:
            mlflow.pytorch.log_model(model, model_name)

class QoSAnomalyMLModel:
    """Real ML model for QoS anomaly detection using Isolation Forest"""
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.tracker = MLflowTracker()
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        features = [
            'latency_ms', 'throughput_mbps', 'jitter_ms', 
            'packet_loss_rate', 'connection_quality', 'signal_strength'
        ]
        
        # Handle missing values
        df_clean = df[features].fillna(df[features].median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df_clean)
        
        return X_scaled
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the anomaly detection model"""
        with self.tracker.start_run("qos_anomaly", "isolation_forest") as run:
            X = self.prepare_features(data)
            
            # Train model
            self.model.fit(X)
            
            # Predict anomalies
            predictions = self.model.predict(X)
            anomaly_scores = self.model.decision_function(X)
            
            # Calculate metrics
            n_anomalies = np.sum(predictions == -1)
            anomaly_rate = n_anomalies / len(predictions)
            
            metrics = {
                'anomaly_rate': anomaly_rate,
                'n_anomalies': n_anomalies,
                'mean_anomaly_score': np.mean(anomaly_scores),
                'std_anomaly_score': np.std(anomaly_scores)
            }
            
            # Log to MLflow
            self.tracker.log_params({
                'contamination': 0.1,
                'n_estimators': 100,
                'random_state': 42
            })
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(self.model, "qos_anomaly_model")
            
            self.logger.info(f"QoS Anomaly Model trained. Anomaly rate: {anomaly_rate:.3f}")
            
            return metrics
    
    def predict(self, new_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies on new data"""
        X = self.prepare_features(new_data)
        predictions = self.model.predict(X)
        anomaly_scores = self.model.decision_function(X)
        
        return predictions, anomaly_scores
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        X = self.prepare_features(test_data)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, cv=5, scoring='neg_mean_squared_error')
        
        metrics = {
            'cv_mean_score': np.mean(cv_scores),
            'cv_std_score': np.std(cv_scores),
            'model_score': self.model.score(X)
        }
        
        return metrics

class FailurePredictionMLModel:
    """Real ML model for failure prediction using Random Forest"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.tracker = MLflowTracker()
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for failure prediction"""
        feature_cols = [
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_latency',
            'error_count', 'warning_count', 'uptime_hours', 'temperature'
        ]
        
        # Handle missing values
        df_clean = df[feature_cols + ['failure_occurred']].fillna(df[feature_cols + ['failure_occurred']].median())
        
        X = df_clean[feature_cols]
        y = df_clean['failure_occurred']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the failure prediction model"""
        with self.tracker.start_run("failure_prediction", "random_forest") as run:
            X, y = self.prepare_features(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'feature_importance_mean': np.mean(self.model.feature_importances_),
                'feature_importance_std': np.std(self.model.feature_importances_)
            }
            
            # Log to MLflow
            self.tracker.log_params({
                'n_estimators': 200,
                'max_depth': 10,
                'class_weight': 'balanced',
                'random_state': 42
            })
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(self.model, "failure_prediction_model")
            
            self.logger.info(f"Failure Prediction Model trained. Accuracy: {metrics['accuracy']:.3f}")
            
            return metrics
    
    def predict(self, new_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict failures on new data"""
        X, _ = self.prepare_features(new_data)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        X, y = self.prepare_features(test_data)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='f1_weighted')
        
        metrics = {
            'cv_f1_mean': np.mean(cv_scores),
            'cv_f1_std': np.std(cv_scores),
            'feature_importance': self.model.feature_importances_.tolist()
        }
        
        return metrics

class TrafficForecastLSTM(nn.Module):
    """LSTM model for traffic forecasting using PyTorch"""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
        super(TrafficForecastLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TrafficForecastMLModel:
    """Real ML model for traffic forecasting using LSTM"""
    
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.model = TrafficForecastLSTM()
        self.scaler = StandardScaler()
        self.tracker = MLflowTracker()
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'traffic_volume') -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare LSTM sequences"""
        feature_cols = ['latency_ms', 'throughput_mbps', 'jitter_ms', 'packet_loss_rate', 'connection_quality', 'signal_strength']
        
        # Prepare data
        features = data[feature_cols].values
        target = data[target_col].values
        
        # Scale data
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target[i])
        
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the LSTM model"""
        with self.tracker.start_run("traffic_forecast", "lstm") as run:
            X, y = self.prepare_sequences(data)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            train_losses = []
            
            for epoch in range(100):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                train_losses.append(epoch_loss / len(train_loader))
                
                if epoch % 20 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")
            
            # Evaluation
            self.model.eval()
            with torch.no_grad():
                test_pred = self.model(X_test)
                test_loss = criterion(test_pred.squeeze(), y_test).item()
                mae = torch.mean(torch.abs(test_pred.squeeze() - y_test)).item()
                mape = torch.mean(torch.abs((test_pred.squeeze() - y_test) / y_test)).item()
            
            metrics = {
                'final_train_loss': train_losses[-1],
                'test_loss': test_loss,
                'mae': mae,
                'mape': mape,
                'r2_score': 1 - (test_loss / torch.var(y_test).item())
            }
            
            # Log to MLflow
            self.tracker.log_params({
                'sequence_length': self.sequence_length,
                'hidden_size': 50,
                'num_layers': 2,
                'learning_rate': 0.001,
                'epochs': 100
            })
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(self.model, "traffic_forecast_model")
            
            self.logger.info(f"Traffic Forecast Model trained. MAE: {mae:.3f}")
            
            return metrics
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Predict traffic on new data"""
        X, _ = self.prepare_sequences(new_data)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.cpu().numpy()
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        X, y = self.prepare_sequences(test_data)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
            mse = torch.mean((predictions.squeeze() - y) ** 2).item()
            mae = torch.mean(torch.abs(predictions.squeeze() - y)).item()
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
        
        return metrics

class EnergyOptimizationMLModel:
    """Real ML model for energy optimization using Gradient Boosting"""
    
    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.tracker = MLflowTracker()
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for energy optimization"""
        feature_cols = [
            'cpu_usage', 'memory_usage', 'active_users', 'data_throughput',
            'temperature', 'uptime_hours', 'load_balancing_score'
        ]
        
        # Handle missing values
        df_clean = df[feature_cols + ['energy_consumption']].fillna(df[feature_cols + ['energy_consumption']].median())
        
        X = df_clean[feature_cols]
        y = df_clean['energy_consumption']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the energy optimization model"""
        with self.tracker.start_run("energy_optimization", "gradient_boosting") as run:
            X, y = self.prepare_features(data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            metrics = {
                'mse': mse,
                'r2_score': r2,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'feature_importance_mean': np.mean(self.model.feature_importances_),
                'feature_importance_std': np.std(self.model.feature_importances_)
            }
            
            # Log to MLflow
            self.tracker.log_params({
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            })
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(self.model, "energy_optimization_model")
            
            self.logger.info(f"Energy Optimization Model trained. R²: {r2:.3f}")
            
            return metrics
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Predict energy consumption on new data"""
        X, _ = self.prepare_features(new_data)
        predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        X, y = self.prepare_features(test_data)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        metrics = {
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'feature_importance': self.model.feature_importances_.tolist()
        }
        
        return metrics

class MLModelFactory:
    """Factory for creating and managing ML models"""
    
    @staticmethod
    def create_model(agent_type: str):
        """Create appropriate ML model for agent type"""
        models = {
            'qos_anomaly': QoSAnomalyMLModel,
            'failure_prediction': FailurePredictionMLModel,
            'traffic_forecast': TrafficForecastMLModel,
            'energy_optimization': EnergyOptimizationMLModel
        }
        
        if agent_type not in models:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return models[agent_type]()

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    n_samples = 1000
    data = pd.DataFrame({
        'latency_ms': np.random.normal(50, 10, n_samples),
        'throughput_mbps': np.random.normal(100, 20, n_samples),
        'jitter_ms': np.random.exponential(5, n_samples),
        'packet_loss_rate': np.random.exponential(0.01, n_samples),
        'connection_quality': np.random.uniform(70, 99, n_samples),
        'signal_strength': np.random.uniform(-90, -60, n_samples),
        'cpu_usage': np.random.uniform(20, 90, n_samples),
        'memory_usage': np.random.uniform(30, 85, n_samples),
        'failure_occurred': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'traffic_volume': np.random.exponential(1000, n_samples),
        'energy_consumption': np.random.normal(2.5, 0.5, n_samples)
    })
    
    # Test models
    qos_model = MLModelFactory.create_model('qos_anomaly')
    qos_metrics = qos_model.train(data)
    print(f"QoS Anomaly Model: {qos_metrics}")
    
    failure_model = MLModelFactory.create_model('failure_prediction')
    failure_metrics = failure_model.train(data)
    print(f"Failure Prediction Model: {failure_metrics}")
