"""
ML Pipelines for Enhanced Telecom AI System

This module provides data processing pipelines for the Enhanced Telecom AI System,
including data ingestion, preprocessing, feature engineering, and model training.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import additional libraries
try:
    from faker import Faker
    from scapy.all import *
    FAKER_AVAILABLE = True
    SCAPY_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    SCAPY_AVAILABLE = False
    print("Faker and/or Scapy not available. Some data generation features will be limited.")

logger = logging.getLogger(__name__)

class MLPipeline:
    """
    Machine Learning Pipeline for the Enhanced Telecom AI System.
    
    Handles data ingestion, preprocessing, feature engineering, and model training.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the ML Pipeline.
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "sample").mkdir(exist_ok=True)
        
        # Initialize data generators
        if FAKER_AVAILABLE:
            self.faker = Faker()
        
        # Pipeline configuration
        self.preprocessing_config = {
            'missing_value_strategy': 'median',
            'outlier_detection': True,
            'feature_scaling': True,
            'feature_engineering': True
        }
        
        logger.info("ML Pipeline initialized")
    
    def generate_sample_data(self, data_type: str, n_samples: int = 1000, 
                           start_date: datetime = None) -> pd.DataFrame:
        """
        Generate sample data for testing and development.
        
        Args:
            data_type: Type of data to generate ('qos', 'traffic', 'energy', 'security', 'failure')
            n_samples: Number of samples to generate
            start_date: Start date for time series data
            
        Returns:
            Generated sample data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        
        if data_type == 'qos':
            return self._generate_qos_data(n_samples, start_date)
        elif data_type == 'traffic':
            return self._generate_traffic_data(n_samples, start_date)
        elif data_type == 'energy':
            return self._generate_energy_data(n_samples, start_date)
        elif data_type == 'security':
            return self._generate_security_data(n_samples, start_date)
        elif data_type == 'failure':
            return self._generate_failure_data(n_samples, start_date)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _generate_qos_data(self, n_samples: int, start_date: datetime) -> pd.DataFrame:
        """Generate QoS sample data."""
        timestamps = pd.date_range(start_date, periods=n_samples, freq='1min')
        
        # Generate realistic QoS metrics
        data = {
            'timestamp': timestamps,
            'latency_ms': np.random.normal(50, 15, n_samples).clip(10, 200),
            'throughput_mbps': np.random.normal(100, 30, n_samples).clip(10, 1000),
            'jitter_ms': np.random.exponential(5, n_samples).clip(0, 50),
            'packet_loss_rate': np.random.beta(2, 98, n_samples),
            'connection_quality': np.random.normal(85, 10, n_samples).clip(0, 100),
            'signal_strength': np.random.normal(-70, 15, n_samples).clip(-120, -30),
            'user_count': np.random.poisson(50, n_samples),
            'data_volume_gb': np.random.exponential(10, n_samples),
            'error_count': np.random.poisson(2, n_samples),
            'warning_count': np.random.poisson(5, n_samples)
        }
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        for idx in anomaly_indices:
            data['latency_ms'][idx] *= np.random.uniform(2, 5)
            data['packet_loss_rate'][idx] *= np.random.uniform(3, 10)
        
        return pd.DataFrame(data)
    
    def _generate_traffic_data(self, n_samples: int, start_date: datetime) -> pd.DataFrame:
        """Generate traffic sample data."""
        timestamps = pd.date_range(start_date, periods=n_samples, freq='1H')
        
        # Generate traffic patterns with daily and weekly seasonality
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        
        # Base traffic pattern
        base_traffic = 1000
        
        # Daily pattern (higher during business hours)
        daily_pattern = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = np.where(day_of_week < 5, 1.2, 0.8)
        
        # Random noise
        noise = np.random.normal(1, 0.2, n_samples)
        
        traffic_volume = base_traffic * daily_pattern * weekly_pattern * noise
        
        data = {
            'timestamp': timestamps,
            'traffic_volume': traffic_volume.clip(0),
            'user_count': (traffic_volume * np.random.uniform(0.1, 0.3, n_samples)).clip(0),
            'data_volume_gb': (traffic_volume * np.random.uniform(0.01, 0.05, n_samples)).clip(0),
            'peak_hour_indicator': (hour >= 8) & (hour <= 20),
            'is_weekend': day_of_week >= 5,
            'network_load_percent': np.random.uniform(20, 80, n_samples),
            'connection_count': np.random.poisson(100, n_samples),
            'bandwidth_utilization': np.random.uniform(30, 90, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _generate_energy_data(self, n_samples: int, start_date: datetime) -> pd.DataFrame:
        """Generate energy consumption sample data."""
        timestamps = pd.date_range(start_date, periods=n_samples, freq='1H')
        
        # Generate energy consumption patterns
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        
        # Base energy consumption
        base_energy = 100  # kWh
        
        # Daily pattern (higher during peak hours)
        daily_pattern = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = np.where(day_of_week < 5, 1.1, 0.9)
        
        # Environmental factors
        temperature = np.random.normal(25, 5, n_samples)
        humidity = np.random.uniform(40, 80, n_samples)
        wind_speed = np.random.exponential(5, n_samples)
        
        # Energy consumption calculation
        energy_consumption = (base_energy * daily_pattern * weekly_pattern * 
                           (1 + (temperature - 25) * 0.02) * 
                           (1 + (humidity - 50) * 0.01))
        
        data = {
            'timestamp': timestamps,
            'energy_consumption_kwh': energy_consumption.clip(0),
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'traffic_load': np.random.uniform(0.3, 1.0, n_samples),
            'user_count': np.random.poisson(100, n_samples),
            'data_volume': np.random.exponential(50, n_samples),
            'base_station_id': np.random.randint(1, 21, n_samples),
            'cell_load': np.random.uniform(0.2, 0.9, n_samples),
            'neighbor_load': np.random.uniform(0.3, 0.8, n_samples),
            'distance_to_users': np.random.uniform(100, 1000, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _generate_security_data(self, n_samples: int, start_date: datetime) -> pd.DataFrame:
        """Generate security sample data."""
        timestamps = pd.date_range(start_date, periods=n_samples, freq='1min')
        
        # Generate network traffic data
        protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']
        connection_states = ['ESTABLISHED', 'SYN_SENT', 'SYN_RECV', 'FIN_WAIT']
        user_agents = ['Mozilla/5.0', 'Chrome/91.0', 'Safari/14.0', 'Edge/91.0']
        request_types = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD']
        
        data = {
            'timestamp': timestamps,
            'source_ip': [self.faker.ipv4() if FAKER_AVAILABLE else f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'dest_ip': [self.faker.ipv4() if FAKER_AVAILABLE else f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
            'source_port': np.random.randint(1024, 65535, n_samples),
            'dest_port': np.random.choice([80, 443, 22, 21, 25, 53, 110, 143, 993, 995], n_samples),
            'protocol': np.random.choice(protocols, n_samples),
            'packet_size': np.random.exponential(1000, n_samples).clip(64, 1500),
            'packet_count': np.random.poisson(10, n_samples),
            'flow_duration': np.random.exponential(60, n_samples).clip(1, 3600),
            'bytes_sent': np.random.exponential(10000, n_samples),
            'bytes_received': np.random.exponential(8000, n_samples),
            'packet_rate': np.random.exponential(100, n_samples),
            'byte_rate': np.random.exponential(1000000, n_samples),
            'connection_state': np.random.choice(connection_states, n_samples),
            'flag_count': np.random.poisson(3, n_samples),
            'urgent_packets': np.random.poisson(0.1, n_samples),
            'user_agent': np.random.choice(user_agents, n_samples),
            'request_type': np.random.choice(request_types, n_samples),
            'threat_type': np.random.choice(['normal', 'ddos', 'intrusion', 'malware', 'botnet'], n_samples, p=[0.9, 0.03, 0.03, 0.02, 0.02])
        }
        
        # Add some security anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        for idx in anomaly_indices:
            data['packet_count'][idx] *= np.random.uniform(5, 20)
            data['byte_rate'][idx] *= np.random.uniform(3, 15)
        
        return pd.DataFrame(data)
    
    def _generate_failure_data(self, n_samples: int, start_date: datetime) -> pd.DataFrame:
        """Generate equipment failure sample data."""
        timestamps = pd.date_range(start_date, periods=n_samples, freq='1H')
        
        # Generate equipment health data
        equipment_types = ['BaseStation', 'Router', 'Switch', 'Gateway', 'Controller']
        
        data = {
            'timestamp': timestamps,
            'equipment_id': np.random.randint(1, 101, n_samples),
            'equipment_type': np.random.choice(equipment_types, n_samples),
            'equipment_age_days': np.random.exponential(365, n_samples).clip(1, 3650),
            'temperature_celsius': np.random.normal(35, 10, n_samples).clip(0, 80),
            'humidity_percent': np.random.uniform(20, 80, n_samples),
            'cpu_usage_percent': np.random.beta(2, 5, n_samples) * 100,
            'memory_usage_percent': np.random.beta(2, 5, n_samples) * 100,
            'disk_usage_percent': np.random.beta(2, 5, n_samples) * 100,
            'network_load_percent': np.random.beta(2, 5, n_samples) * 100,
            'uptime_hours': np.random.exponential(720, n_samples).clip(1, 8760),
            'restart_count': np.random.poisson(2, n_samples),
            'error_log_count': np.random.poisson(5, n_samples),
            'warning_log_count': np.random.poisson(10, n_samples),
            'failure_occurred': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        # Correlate failure with equipment health
        failure_indices = np.where(data['failure_occurred'] == 1)[0]
        for idx in failure_indices:
            data['cpu_usage_percent'][idx] = np.random.uniform(80, 100)
            data['memory_usage_percent'][idx] = np.random.uniform(85, 100)
            data['temperature_celsius'][idx] = np.random.uniform(60, 80)
            data['error_log_count'][idx] = np.random.poisson(20)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Preprocess data for machine learning.
        
        Args:
            data: Raw data
            config: Preprocessing configuration
            
        Returns:
            Preprocessed data
        """
        if config is None:
            config = self.preprocessing_config
        
        processed_data = data.copy()
        
        # Handle missing values
        if config.get('missing_value_strategy') == 'median':
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_columns] = processed_data[numeric_columns].fillna(
                processed_data[numeric_columns].median()
            )
        elif config.get('missing_value_strategy') == 'mean':
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_columns] = processed_data[numeric_columns].fillna(
                processed_data[numeric_columns].mean()
            )
        
        # Handle categorical variables
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            processed_data[col] = processed_data[col].fillna('unknown')
        
        # Remove outliers if configured
        if config.get('outlier_detection', False):
            processed_data = self._remove_outliers(processed_data)
        
        # Feature engineering
        if config.get('feature_engineering', False):
            processed_data = self._engineer_features(processed_data)
        
        return processed_data
    
    def _remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from data.
        
        Args:
            data: Input data
            method: Outlier detection method
            
        Returns:
            Data with outliers removed
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features.
        
        Args:
            data: Input data
            
        Returns:
            Data with engineered features
        """
        engineered_data = data.copy()
        
        # Time-based features
        if 'timestamp' in engineered_data.columns:
            timestamps = pd.to_datetime(engineered_data['timestamp'])
            engineered_data['hour'] = timestamps.dt.hour
            engineered_data['day_of_week'] = timestamps.dt.dayofweek
            engineered_data['month'] = timestamps.dt.month
            engineered_data['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
            engineered_data['is_peak_hour'] = ((timestamps.dt.hour >= 8) & 
                                             (timestamps.dt.hour <= 20)).astype(int)
        
        # Interaction features
        if 'latency_ms' in engineered_data.columns and 'throughput_mbps' in engineered_data.columns:
            engineered_data['latency_throughput_ratio'] = (
                engineered_data['latency_ms'] / (engineered_data['throughput_mbps'] + 1e-6)
            )
        
        if 'cpu_usage_percent' in engineered_data.columns and 'memory_usage_percent' in engineered_data.columns:
            engineered_data['cpu_memory_ratio'] = (
                engineered_data['cpu_usage_percent'] / (engineered_data['memory_usage_percent'] + 1e-6)
            )
        
        # Statistical features
        numeric_columns = engineered_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['timestamp', 'hour', 'day_of_week', 'month']:
                # Rolling statistics
                engineered_data[f'{col}_rolling_mean_24h'] = engineered_data[col].rolling(24).mean()
                engineered_data[f'{col}_rolling_std_24h'] = engineered_data[col].rolling(24).std()
                
                # Lag features
                engineered_data[f'{col}_lag_1h'] = engineered_data[col].shift(1)
                engineered_data[f'{col}_lag_24h'] = engineered_data[col].shift(24)
        
        return engineered_data
    
    def split_data(self, data: pd.DataFrame, target_column: str, 
                   test_size: float = 0.2, validation_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data
            target_column: Target column name
            test_size: Test set size ratio
            validation_size: Validation set size ratio
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            data, test_size=test_size, random_state=42, stratify=data[target_column] if target_column in data.columns else None
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val, test_size=validation_size/(1-test_size), random_state=42, 
            stratify=train_val[target_column] if target_column in train_val.columns else None
        )
        
        return train, val, test
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> str:
        """
        Save processed data to file.
        
        Args:
            data: Processed data
            filename: Filename to save
            
        Returns:
            Path to saved file
        """
        filepath = self.data_dir / "processed" / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
        return str(filepath)
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from file.
        
        Args:
            filename: Filename to load
            
        Returns:
            Loaded data
        """
        filepath = self.data_dir / "processed" / filename
        data = pd.read_csv(filepath)
        logger.info(f"Processed data loaded from {filepath}")
        return data
    
    def generate_training_datasets(self, output_dir: str = None) -> Dict[str, str]:
        """
        Generate training datasets for all agents.
        
        Args:
            output_dir: Output directory for datasets
            
        Returns:
            Dictionary mapping agent names to dataset filepaths
        """
        if output_dir is None:
            output_dir = self.data_dir / "processed"
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        datasets = {}
        
        # Generate datasets for each agent
        agent_data_types = {
            'qos_anomaly': 'qos',
            'failure_prediction': 'failure',
            'traffic_forecast': 'traffic',
            'energy_optimize': 'energy',
            'security_detection': 'security',
            'data_quality': 'qos'  # Use QoS data for data quality monitoring
        }
        
        for agent_name, data_type in agent_data_types.items():
            logger.info(f"Generating dataset for {agent_name}...")
            
            # Generate sample data
            sample_data = self.generate_sample_data(data_type, n_samples=5000)
            
            # Preprocess data
            processed_data = self.preprocess_data(sample_data)
            
            # Save dataset
            filename = f"{agent_name}_dataset.csv"
            filepath = output_dir / filename
            processed_data.to_csv(filepath, index=False)
            datasets[agent_name] = str(filepath)
            
            logger.info(f"Dataset for {agent_name} saved to {filepath}")
        
        return datasets
    
    def create_data_validation_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create data validation report.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': {},
            'data_types': {},
            'outliers': {},
            'data_quality_score': 0
        }
        
        # Check missing values
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_ratio = missing_count / len(data)
            report['missing_values'][col] = {
                'count': missing_count,
                'ratio': missing_ratio,
                'severity': 'high' if missing_ratio > 0.1 else 'medium' if missing_ratio > 0.05 else 'low'
            }
        
        # Check data types
        for col in data.columns:
            report['data_types'][col] = str(data[col].dtype)
        
        # Check outliers for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            report['outliers'][col] = {
                'count': outliers,
                'ratio': outliers / len(data)
            }
        
        # Calculate data quality score
        quality_score = 100
        for col in data.columns:
            missing_ratio = report['missing_values'][col]['ratio']
            quality_score -= missing_ratio * 20  # Penalty for missing values
        
        report['data_quality_score'] = max(0, quality_score)
        
        return report
