"""
Sample Data Generator for Enhanced Telecom AI System

This module generates realistic sample data for testing and development
of the Enhanced Telecom AI System.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import additional libraries
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    print("Faker not available. Some data generation features will be limited.")

logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """
    Generator for realistic sample data for the Enhanced Telecom AI System.
    
    Creates synthetic but realistic data for all system components.
    """
    
    def __init__(self, output_dir: str = "data/sample"):
        """
        Initialize the sample data generator.
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if FAKER_AVAILABLE:
            self.faker = Faker()
        
        # Data generation configuration
        self.config = {
            'qos_samples': 10000,
            'traffic_samples': 8760,  # 1 year of hourly data
            'energy_samples': 8760,
            'security_samples': 50000,
            'failure_samples': 5000,
            'data_quality_samples': 10000
        }
        
        logger.info("Sample Data Generator initialized")
    
    def generate_all_datasets(self) -> Dict[str, str]:
        """
        Generate all sample datasets.
        
        Returns:
            Dictionary mapping dataset names to file paths
        """
        datasets = {}
        
        logger.info("Generating all sample datasets...")
        
        # Generate QoS data
        qos_data = self.generate_qos_data()
        qos_path = self.output_dir / "qos_sample_data.csv"
        qos_data.to_csv(qos_path, index=False)
        datasets['qos'] = str(qos_path)
        logger.info(f"QoS data generated: {qos_path}")
        
        # Generate traffic data
        traffic_data = self.generate_traffic_data()
        traffic_path = self.output_dir / "traffic_sample_data.csv"
        traffic_data.to_csv(traffic_path, index=False)
        datasets['traffic'] = str(traffic_path)
        logger.info(f"Traffic data generated: {traffic_path}")
        
        # Generate energy data
        energy_data = self.generate_energy_data()
        energy_path = self.output_dir / "energy_sample_data.csv"
        energy_data.to_csv(energy_path, index=False)
        datasets['energy'] = str(energy_path)
        logger.info(f"Energy data generated: {energy_path}")
        
        # Generate security data
        security_data = self.generate_security_data()
        security_path = self.output_dir / "security_sample_data.csv"
        security_data.to_csv(security_path, index=False)
        datasets['security'] = str(security_path)
        logger.info(f"Security data generated: {security_path}")
        
        # Generate failure data
        failure_data = self.generate_failure_data()
        failure_path = self.output_dir / "failure_sample_data.csv"
        failure_data.to_csv(failure_path, index=False)
        datasets['failure'] = str(failure_path)
        logger.info(f"Failure data generated: {failure_path}")
        
        # Generate data quality data
        data_quality_data = self.generate_data_quality_data()
        dq_path = self.output_dir / "data_quality_sample_data.csv"
        data_quality_data.to_csv(dq_path, index=False)
        datasets['data_quality'] = str(dq_path)
        logger.info(f"Data quality data generated: {dq_path}")
        
        # Save dataset metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'datasets': datasets,
            'config': self.config
        }
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("All sample datasets generated successfully")
        return datasets
    
    def generate_qos_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate QoS (Quality of Service) sample data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            QoS sample data
        """
        if n_samples is None:
            n_samples = self.config['qos_samples']
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=30)
        timestamps = pd.date_range(start_time, periods=n_samples, freq='1min')
        
        # Generate realistic QoS metrics
        data = {
            'timestamp': timestamps,
            'cell_id': np.random.randint(1, 101, n_samples),
            'user_id': np.random.randint(1000, 10000, n_samples),
            'latency_ms': self._generate_latency_data(n_samples),
            'throughput_mbps': self._generate_throughput_data(n_samples),
            'jitter_ms': self._generate_jitter_data(n_samples),
            'packet_loss_rate': self._generate_packet_loss_data(n_samples),
            'connection_quality': self._generate_connection_quality_data(n_samples),
            'signal_strength': self._generate_signal_strength_data(n_samples),
            'user_count': np.random.poisson(50, n_samples),
            'data_volume_gb': np.random.exponential(10, n_samples),
            'error_count': np.random.poisson(2, n_samples),
            'warning_count': np.random.poisson(5, n_samples),
            'session_duration_minutes': np.random.exponential(30, n_samples),
            'handover_count': np.random.poisson(1, n_samples)
        }
        
        # Add some QoS anomalies
        self._add_qos_anomalies(data, n_samples)
        
        return pd.DataFrame(data)
    
    def generate_traffic_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate network traffic sample data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Traffic sample data
        """
        if n_samples is None:
            n_samples = self.config['traffic_samples']
        
        # Generate timestamps (hourly for 1 year)
        start_time = datetime.now() - timedelta(days=365)
        timestamps = pd.date_range(start_time, periods=n_samples, freq='1H')
        
        # Generate traffic patterns with seasonality
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        month = timestamps.month
        
        # Base traffic pattern
        base_traffic = 1000
        
        # Daily pattern (higher during business hours)
        daily_pattern = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = np.where(day_of_week < 5, 1.2, 0.8)
        
        # Monthly pattern (higher in certain months)
        monthly_pattern = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
        
        # Random noise
        noise = np.random.normal(1, 0.2, n_samples)
        
        traffic_volume = base_traffic * daily_pattern * weekly_pattern * monthly_pattern * noise
        
        data = {
            'timestamp': timestamps,
            'traffic_volume': np.clip(traffic_volume, 0, None),
            'user_count': np.clip(traffic_volume * np.random.uniform(0.1, 0.3, n_samples), 0, None),
            'data_volume_gb': np.clip(traffic_volume * np.random.uniform(0.01, 0.05, n_samples), 0, None),
            'peak_hour_indicator': (hour >= 8) & (hour <= 20),
            'is_weekend': day_of_week >= 5,
            'network_load_percent': np.random.uniform(20, 80, n_samples),
            'connection_count': np.random.poisson(100, n_samples),
            'bandwidth_utilization': np.random.uniform(30, 90, n_samples),
            'protocol_distribution': np.random.choice(['HTTP', 'HTTPS', 'FTP', 'SSH', 'DNS'], n_samples),
            'geographic_region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_energy_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate energy consumption sample data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Energy sample data
        """
        if n_samples is None:
            n_samples = self.config['energy_samples']
        
        # Generate timestamps (hourly for 1 year)
        start_time = datetime.now() - timedelta(days=365)
        timestamps = pd.date_range(start_time, periods=n_samples, freq='1H')
        
        # Generate environmental data
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        month = timestamps.month
        
        # Environmental factors
        temperature = self._generate_temperature_data(n_samples, month)
        humidity = np.random.uniform(40, 80, n_samples)
        wind_speed = np.random.exponential(5, n_samples)
        
        # Traffic load (correlated with energy consumption)
        traffic_load = self._generate_traffic_load_data(n_samples, hour, day_of_week)
        
        # Energy consumption calculation
        base_energy = 100  # kWh
        daily_pattern = 1 + 0.3 * np.sin(2 * np.pi * hour / 24)
        weekly_pattern = np.where(day_of_week < 5, 1.1, 0.9)
        seasonal_pattern = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
        temperature_factor = 1 + (temperature - 25) * 0.02
        traffic_factor = 1 + traffic_load * 0.5
        
        energy_consumption = (base_energy * daily_pattern * weekly_pattern * 
                            seasonal_pattern * temperature_factor * traffic_factor)
        
        data = {
            'timestamp': timestamps,
            'energy_consumption_kwh': np.clip(energy_consumption, 0, None),
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'traffic_load': traffic_load,
            'user_count': np.random.poisson(100, n_samples),
            'data_volume': np.random.exponential(50, n_samples),
            'base_station_id': np.random.randint(1, 21, n_samples),
            'cell_load': np.random.uniform(0.2, 0.9, n_samples),
            'neighbor_load': np.random.uniform(0.3, 0.8, n_samples),
            'distance_to_users': np.random.uniform(100, 1000, n_samples),
            'power_efficiency': np.random.uniform(0.7, 0.95, n_samples),
            'cooling_load': energy_consumption * np.random.uniform(0.1, 0.3, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def generate_security_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate security sample data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Security sample data
        """
        if n_samples is None:
            n_samples = self.config['security_samples']
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=7)
        timestamps = pd.date_range(start_time, periods=n_samples, freq='1min')
        
        # Generate network traffic data
        protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'FTP', 'SSH']
        connection_states = ['ESTABLISHED', 'SYN_SENT', 'SYN_RECV', 'FIN_WAIT', 'CLOSE_WAIT']
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'curl/7.68.0',
            'PostmanRuntime/7.26.8'
        ]
        request_types = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS']
        
        # Generate threat types with realistic distribution
        threat_types = np.random.choice(
            ['normal', 'ddos', 'intrusion', 'malware', 'botnet', 'scanning', 'brute_force'],
            n_samples,
            p=[0.85, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01]
        )
        
        data = {
            'timestamp': timestamps,
            'source_ip': [self._generate_ip_address() for _ in range(n_samples)],
            'dest_ip': [self._generate_ip_address() for _ in range(n_samples)],
            'source_port': np.random.randint(1024, 65535, n_samples),
            'dest_port': np.random.choice([80, 443, 22, 21, 25, 53, 110, 143, 993, 995], n_samples),
            'protocol': np.random.choice(protocols, n_samples),
            'packet_size': np.clip(np.random.exponential(1000, n_samples), 64, 1500),
            'packet_count': self._generate_packet_count_data(n_samples, threat_types),
            'flow_duration': np.clip(np.random.exponential(60, n_samples), 1, 3600),
            'bytes_sent': np.random.exponential(10000, n_samples),
            'bytes_received': np.random.exponential(8000, n_samples),
            'packet_rate': np.random.exponential(100, n_samples),
            'byte_rate': np.random.exponential(1000000, n_samples),
            'connection_state': np.random.choice(connection_states, n_samples),
            'flag_count': np.random.poisson(3, n_samples),
            'urgent_packets': np.random.poisson(0.1, n_samples),
            'user_agent': np.random.choice(user_agents, n_samples),
            'request_type': np.random.choice(request_types, n_samples),
            'threat_type': threat_types,
            'severity': self._generate_severity_data(threat_types)
        }
        
        # Add security anomalies
        self._add_security_anomalies(data, threat_types)
        
        return pd.DataFrame(data)
    
    def generate_failure_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate equipment failure sample data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Failure sample data
        """
        if n_samples is None:
            n_samples = self.config['failure_samples']
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=30)
        timestamps = pd.date_range(start_time, periods=n_samples, freq='1H')
        
        # Generate equipment data
        equipment_types = ['BaseStation', 'Router', 'Switch', 'Gateway', 'Controller', 'Antenna']
        equipment_ids = np.random.randint(1, 201, n_samples)
        
        # Generate failure labels (5% failure rate)
        failure_occurred = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        
        data = {
            'timestamp': timestamps,
            'equipment_id': equipment_ids,
            'equipment_type': np.random.choice(equipment_types, n_samples),
            'equipment_age_days': np.clip(np.random.exponential(365, n_samples), 1, 3650),
            'temperature_celsius': self._generate_temperature_data(n_samples),
            'humidity_percent': np.random.uniform(20, 80, n_samples),
            'cpu_usage_percent': self._generate_cpu_usage_data(n_samples, failure_occurred),
            'memory_usage_percent': self._generate_memory_usage_data(n_samples, failure_occurred),
            'disk_usage_percent': np.random.beta(2, 5, n_samples) * 100,
            'network_load_percent': np.random.beta(2, 5, n_samples) * 100,
            'uptime_hours': np.clip(np.random.exponential(720, n_samples), 1, 8760),
            'restart_count': np.random.poisson(2, n_samples),
            'error_log_count': self._generate_error_log_data(n_samples, failure_occurred),
            'warning_log_count': np.random.poisson(10, n_samples),
            'failure_occurred': failure_occurred,
            'maintenance_due': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'last_maintenance_days': np.clip(np.random.exponential(90, n_samples), 1, 365)
        }
        
        # Correlate failure with equipment health
        self._correlate_failure_with_health(data, failure_occurred)
        
        return pd.DataFrame(data)
    
    def generate_data_quality_data(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate data quality sample data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Data quality sample data
        """
        if n_samples is None:
            n_samples = self.config['data_quality_samples']
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=7)
        timestamps = pd.date_range(start_time, periods=n_samples, freq='1min')
        
        # Generate data quality metrics
        data = {
            'timestamp': timestamps,
            'latency_ms': self._generate_latency_data(n_samples),
            'throughput_mbps': self._generate_throughput_data(n_samples),
            'jitter_ms': self._generate_jitter_data(n_samples),
            'packet_loss_rate': self._generate_packet_loss_data(n_samples),
            'connection_quality': self._generate_connection_quality_data(n_samples),
            'signal_strength': self._generate_signal_strength_data(n_samples),
            'user_count': np.random.poisson(50, n_samples),
            'data_volume_gb': np.random.exponential(10, n_samples),
            'error_count': np.random.poisson(2, n_samples),
            'warning_count': np.random.poisson(5, n_samples),
            'data_completeness': np.random.uniform(0.8, 1.0, n_samples),
            'data_consistency': np.random.uniform(0.7, 1.0, n_samples),
            'data_accuracy': np.random.uniform(0.85, 1.0, n_samples),
            'data_timeliness': np.random.uniform(0.9, 1.0, n_samples)
        }
        
        # Add data quality issues
        self._add_data_quality_issues(data, n_samples)
        
        return pd.DataFrame(data)
    
    def _generate_latency_data(self, n_samples: int) -> np.ndarray:
        """Generate realistic latency data."""
        base_latency = np.random.normal(50, 15, n_samples)
        # Add some high latency spikes
        spike_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        base_latency[spike_indices] *= np.random.uniform(2, 5, len(spike_indices))
        return np.clip(base_latency, 10, 500)
    
    def _generate_throughput_data(self, n_samples: int) -> np.ndarray:
        """Generate realistic throughput data."""
        base_throughput = np.random.normal(100, 30, n_samples)
        # Add some low throughput periods
        low_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        base_throughput[low_indices] *= np.random.uniform(0.1, 0.5, len(low_indices))
        return np.clip(base_throughput, 1, 1000)
    
    def _generate_jitter_data(self, n_samples: int) -> np.ndarray:
        """Generate realistic jitter data."""
        return np.clip(np.random.exponential(5, n_samples), 0, 50)
    
    def _generate_packet_loss_data(self, n_samples: int) -> np.ndarray:
        """Generate realistic packet loss data."""
        return np.clip(np.random.beta(2, 98, n_samples), 0, 0.1)
    
    def _generate_connection_quality_data(self, n_samples: int) -> np.ndarray:
        """Generate realistic connection quality data."""
        return np.clip(np.random.normal(85, 10, n_samples), 0, 100)
    
    def _generate_signal_strength_data(self, n_samples: int) -> np.ndarray:
        """Generate realistic signal strength data."""
        return np.clip(np.random.normal(-70, 15, n_samples), -120, -30)
    
    def _generate_temperature_data(self, n_samples: int, month: np.ndarray = None) -> np.ndarray:
        """Generate realistic temperature data."""
        if month is not None:
            # Seasonal temperature variation
            seasonal_temp = 25 + 10 * np.sin(2 * np.pi * month / 12)
            return np.clip(np.random.normal(seasonal_temp, 5, n_samples), -10, 50)
        else:
            return np.clip(np.random.normal(25, 5, n_samples), -10, 50)
    
    def _generate_traffic_load_data(self, n_samples: int, hour: np.ndarray, day_of_week: np.ndarray) -> np.ndarray:
        """Generate realistic traffic load data."""
        # Daily pattern
        daily_pattern = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
        # Weekly pattern
        weekly_pattern = np.where(day_of_week < 5, 1.2, 0.8)
        # Base load
        base_load = np.random.uniform(0.3, 1.0, n_samples)
        return base_load * daily_pattern * weekly_pattern
    
    def _generate_cpu_usage_data(self, n_samples: int, failure_occurred: np.ndarray) -> np.ndarray:
        """Generate CPU usage data correlated with failures."""
        cpu_usage = np.random.beta(2, 5, n_samples) * 100
        # Higher CPU usage for failed equipment
        failure_indices = np.where(failure_occurred == 1)[0]
        cpu_usage[failure_indices] = np.random.uniform(80, 100, len(failure_indices))
        return cpu_usage
    
    def _generate_memory_usage_data(self, n_samples: int, failure_occurred: np.ndarray) -> np.ndarray:
        """Generate memory usage data correlated with failures."""
        memory_usage = np.random.beta(2, 5, n_samples) * 100
        # Higher memory usage for failed equipment
        failure_indices = np.where(failure_occurred == 1)[0]
        memory_usage[failure_indices] = np.random.uniform(85, 100, len(failure_indices))
        return memory_usage
    
    def _generate_error_log_data(self, n_samples: int, failure_occurred: np.ndarray) -> np.ndarray:
        """Generate error log data correlated with failures."""
        error_count = np.random.poisson(5, n_samples)
        # More errors for failed equipment
        failure_indices = np.where(failure_occurred == 1)[0]
        error_count[failure_indices] = np.random.poisson(20, len(failure_indices))
        return error_count
    
    def _generate_packet_count_data(self, n_samples: int, threat_types: np.ndarray) -> np.ndarray:
        """Generate packet count data with threat correlation."""
        packet_count = np.random.poisson(10, n_samples)
        # Higher packet count for DDoS attacks
        ddos_indices = np.where(threat_types == 'ddos')[0]
        packet_count[ddos_indices] = np.random.poisson(100, len(ddos_indices))
        return packet_count
    
    def _generate_severity_data(self, threat_types: np.ndarray) -> np.ndarray:
        """Generate severity data based on threat types."""
        severity_map = {
            'normal': 'low',
            'ddos': 'high',
            'intrusion': 'high',
            'malware': 'critical',
            'botnet': 'high',
            'scanning': 'medium',
            'brute_force': 'medium'
        }
        return np.array([severity_map.get(threat, 'low') for threat in threat_types])
    
    def _generate_ip_address(self) -> str:
        """Generate a random IP address."""
        if FAKER_AVAILABLE:
            return self.faker.ipv4()
        else:
            return f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
    
    def _add_qos_anomalies(self, data: Dict, n_samples: int):
        """Add QoS anomalies to the data."""
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        for idx in anomaly_indices:
            data['latency_ms'][idx] *= np.random.uniform(2, 5)
            data['packet_loss_rate'][idx] *= np.random.uniform(3, 10)
            data['connection_quality'][idx] *= np.random.uniform(0.3, 0.7)
    
    def _add_security_anomalies(self, data: Dict, threat_types: np.ndarray):
        """Add security anomalies to the data."""
        # Increase packet count for DDoS attacks
        ddos_indices = np.where(threat_types == 'ddos')[0]
        for idx in ddos_indices:
            data['packet_count'][idx] *= np.random.uniform(5, 20)
            data['byte_rate'][idx] *= np.random.uniform(3, 15)
        
        # Increase urgent packets for intrusions
        intrusion_indices = np.where(threat_types == 'intrusion')[0]
        for idx in intrusion_indices:
            data['urgent_packets'][idx] = np.random.poisson(5)
            data['flag_count'][idx] = np.random.poisson(10)
    
    def _correlate_failure_with_health(self, data: Dict, failure_occurred: np.ndarray):
        """Correlate failure with equipment health metrics."""
        failure_indices = np.where(failure_occurred == 1)[0]
        for idx in failure_indices:
            data['cpu_usage_percent'][idx] = np.random.uniform(80, 100)
            data['memory_usage_percent'][idx] = np.random.uniform(85, 100)
            data['temperature_celsius'][idx] = np.random.uniform(60, 80)
            data['error_log_count'][idx] = np.random.poisson(20)
    
    def _add_data_quality_issues(self, data: Dict, n_samples: int):
        """Add data quality issues to the data."""
        # Add missing values
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        for idx in missing_indices:
            col = np.random.choice(['latency_ms', 'throughput_mbps', 'jitter_ms'])
            data[col][idx] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        for idx in outlier_indices:
            col = np.random.choice(['latency_ms', 'throughput_mbps'])
            if col == 'latency_ms':
                data[col][idx] = np.random.uniform(500, 1000)
            else:
                data[col][idx] = np.random.uniform(0.1, 1)
        
        # Add inconsistent data
        inconsistent_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
        for idx in inconsistent_indices:
            # High latency but high throughput (inconsistent)
            data['latency_ms'][idx] = np.random.uniform(200, 500)
            data['throughput_mbps'][idx] = np.random.uniform(800, 1000)
