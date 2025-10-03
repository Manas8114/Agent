#!/usr/bin/env python3
"""
IoT & Cross-Domain Integration for Telecom AI 3.0
Enables ingestion of IoT/satellite/cloud-native KPIs and cross-domain optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid
import requests
from abc import ABC, abstractmethod

# IoT and cloud integration imports
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("MQTT not available. Install with: pip install paho-mqtt")

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("AWS SDK not available. Install with: pip install boto3")

class IoTDeviceType(Enum):
    """IoT device types"""
    SMART_CITY_SENSOR = "smart_city_sensor"
    TRAFFIC_SENSOR = "traffic_sensor"
    AIR_QUALITY_SENSOR = "air_quality_sensor"
    WEATHER_STATION = "weather_station"
    SATELLITE_LINK = "satellite_link"
    CLOUD_APPLICATION = "cloud_application"
    EDGE_DEVICE = "edge_device"

class DataSource(Enum):
    """Data source types"""
    IOT_SENSORS = "iot_sensors"
    SATELLITE = "satellite"
    CLOUD_APIS = "cloud_apis"
    EDGE_COMPUTING = "edge_computing"
    SMART_CITY = "smart_city"

@dataclass
class IoTDevice:
    """IoT device configuration"""
    device_id: str
    device_type: IoTDeviceType
    location: Dict[str, float]  # lat, lon
    capabilities: List[str]
    data_format: Dict[str, Any]
    update_interval: int  # seconds
    last_update: datetime
    status: str = "active"

@dataclass
class IoTDataPoint:
    """IoT data point"""
    device_id: str
    timestamp: datetime
    data: Dict[str, Any]
    quality_score: float
    source: DataSource

@dataclass
class CrossDomainKPI:
    """Cross-domain KPI"""
    kpi_id: str
    name: str
    value: float
    unit: str
    domain: str
    timestamp: datetime
    confidence: float

class IoTDataCollector(ABC):
    """Abstract base class for IoT data collectors"""
    
    @abstractmethod
    async def collect_data(self) -> List[IoTDataPoint]:
        """Collect data from IoT devices"""
        pass
    
    @abstractmethod
    def get_device_status(self) -> Dict[str, Any]:
        """Get device status"""
        pass

class SmartCityCollector(IoTDataCollector):
    """Smart city sensor data collector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.devices = {}
        self._initialize_devices()
    
    def _initialize_devices(self):
        """Initialize smart city devices"""
        # Traffic sensors
        for i in range(5):
            device_id = f"traffic_sensor_{i}"
            self.devices[device_id] = IoTDevice(
                device_id=device_id,
                device_type=IoTDeviceType.TRAFFIC_SENSOR,
                location={"lat": 40.7128 + np.random.uniform(-0.1, 0.1), "lon": -74.0060 + np.random.uniform(-0.1, 0.1)},
                capabilities=["traffic_count", "speed_measurement", "congestion_detection"],
                data_format={"traffic_count": "int", "avg_speed": "float", "congestion_level": "float"},
                update_interval=30,
                last_update=datetime.now()
            )
        
        # Air quality sensors
        for i in range(3):
            device_id = f"air_quality_sensor_{i}"
            self.devices[device_id] = IoTDevice(
                device_id=device_id,
                device_type=IoTDeviceType.AIR_QUALITY_SENSOR,
                location={"lat": 40.7128 + np.random.uniform(-0.1, 0.1), "lon": -74.0060 + np.random.uniform(-0.1, 0.1)},
                capabilities=["pm2_5", "pm10", "co2", "temperature", "humidity"],
                data_format={"pm2_5": "float", "pm10": "float", "co2": "float", "temperature": "float", "humidity": "float"},
                update_interval=60,
                last_update=datetime.now()
            )
        
        # Weather stations
        for i in range(2):
            device_id = f"weather_station_{i}"
            self.devices[device_id] = IoTDevice(
                device_id=device_id,
                device_type=IoTDeviceType.WEATHER_STATION,
                location={"lat": 40.7128 + np.random.uniform(-0.1, 0.1), "lon": -74.0060 + np.random.uniform(-0.1, 0.1)},
                capabilities=["temperature", "humidity", "pressure", "wind_speed", "precipitation"],
                data_format={"temperature": "float", "humidity": "float", "pressure": "float", "wind_speed": "float", "precipitation": "float"},
                update_interval=300,
                last_update=datetime.now()
            )
    
    async def collect_data(self) -> List[IoTDataPoint]:
        """Collect data from smart city devices"""
        data_points = []
        
        for device_id, device in self.devices.items():
            try:
                # Simulate data collection
                if device.device_type == IoTDeviceType.TRAFFIC_SENSOR:
                    data = {
                        "traffic_count": np.random.randint(10, 100),
                        "avg_speed": np.random.uniform(20, 60),
                        "congestion_level": np.random.uniform(0, 1)
                    }
                elif device.device_type == IoTDeviceType.AIR_QUALITY_SENSOR:
                    data = {
                        "pm2_5": np.random.uniform(10, 50),
                        "pm10": np.random.uniform(20, 80),
                        "co2": np.random.uniform(400, 600),
                        "temperature": np.random.uniform(15, 30),
                        "humidity": np.random.uniform(30, 80)
                    }
                elif device.device_type == IoTDeviceType.WEATHER_STATION:
                    data = {
                        "temperature": np.random.uniform(10, 35),
                        "humidity": np.random.uniform(20, 90),
                        "pressure": np.random.uniform(1000, 1020),
                        "wind_speed": np.random.uniform(0, 20),
                        "precipitation": np.random.uniform(0, 10)
                    }
                else:
                    continue
                
                # Create data point
                data_point = IoTDataPoint(
                    device_id=device_id,
                    timestamp=datetime.now(),
                    data=data,
                    quality_score=np.random.uniform(0.8, 1.0),
                    source=DataSource.SMART_CITY
                )
                
                data_points.append(data_point)
                device.last_update = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Failed to collect data from device {device_id}: {e}")
        
        return data_points
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get device status"""
        return {
            "total_devices": len(self.devices),
            "active_devices": len([d for d in self.devices.values() if d.status == "active"]),
            "device_types": {
                device_type.value: len([d for d in self.devices.values() if d.device_type == device_type])
                for device_type in IoTDeviceType
            },
            "last_update": max([d.last_update for d in self.devices.values()]).isoformat()
        }

class SatelliteCollector(IoTDataCollector):
    """Satellite link data collector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.satellite_links = {}
        self._initialize_satellite_links()
    
    def _initialize_satellite_links(self):
        """Initialize satellite links"""
        for i in range(3):
            link_id = f"satellite_link_{i}"
            self.satellite_links[link_id] = {
                "link_id": link_id,
                "satellite_name": f"SAT-{i+1}",
                "frequency": f"{12 + i}GHz",
                "location": {"lat": 40.7128 + np.random.uniform(-0.5, 0.5), "lon": -74.0060 + np.random.uniform(-0.5, 0.5)},
                "status": "active",
                "last_update": datetime.now()
            }
    
    async def collect_data(self) -> List[IoTDataPoint]:
        """Collect data from satellite links"""
        data_points = []
        
        for link_id, link in self.satellite_links.items():
            try:
                # Simulate satellite data
                data = {
                    "latency_ms": np.random.uniform(200, 800),
                    "throughput_mbps": np.random.uniform(10, 100),
                    "signal_strength": np.random.uniform(-80, -40),
                    "weather_interference": np.random.uniform(0, 1),
                    "link_utilization": np.random.uniform(0.3, 0.9),
                    "error_rate": np.random.uniform(0.001, 0.01)
                }
                
                # Create data point
                data_point = IoTDataPoint(
                    device_id=link_id,
                    timestamp=datetime.now(),
                    data=data,
                    quality_score=np.random.uniform(0.7, 1.0),
                    source=DataSource.SATELLITE
                )
                
                data_points.append(data_point)
                link["last_update"] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Failed to collect data from satellite link {link_id}: {e}")
        
        return data_points
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get satellite link status"""
        return {
            "total_links": len(self.satellite_links),
            "active_links": len([l for l in self.satellite_links.values() if l["status"] == "active"]),
            "last_update": max([l["last_update"] for l in self.satellite_links.values()]).isoformat()
        }

class CloudAPICollector(IoTDataCollector):
    """Cloud API data collector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_endpoints = config.get('api_endpoints', [])
        self._initialize_api_endpoints()
    
    def _initialize_api_endpoints(self):
        """Initialize cloud API endpoints"""
        if not self.api_endpoints:
            # Default API endpoints
            self.api_endpoints = [
                {
                    "name": "cloud_app_1",
                    "url": "https://api.example.com/telemetry",
                    "auth_type": "bearer",
                    "update_interval": 60
                },
                {
                    "name": "cloud_app_2",
                    "url": "https://api.example.com/metrics",
                    "auth_type": "api_key",
                    "update_interval": 30
                }
            ]
    
    async def collect_data(self) -> List[IoTDataPoint]:
        """Collect data from cloud APIs"""
        data_points = []
        
        for endpoint in self.api_endpoints:
            try:
                # Simulate API data collection
                data = {
                    "api_response_time": np.random.uniform(50, 500),
                    "throughput": np.random.uniform(100, 1000),
                    "error_rate": np.random.uniform(0.001, 0.05),
                    "cpu_usage": np.random.uniform(20, 80),
                    "memory_usage": np.random.uniform(30, 90),
                    "active_connections": np.random.randint(100, 1000)
                }
                
                # Create data point
                data_point = IoTDataPoint(
                    device_id=endpoint["name"],
                    timestamp=datetime.now(),
                    data=data,
                    quality_score=np.random.uniform(0.8, 1.0),
                    source=DataSource.CLOUD_APIS
                )
                
                data_points.append(data_point)
                
            except Exception as e:
                self.logger.error(f"Failed to collect data from API {endpoint['name']}: {e}")
        
        return data_points
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get API endpoint status"""
        return {
            "total_endpoints": len(self.api_endpoints),
            "active_endpoints": len(self.api_endpoints),
            "last_update": datetime.now().isoformat()
        }

class IoTIntegrationManager:
    """IoT & Cross-Domain Integration Manager"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Data collectors
        self.collectors = {}
        self._initialize_collectors()
        
        # Data storage
        self.iot_data = []
        self.cross_domain_kpis = []
        
        # Integration settings
        self.data_retention_hours = self.config.get('data_retention_hours', 24)
        self.collection_interval = self.config.get('collection_interval', 30)
        
        # Cross-domain optimization
        self.optimization_enabled = self.config.get('optimization_enabled', True)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        
        # Start data collection
        self.collection_thread = None
        self.is_running = False
    
    def _initialize_collectors(self):
        """Initialize data collectors"""
        # Smart city collector
        self.collectors['smart_city'] = SmartCityCollector(
            self.config.get('smart_city', {})
        )
        
        # Satellite collector
        self.collectors['satellite'] = SatelliteCollector(
            self.config.get('satellite', {})
        )
        
        # Cloud API collector
        self.collectors['cloud_apis'] = CloudAPICollector(
            self.config.get('cloud_apis', {})
        )
    
    def start_collection(self):
        """Start IoT data collection"""
        if self.is_running:
            self.logger.warning("Data collection already running")
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.start()
        
        self.logger.info("Started IoT data collection")
    
    def stop_collection(self):
        """Stop IoT data collection"""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join()
        
        self.logger.info("Stopped IoT data collection")
    
    def _collection_loop(self):
        """Main data collection loop"""
        while self.is_running:
            try:
                # Collect data from all sources
                asyncio.run(self._collect_all_data())
                
                # Process cross-domain correlations
                if self.optimization_enabled:
                    self._process_cross_domain_correlations()
                
                # Clean old data
                self._clean_old_data()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Data collection loop error: {e}")
                time.sleep(5)
    
    async def _collect_all_data(self):
        """Collect data from all sources"""
        all_data = []
        
        for collector_name, collector in self.collectors.items():
            try:
                data_points = await collector.collect_data()
                all_data.extend(data_points)
                
                self.logger.debug(f"Collected {len(data_points)} data points from {collector_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to collect data from {collector_name}: {e}")
        
        # Store data
        self.iot_data.extend(all_data)
        
        # Create cross-domain KPIs
        self._create_cross_domain_kpis(all_data)
    
    def _create_cross_domain_kpis(self, data_points: List[IoTDataPoint]):
        """Create cross-domain KPIs from IoT data"""
        for data_point in data_points:
            try:
                # Extract relevant KPIs based on device type
                if data_point.device_id.startswith("traffic_sensor"):
                    # Traffic-related KPIs
                    if "traffic_count" in data_point.data:
                        kpi = CrossDomainKPI(
                            kpi_id=f"traffic_count_{data_point.device_id}",
                            name="Traffic Count",
                            value=float(data_point.data["traffic_count"]),
                            unit="vehicles/hour",
                            domain="smart_city",
                            timestamp=data_point.timestamp,
                            confidence=data_point.quality_score
                        )
                        self.cross_domain_kpis.append(kpi)
                
                elif data_point.device_id.startswith("air_quality_sensor"):
                    # Air quality KPIs
                    if "pm2_5" in data_point.data:
                        kpi = CrossDomainKPI(
                            kpi_id=f"pm2_5_{data_point.device_id}",
                            name="PM2.5 Level",
                            value=float(data_point.data["pm2_5"]),
                            unit="μg/m³",
                            domain="environmental",
                            timestamp=data_point.timestamp,
                            confidence=data_point.quality_score
                        )
                        self.cross_domain_kpis.append(kpi)
                
                elif data_point.device_id.startswith("satellite_link"):
                    # Satellite KPIs
                    if "latency_ms" in data_point.data:
                        kpi = CrossDomainKPI(
                            kpi_id=f"satellite_latency_{data_point.device_id}",
                            name="Satellite Latency",
                            value=float(data_point.data["latency_ms"]),
                            unit="ms",
                            domain="satellite",
                            timestamp=data_point.timestamp,
                            confidence=data_point.quality_score
                        )
                        self.cross_domain_kpis.append(kpi)
                
                elif data_point.device_id.startswith("cloud_app"):
                    # Cloud application KPIs
                    if "api_response_time" in data_point.data:
                        kpi = CrossDomainKPI(
                            kpi_id=f"api_response_time_{data_point.device_id}",
                            name="API Response Time",
                            value=float(data_point.data["api_response_time"]),
                            unit="ms",
                            domain="cloud",
                            timestamp=data_point.timestamp,
                            confidence=data_point.quality_score
                        )
                        self.cross_domain_kpis.append(kpi)
                
            except Exception as e:
                self.logger.error(f"Failed to create KPI from data point {data_point.device_id}: {e}")
    
    def _process_cross_domain_correlations(self):
        """Process cross-domain correlations for optimization"""
        try:
            # Group KPIs by domain
            domain_kpis = {}
            for kpi in self.cross_domain_kpis:
                if kpi.domain not in domain_kpis:
                    domain_kpis[kpi.domain] = []
                domain_kpis[kpi.domain].append(kpi)
            
            # Find correlations between domains
            correlations = self._find_domain_correlations(domain_kpis)
            
            # Apply cross-domain optimizations
            if correlations:
                self._apply_cross_domain_optimizations(correlations)
            
        except Exception as e:
            self.logger.error(f"Failed to process cross-domain correlations: {e}")
    
    def _find_domain_correlations(self, domain_kpis: Dict[str, List[CrossDomainKPI]]) -> List[Dict[str, Any]]:
        """Find correlations between different domains"""
        correlations = []
        
        domains = list(domain_kpis.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1, domain2 = domains[i], domains[j]
                
                # Find KPIs with similar timestamps
                kpis1 = domain_kpis[domain1]
                kpis2 = domain_kpis[domain2]
                
                # Calculate correlation
                correlation = self._calculate_correlation(kpis1, kpis2)
                
                if correlation > self.correlation_threshold:
                    correlations.append({
                        'domain1': domain1,
                        'domain2': domain2,
                        'correlation': correlation,
                        'kpis1': len(kpis1),
                        'kpis2': len(kpis2)
                    })
        
        return correlations
    
    def _calculate_correlation(self, kpis1: List[CrossDomainKPI], kpis2: List[CrossDomainKPI]) -> float:
        """Calculate correlation between two sets of KPIs"""
        if not kpis1 or not kpis2:
            return 0.0
        
        # Simple correlation based on value similarity
        values1 = [kpi.value for kpi in kpis1]
        values2 = [kpi.value for kpi in kpis2]
        
        # Normalize values
        if values1 and values2:
            values1_norm = [(v - min(values1)) / (max(values1) - min(values1)) if max(values1) > min(values1) else 0.5 for v in values1]
            values2_norm = [(v - min(values2)) / (max(values2) - min(values2)) if max(values2) > min(values2) else 0.5 for v in values2]
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(values1_norm, values2_norm)[0, 1] if len(values1_norm) > 1 and len(values2_norm) > 1 else 0.0
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _apply_cross_domain_optimizations(self, correlations: List[Dict[str, Any]]):
        """Apply cross-domain optimizations based on correlations"""
        for correlation in correlations:
            self.logger.info(f"Applying cross-domain optimization: {correlation['domain1']} ↔ {correlation['domain2']} "
                           f"(correlation: {correlation['correlation']:.3f})")
            
            # Example optimizations based on domain correlations
            if correlation['domain1'] == 'smart_city' and correlation['domain2'] == 'satellite':
                self._optimize_smart_city_satellite(correlation)
            elif correlation['domain1'] == 'cloud' and correlation['domain2'] == 'environmental':
                self._optimize_cloud_environmental(correlation)
    
    def _optimize_smart_city_satellite(self, correlation: Dict[str, Any]):
        """Optimize smart city and satellite integration"""
        self.logger.info("Optimizing smart city and satellite integration")
        # Implement optimization logic
    
    def _optimize_cloud_environmental(self, correlation: Dict[str, Any]):
        """Optimize cloud and environmental integration"""
        self.logger.info("Optimizing cloud and environmental integration")
        # Implement optimization logic
    
    def _clean_old_data(self):
        """Clean old data based on retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=self.data_retention_hours)
        
        # Clean IoT data
        self.iot_data = [dp for dp in self.iot_data if dp.timestamp > cutoff_time]
        
        # Clean cross-domain KPIs
        self.cross_domain_kpis = [kpi for kpi in self.cross_domain_kpis if kpi.timestamp > cutoff_time]
    
    def get_iot_kpis(self, domain: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get IoT KPIs"""
        kpis = self.cross_domain_kpis
        
        if domain:
            kpis = [kpi for kpi in kpis if kpi.domain == domain]
        
        # Return recent KPIs
        recent_kpis = kpis[-limit:] if kpis else []
        
        return [
            {
                'kpi_id': kpi.kpi_id,
                'name': kpi.name,
                'value': kpi.value,
                'unit': kpi.unit,
                'domain': kpi.domain,
                'timestamp': kpi.timestamp.isoformat(),
                'confidence': kpi.confidence
            }
            for kpi in recent_kpis
        ]
    
    def get_cross_domain_status(self) -> Dict[str, Any]:
        """Get cross-domain status"""
        return {
            'collectors': {
                name: collector.get_device_status()
                for name, collector in self.collectors.items()
            },
            'total_data_points': len(self.iot_data),
            'total_kpis': len(self.cross_domain_kpis),
            'domains': list(set(kpi.domain for kpi in self.cross_domain_kpis)),
            'optimization_enabled': self.optimization_enabled,
            'last_update': datetime.now().isoformat()
        }
    
    def get_device_health(self) -> Dict[str, Any]:
        """Get device health status"""
        health_status = {}
        
        for collector_name, collector in self.collectors.items():
            try:
                status = collector.get_device_status()
                health_status[collector_name] = {
                    'status': 'healthy',
                    'details': status
                }
            except Exception as e:
                health_status[collector_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_status

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test IoT Integration Manager
    print("Testing IoT & Cross-Domain Integration Manager...")
    
    iot_manager = IoTIntegrationManager({
        'collection_interval': 10,
        'data_retention_hours': 1,
        'optimization_enabled': True,
        'correlation_threshold': 0.5
    })
    
    # Start data collection
    iot_manager.start_collection()
    
    # Let it collect data for a bit
    time.sleep(30)
    
    # Get IoT KPIs
    kpis = iot_manager.get_iot_kpis()
    print(f"Collected {len(kpis)} IoT KPIs")
    
    # Get cross-domain status
    status = iot_manager.get_cross_domain_status()
    print(f"Cross-domain status: {status}")
    
    # Get device health
    health = iot_manager.get_device_health()
    print(f"Device health: {health}")
    
    # Stop collection
    iot_manager.stop_collection()
    
    print("IoT Integration Manager testing completed!")
