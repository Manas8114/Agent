#!/usr/bin/env python3
"""
Real Data Sources for Enhanced Telecom AI System
Implements real data collection from actual sources where possible
"""

import asyncio
import aiohttp
import psutil
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import subprocess
import platform

logger = logging.getLogger(__name__)

@dataclass
class RealTimeMetric:
    """Real-time metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    source: str

class RealDataCollector:
    """Collects real data from actual system sources"""
    
    def __init__(self):
        self.session = None
        self.metrics_buffer = []
        self.is_running = False
        
    async def start(self):
        """Start real data collection"""
        self.session = aiohttp.ClientSession()
        self.is_running = True
        logger.info("Real data collector started")
        
    async def stop(self):
        """Stop real data collection"""
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info("Real data collector stopped")
    
    async def collect_system_metrics(self) -> List[RealTimeMetric]:
        """Collect real system metrics"""
        metrics = []
        now = datetime.now()
        
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                source="system"
            ))
            
            # Memory Usage
            memory = psutil.virtual_memory()
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="memory_usage",
                value=memory.percent,
                unit="percent",
                source="system"
            ))
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="disk_usage",
                value=disk_percent,
                unit="percent",
                source="system"
            ))
            
            # Network I/O
            net_io = psutil.net_io_counters()
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="network_bytes_sent",
                value=net_io.bytes_sent,
                unit="bytes",
                source="system"
            ))
            
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="network_bytes_recv",
                value=net_io.bytes_recv,
                unit="bytes",
                source="system"
            ))
            
            # Process count
            process_count = len(psutil.pids())
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="process_count",
                value=process_count,
                unit="count",
                source="system"
            ))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
        return metrics
    
    async def collect_network_metrics(self) -> List[RealTimeMetric]:
        """Collect real network metrics"""
        metrics = []
        now = datetime.now()
        
        try:
            # Network latency to localhost
            start_time = time.time()
            result = subprocess.run(['ping', '-c', '1', 'localhost'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                latency = (time.time() - start_time) * 1000  # Convert to ms
                metrics.append(RealTimeMetric(
                    timestamp=now,
                    metric_name="network_latency",
                    value=latency,
                    unit="milliseconds",
                    source="network"
                ))
            
            # Network connections
            connections = len(psutil.net_connections())
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="network_connections",
                value=connections,
                unit="count",
                source="network"
            ))
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            
        return metrics
    
    async def collect_api_metrics(self) -> List[RealTimeMetric]:
        """Collect real API metrics"""
        metrics = []
        now = datetime.now()
        
        try:
            # Test API response time
            start_time = time.time()
            async with self.session.get('http://localhost:8000/api/v1/health') as response:
                response_time = (time.time() - start_time) * 1000
                metrics.append(RealTimeMetric(
                    timestamp=now,
                    metric_name="api_response_time",
                    value=response_time,
                    unit="milliseconds",
                    source="api"
                ))
                
                if response.status == 200:
                    metrics.append(RealTimeMetric(
                        timestamp=now,
                        metric_name="api_health_status",
                        value=1.0,
                        unit="status",
                        source="api"
                    ))
                else:
                    metrics.append(RealTimeMetric(
                        timestamp=now,
                        metric_name="api_health_status",
                        value=0.0,
                        unit="status",
                        source="api"
                    ))
                    
        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}")
            # Add error metric
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="api_health_status",
                value=0.0,
                unit="status",
                source="api"
            ))
            
        return metrics
    
    async def collect_telecom_simulation_metrics(self) -> List[RealTimeMetric]:
        """Collect realistic telecom simulation metrics"""
        metrics = []
        now = datetime.now()
        
        try:
            # Simulate realistic telecom KPIs based on system load
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # QoS Metrics (based on system performance)
            latency = max(5.0, cpu_usage * 0.1 + np.random.normal(10, 2))
            throughput = max(100, 1000 - (cpu_usage * 5) + np.random.normal(0, 50))
            jitter = max(0.1, latency * 0.05 + np.random.normal(0, 0.5))
            packet_loss = max(0.0, min(5.0, (100 - memory_usage) * 0.1 + np.random.normal(0, 0.5)))
            
            metrics.extend([
                RealTimeMetric(now, "qos_latency", latency, "ms", "telecom_sim"),
                RealTimeMetric(now, "qos_throughput", throughput, "Mbps", "telecom_sim"),
                RealTimeMetric(now, "qos_jitter", jitter, "ms", "telecom_sim"),
                RealTimeMetric(now, "qos_packet_loss", packet_loss, "percent", "telecom_sim"),
                RealTimeMetric(now, "qos_availability", max(95.0, 100 - packet_loss), "percent", "telecom_sim"),
            ])
            
            # Energy metrics (based on CPU usage)
            energy_consumption = cpu_usage * 0.5 + np.random.normal(50, 10)
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="energy_consumption",
                value=energy_consumption,
                unit="watts",
                source="telecom_sim"
            ))
            
            # Security metrics (simulated based on system activity)
            security_score = max(70.0, 100 - (cpu_usage * 0.2) + np.random.normal(0, 5))
            metrics.append(RealTimeMetric(
                timestamp=now,
                metric_name="security_score",
                value=security_score,
                unit="score",
                source="telecom_sim"
            ))
            
        except Exception as e:
            logger.error(f"Error collecting telecom simulation metrics: {e}")
            
        return metrics
    
    async def collect_all_metrics(self) -> List[RealTimeMetric]:
        """Collect all available real metrics"""
        all_metrics = []
        
        try:
            # System metrics
            system_metrics = await self.collect_system_metrics()
            all_metrics.extend(system_metrics)
            
            # Network metrics
            network_metrics = await self.collect_network_metrics()
            all_metrics.extend(network_metrics)
            
            # API metrics
            api_metrics = await self.collect_api_metrics()
            all_metrics.extend(api_metrics)
            
            # Telecom simulation metrics
            telecom_metrics = await self.collect_telecom_simulation_metrics()
            all_metrics.extend(telecom_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting all metrics: {e}")
            
        return all_metrics
    
    def get_metrics_dataframe(self, metrics: List[RealTimeMetric]) -> pd.DataFrame:
        """Convert metrics to DataFrame"""
        data = []
        for metric in metrics:
            data.append({
                'timestamp': metric.timestamp,
                'metric_name': metric.metric_name,
                'value': metric.value,
                'unit': metric.unit,
                'source': metric.source
            })
        
        return pd.DataFrame(data)
    
    async def start_continuous_collection(self, interval: int = 5):
        """Start continuous real data collection"""
        logger.info(f"Starting continuous data collection every {interval} seconds")
        
        while self.is_running:
            try:
                metrics = await self.collect_all_metrics()
                self.metrics_buffer.extend(metrics)
                
                # Keep only last 1000 metrics to prevent memory issues
                if len(self.metrics_buffer) > 1000:
                    self.metrics_buffer = self.metrics_buffer[-1000:]
                
                logger.info(f"Collected {len(metrics)} real metrics")
                
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
            
            await asyncio.sleep(interval)
    
    def get_latest_metrics(self, limit: int = 100) -> List[RealTimeMetric]:
        """Get latest collected metrics"""
        return self.metrics_buffer[-limit:] if self.metrics_buffer else []
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        if not self.metrics_buffer:
            return {"status": "no_data"}
        
        df = self.get_metrics_dataframe(self.metrics_buffer)
        
        summary = {
            "total_metrics": len(self.metrics_buffer),
            "sources": df['source'].unique().tolist(),
            "metric_types": df['metric_name'].unique().tolist(),
            "latest_timestamp": df['timestamp'].max().isoformat() if not df.empty else None,
            "earliest_timestamp": df['timestamp'].min().isoformat() if not df.empty else None
        }
        
        return summary

# Global collector instance
real_data_collector = RealDataCollector()

async def start_real_data_collection():
    """Start real data collection"""
    await real_data_collector.start()
    # Start continuous collection in background
    asyncio.create_task(real_data_collector.start_continuous_collection())

def get_real_metrics() -> List[RealTimeMetric]:
    """Get latest real metrics"""
    return real_data_collector.get_latest_metrics()

def get_real_metrics_summary() -> Dict[str, Any]:
    """Get real metrics summary"""
    return real_data_collector.get_metrics_summary()

