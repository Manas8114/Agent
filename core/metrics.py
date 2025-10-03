"""
Metrics Collector for Enhanced Telecom AI System

This module provides comprehensive metrics collection and analysis
for the Enhanced Telecom AI System.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
import time
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Comprehensive metrics collector for the Enhanced Telecom AI System.
    
    Collects, stores, and analyzes metrics from all system components.
    """
    
    def __init__(self, max_history_size: int = 10000):
        """
        Initialize the metrics collector.
        
        Args:
            max_history_size: Maximum number of metrics to keep in history
        """
        self.max_history_size = max_history_size
        self.metrics_history = deque(maxlen=max_history_size)
        self.agent_metrics = defaultdict(list)
        self.system_metrics = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        self.lock = threading.Lock()
        
        # Initialize metric categories
        self.metric_categories = {
            'agent_performance': [
                'accuracy', 'precision', 'recall', 'f1_score', 'auc_score',
                'mae', 'mse', 'rmse', 'r2_score', 'anomaly_rate'
            ],
            'system_performance': [
                'cpu_usage', 'memory_usage', 'disk_usage', 'network_usage',
                'response_time', 'throughput', 'error_rate', 'availability'
            ],
            'business_metrics': [
                'energy_savings', 'cost_reduction', 'service_quality',
                'user_satisfaction', 'network_efficiency', 'security_score'
            ],
            'coordination_metrics': [
                'coordination_score', 'agent_consensus', 'recommendation_quality',
                'decision_accuracy', 'response_time', 'coordination_efficiency'
            ]
        }
        
        logger.info("Metrics Collector initialized")
    
    def record_metric(self, category: str, metric_name: str, value: float, 
                     timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """
        Record a metric value.
        
        Args:
            category: Metric category
            metric_name: Name of the metric
            value: Metric value
            timestamp: Timestamp (defaults to now)
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_entry = {
            'timestamp': timestamp,
            'category': category,
            'metric_name': metric_name,
            'value': value,
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.metrics_history.append(metric_entry)
            
            # Store in category-specific storage
            if category == 'agent_performance':
                self.agent_metrics[metric_name].append((timestamp, value))
            elif category == 'system_performance':
                self.system_metrics[metric_name].append((timestamp, value))
            elif category == 'business_metrics':
                self.performance_metrics[metric_name].append((timestamp, value))
    
    def record_agent_metrics(self, agent_name: str, metrics: Dict[str, float], 
                           timestamp: datetime = None):
        """
        Record metrics for a specific agent.
        
        Args:
            agent_name: Name of the agent
            metrics: Dictionary of metrics
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            self.record_metric(
                'agent_performance',
                f"{agent_name}_{metric_name}",
                value,
                timestamp,
                {'agent_name': agent_name}
            )
    
    def record_system_metrics(self, metrics: Dict[str, float], 
                            timestamp: datetime = None):
        """
        Record system-level metrics.
        
        Args:
            metrics: Dictionary of system metrics
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            self.record_metric(
                'system_performance',
                metric_name,
                value,
                timestamp
            )
    
    def record_business_metrics(self, metrics: Dict[str, float], 
                              timestamp: datetime = None):
        """
        Record business-level metrics.
        
        Args:
            metrics: Dictionary of business metrics
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            self.record_metric(
                'business_metrics',
                metric_name,
                value,
                timestamp
            )
    
    def get_metric_history(self, category: str = None, metric_name: str = None, 
                          start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
        """
        Get metric history with optional filtering.
        
        Args:
            category: Filter by category
            metric_name: Filter by metric name
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            List of metric entries
        """
        with self.lock:
            history = list(self.metrics_history)
        
        # Apply filters
        if category:
            history = [entry for entry in history if entry['category'] == category]
        
        if metric_name:
            history = [entry for entry in history if entry['metric_name'] == metric_name]
        
        if start_time:
            history = [entry for entry in history if entry['timestamp'] >= start_time]
        
        if end_time:
            history = [entry for entry in history if entry['timestamp'] <= end_time]
        
        return history
    
    def get_agent_performance_summary(self, agent_name: str = None, 
                                    time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for agents.
        
        Args:
            agent_name: Specific agent name (None for all agents)
            time_window_hours: Time window in hours
            
        Returns:
            Performance summary
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        # Get agent metrics
        agent_metrics = self.get_metric_history(
            category='agent_performance',
            start_time=start_time,
            end_time=end_time
        )
        
        if agent_name:
            agent_metrics = [m for m in agent_metrics 
                           if m['metadata'].get('agent_name') == agent_name]
        
        # Calculate summary statistics
        summary = {}
        metric_values = defaultdict(list)
        
        for metric in agent_metrics:
            metric_name = metric['metric_name']
            value = metric['value']
            metric_values[metric_name].append(value)
        
        for metric_name, values in metric_values.items():
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1]
                }
        
        return summary
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dictionary containing current system metrics
        """
        try:
            with self.lock:
                # Get recent metrics (last 5 minutes)
                recent_metrics = [
                    m for m in self.metrics_history 
                    if m.get('timestamp', datetime.min) > datetime.now() - timedelta(minutes=5)
                ]
                
                if not recent_metrics:
                    # Return default metrics if no recent data
                    return {
                        'cpu_usage': 0.0,
                        'memory_usage': 0.0,
                        'disk_usage': 0.0,
                        'network_latency': 0.0,
                        'response_time': 0.0,
                        'throughput': 0.0,
                        'error_rate': 0.0,
                        'availability': 100.0
                    }
                
                # Calculate current metrics
                system_metrics = {}
                
                # CPU Usage
                cpu_values = [m.get('cpu_usage', 0) for m in recent_metrics if 'cpu_usage' in m]
                system_metrics['cpu_usage'] = np.mean(cpu_values) if cpu_values else 0.0
                
                # Memory Usage
                memory_values = [m.get('memory_usage', 0) for m in recent_metrics if 'memory_usage' in m]
                system_metrics['memory_usage'] = np.mean(memory_values) if memory_values else 0.0
                
                # Disk Usage
                disk_values = [m.get('disk_usage', 0) for m in recent_metrics if 'disk_usage' in m]
                system_metrics['disk_usage'] = np.mean(disk_values) if disk_values else 0.0
                
                # Network Latency
                latency_values = [m.get('network_latency', 0) for m in recent_metrics if 'network_latency' in m]
                system_metrics['network_latency'] = np.mean(latency_values) if latency_values else 0.0
                
                # Response Time
                response_values = [m.get('response_time', 0) for m in recent_metrics if 'response_time' in m]
                system_metrics['response_time'] = np.mean(response_values) if response_values else 0.0
                
                # Throughput
                throughput_values = [m.get('throughput', 0) for m in recent_metrics if 'throughput' in m]
                system_metrics['throughput'] = np.mean(throughput_values) if throughput_values else 0.0
                
                # Error Rate
                error_values = [m.get('error_rate', 0) for m in recent_metrics if 'error_rate' in m]
                system_metrics['error_rate'] = np.mean(error_values) if error_values else 0.0
                
                # Availability
                availability_values = [m.get('availability', 100) for m in recent_metrics if 'availability' in m]
                system_metrics['availability'] = np.mean(availability_values) if availability_values else 100.0
                
                return system_metrics
                
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'network_latency': 0.0,
                'response_time': 0.0,
                'throughput': 0.0,
                'error_rate': 0.0,
                'availability': 100.0
            }
    
    def get_system_health_score(self) -> float:
        """
        Calculate overall system health score.
        
        Returns:
            System health score (0-100)
        """
        # Get recent system metrics
        recent_metrics = self.get_metric_history(
            category='system_performance',
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        if not recent_metrics:
            return 50.0  # Default score if no metrics
        
        # Calculate health components
        health_components = {}
        
        # CPU usage (lower is better)
        cpu_metrics = [m for m in recent_metrics if m['metric_name'] == 'cpu_usage']
        if cpu_metrics:
            avg_cpu = np.mean([m['value'] for m in cpu_metrics])
            health_components['cpu'] = max(0, 100 - avg_cpu)
        
        # Memory usage (lower is better)
        memory_metrics = [m for m in recent_metrics if m['metric_name'] == 'memory_usage']
        if memory_metrics:
            avg_memory = np.mean([m['value'] for m in memory_metrics])
            health_components['memory'] = max(0, 100 - avg_memory)
        
        # Error rate (lower is better)
        error_metrics = [m for m in recent_metrics if m['metric_name'] == 'error_rate']
        if error_metrics:
            avg_error_rate = np.mean([m['value'] for m in error_metrics])
            health_components['error_rate'] = max(0, 100 - (avg_error_rate * 100))
        
        # Availability (higher is better)
        availability_metrics = [m for m in recent_metrics if m['metric_name'] == 'availability']
        if availability_metrics:
            avg_availability = np.mean([m['value'] for m in availability_metrics])
            health_components['availability'] = avg_availability
        
        # Calculate weighted average
        if health_components:
            weights = {'cpu': 0.25, 'memory': 0.25, 'error_rate': 0.25, 'availability': 0.25}
            health_score = sum(health_components.get(metric, 50) * weight 
                             for metric, weight in weights.items())
        else:
            health_score = 50.0
        
        return min(100, max(0, health_score))
    
    def get_business_impact_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get business impact metrics.
        
        Args:
            time_window_hours: Time window in hours
            
        Returns:
            Business impact metrics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        business_metrics = self.get_metric_history(
            category='business_metrics',
            start_time=start_time,
            end_time=end_time
        )
        
        impact_metrics = {}
        
        for metric in business_metrics:
            metric_name = metric['metric_name']
            value = metric['value']
            
            if metric_name not in impact_metrics:
                impact_metrics[metric_name] = []
            impact_metrics[metric_name].append(value)
        
        # Calculate impact summaries
        impact_summary = {}
        for metric_name, values in impact_metrics.items():
            if values:
                impact_summary[metric_name] = {
                    'current_value': values[-1],
                    'average_value': np.mean(values),
                    'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'declining',
                    'total_impact': np.sum(values)
                }
        
        return impact_summary
    
    def get_coordination_analytics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get coordination analytics.
        
        Args:
            time_window_hours: Time window in hours
            
        Returns:
            Coordination analytics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        coordination_metrics = self.get_metric_history(
            category='coordination_metrics',
            start_time=start_time,
            end_time=end_time
        )
        
        analytics = {
            'total_coordination_events': len(coordination_metrics),
            'coordination_score_trend': [],
            'agent_consensus_trend': [],
            'recommendation_quality_trend': []
        }
        
        # Analyze trends
        for metric in coordination_metrics:
            if metric['metric_name'] == 'coordination_score':
                analytics['coordination_score_trend'].append(metric['value'])
            elif metric['metric_name'] == 'agent_consensus':
                analytics['agent_consensus_trend'].append(metric['value'])
            elif metric['metric_name'] == 'recommendation_quality':
                analytics['recommendation_quality_trend'].append(metric['value'])
        
        # Calculate trend statistics
        for trend_name in ['coordination_score_trend', 'agent_consensus_trend', 'recommendation_quality_trend']:
            trend_values = analytics[trend_name]
            if trend_values:
                analytics[f'{trend_name}_avg'] = np.mean(trend_values)
                analytics[f'{trend_name}_std'] = np.std(trend_values)
                analytics[f'{trend_name}_trend'] = 'improving' if len(trend_values) > 1 and trend_values[-1] > trend_values[0] else 'stable'
        
        return analytics
    
    def generate_metrics_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.
        
        Args:
            time_window_hours: Time window in hours
            
        Returns:
            Comprehensive metrics report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_window_hours': time_window_hours,
            'system_health_score': self.get_system_health_score(),
            'agent_performance': self.get_agent_performance_summary(time_window_hours=time_window_hours),
            'business_impact': self.get_business_impact_metrics(time_window_hours),
            'coordination_analytics': self.get_coordination_analytics(time_window_hours),
            'metric_categories': self.metric_categories,
            'total_metrics_collected': len(self.metrics_history)
        }
        
        return report
    
    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """
        Export metrics to file.
        
        Args:
            filepath: Path to export file
            format: Export format ('json', 'csv')
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(list(self.metrics_history), f, default=str, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(list(self.metrics_history))
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metrics exported to {filepath}")
    
    def clear_old_metrics(self, older_than_hours: int = 168) -> int:
        """
        Clear metrics older than specified hours.
        
        Args:
            older_than_hours: Age threshold in hours
            
        Returns:
            Number of metrics cleared
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self.lock:
            original_size = len(self.metrics_history)
            
            # Filter out old metrics
            self.metrics_history = deque(
                [m for m in self.metrics_history if m['timestamp'] > cutoff_time],
                maxlen=self.max_history_size
            )
            
            cleared_count = original_size - len(self.metrics_history)
        
        logger.info(f"Cleared {cleared_count} old metrics")
        return cleared_count
    
    def get_metric_alerts(self, alert_thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Get metric alerts based on thresholds.
        
        Args:
            alert_thresholds: Dictionary of metric thresholds
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Get recent metrics
        recent_metrics = self.get_metric_history(
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        for metric in recent_metrics:
            metric_name = metric['metric_name']
            value = metric['value']
            
            if metric_name in alert_thresholds:
                thresholds = alert_thresholds[metric_name]
                
                # Check for threshold violations
                if 'max' in thresholds and value > thresholds['max']:
                    alerts.append({
                        'type': 'threshold_exceeded',
                        'metric_name': metric_name,
                        'value': value,
                        'threshold': thresholds['max'],
                        'severity': 'high',
                        'timestamp': metric['timestamp']
                    })
                elif 'min' in thresholds and value < thresholds['min']:
                    alerts.append({
                        'type': 'threshold_below',
                        'metric_name': metric_name,
                        'value': value,
                        'threshold': thresholds['min'],
                        'severity': 'medium',
                        'timestamp': metric['timestamp']
                    })
        
        return alerts
