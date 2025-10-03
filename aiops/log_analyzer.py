#!/usr/bin/env python3
"""
AIOps Log Analyzer with Root Cause Analysis
Implements intelligent log parsing and automated root cause analysis
"""

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict, Counter
import asyncio
import aiofiles
from pathlib import Path

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    DATABASE_QUERY = "database_query"
    ML_MODEL_TRAINING = "ml_model_training"
    ML_MODEL_PREDICTION = "ml_model_prediction"
    AGENT_COORDINATION = "agent_coordination"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_ISSUE = "performance_issue"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class LogEntry:
    """Represents a parsed log entry"""
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    event_type: EventType
    metadata: Dict[str, Any]
    raw_line: str

@dataclass
class AnomalyPattern:
    """Represents an anomaly pattern in logs"""
    pattern_id: str
    name: str
    description: str
    regex_pattern: str
    severity: LogLevel
    event_type: EventType
    root_cause: str
    remediation: str

class LogParser:
    """Intelligent log parser with pattern recognition"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.anomaly_patterns = self._load_anomaly_patterns()
        self.service_patterns = self._load_service_patterns()
    
    def _load_anomaly_patterns(self) -> List[AnomalyPattern]:
        """Load predefined anomaly patterns"""
        return [
            AnomalyPattern(
                pattern_id="high_latency",
                name="High Latency Detection",
                description="Detects high latency in API responses",
                regex_pattern=r"latency.*?(\d+\.?\d*)\s*ms.*?(?:high|exceeded|critical)",
                severity=LogLevel.WARNING,
                event_type=EventType.PERFORMANCE_ISSUE,
                root_cause="Network congestion or server overload",
                remediation="Scale resources or optimize network routing"
            ),
            AnomalyPattern(
                pattern_id="database_connection_failure",
                name="Database Connection Failure",
                description="Detects database connection issues",
                regex_pattern=r"(?:connection|connect).*?(?:failed|error|timeout|refused)",
                severity=LogLevel.ERROR,
                event_type=EventType.ERROR_OCCURRED,
                root_cause="Database server unavailable or network issues",
                remediation="Check database status and network connectivity"
            ),
            AnomalyPattern(
                pattern_id="memory_exhaustion",
                name="Memory Exhaustion",
                description="Detects memory exhaustion issues",
                regex_pattern=r"(?:memory|ram).*?(?:exhausted|full|out of memory|oom)",
                severity=LogLevel.CRITICAL,
                event_type=EventType.PERFORMANCE_ISSUE,
                root_cause="Insufficient memory allocation or memory leak",
                remediation="Increase memory allocation or fix memory leaks"
            ),
            AnomalyPattern(
                pattern_id="security_breach",
                name="Security Breach Attempt",
                description="Detects potential security breaches",
                regex_pattern=r"(?:unauthorized|suspicious|attack|breach|intrusion)",
                severity=LogLevel.CRITICAL,
                event_type=EventType.SECURITY_ALERT,
                root_cause="Potential security attack or unauthorized access",
                remediation="Block suspicious IPs and increase security monitoring"
            ),
            AnomalyPattern(
                pattern_id="ml_model_failure",
                name="ML Model Failure",
                description="Detects ML model training or prediction failures",
                regex_pattern=r"(?:model|ml|training|prediction).*?(?:failed|error|exception)",
                severity=LogLevel.ERROR,
                event_type=EventType.ML_MODEL_TRAINING,
                root_cause="ML model training failed or prediction error",
                remediation="Check model data quality and retrain if necessary"
            ),
            AnomalyPattern(
                pattern_id="agent_coordination_failure",
                name="Agent Coordination Failure",
                description="Detects AI agent coordination issues",
                regex_pattern=r"(?:agent|coordination).*?(?:failed|error|timeout|disconnected)",
                severity=LogLevel.ERROR,
                event_type=EventType.AGENT_COORDINATION,
                root_cause="AI agent communication or coordination failure",
                remediation="Restart agents and check communication channels"
            )
        ]
    
    def _load_service_patterns(self) -> Dict[str, str]:
        """Load service identification patterns"""
        return {
            'api_server': r'(?:api|server|endpoint)',
            'database': r'(?:database|db|sql|postgres|mysql)',
            'ml_agent': r'(?:agent|ml|model|ai)',
            'monitoring': r'(?:monitor|prometheus|grafana|metrics)',
            'security': r'(?:security|auth|login|token)',
            'network': r'(?:network|traffic|bandwidth|latency)'
        }
    
    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line"""
        try:
            # Common log format: [timestamp] level service: message
            timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]'
            level_pattern = r'(DEBUG|INFO|WARNING|ERROR|CRITICAL)'
            service_pattern = r'(\w+):'
            message_pattern = r':\s*(.*)'
            
            full_pattern = f'{timestamp_pattern}\\s+{level_pattern}\\s+{service_pattern}{message_pattern}'
            
            match = re.match(full_pattern, line.strip())
            if not match:
                return None
            
            timestamp_str, level_str, service, message = match.groups()
            
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f' if '.' in timestamp_str else '%Y-%m-%d %H:%M:%S')
            
            # Parse log level
            level = LogLevel(level_str)
            
            # Determine event type
            event_type = self._classify_event_type(message, service)
            
            # Extract metadata
            metadata = self._extract_metadata(message)
            
            return LogEntry(
                timestamp=timestamp,
                level=level,
                service=service,
                message=message,
                event_type=event_type,
                metadata=metadata,
                raw_line=line
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse log line: {e}")
            return None
    
    def _classify_event_type(self, message: str, service: str) -> EventType:
        """Classify event type based on message content"""
        message_lower = message.lower()
        
        if 'startup' in message_lower or 'started' in message_lower:
            return EventType.SYSTEM_STARTUP
        elif 'shutdown' in message_lower or 'stopped' in message_lower:
            return EventType.SYSTEM_SHUTDOWN
        elif 'request' in message_lower or 'api' in message_lower:
            return EventType.API_REQUEST
        elif 'response' in message_lower or 'status' in message_lower:
            return EventType.API_RESPONSE
        elif 'database' in message_lower or 'query' in message_lower:
            return EventType.DATABASE_QUERY
        elif 'training' in message_lower or 'model' in message_lower:
            return EventType.ML_MODEL_TRAINING
        elif 'prediction' in message_lower or 'inference' in message_lower:
            return EventType.ML_MODEL_PREDICTION
        elif 'coordination' in message_lower or 'agent' in message_lower:
            return EventType.AGENT_COORDINATION
        elif 'security' in message_lower or 'alert' in message_lower:
            return EventType.SECURITY_ALERT
        elif 'performance' in message_lower or 'slow' in message_lower:
            return EventType.PERFORMANCE_ISSUE
        else:
            return EventType.ERROR_OCCURRED
    
    def _extract_metadata(self, message: str) -> Dict[str, Any]:
        """Extract metadata from log message"""
        metadata = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+\.?\d*', message)
        if numbers:
            metadata['numbers'] = [float(n) if '.' in n else int(n) for n in numbers]
        
        # Extract IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, message)
        if ips:
            metadata['ip_addresses'] = ips
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, message)
        if urls:
            metadata['urls'] = urls
        
        # Extract error codes
        error_codes = re.findall(r'error\s*:?\s*(\d+)', message, re.IGNORECASE)
        if error_codes:
            metadata['error_codes'] = [int(code) for code in error_codes]
        
        return metadata

class AnomalyDetector:
    """Detects anomalies in log patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = LogParser()
        self.anomaly_history = []
    
    def detect_anomalies(self, log_entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect anomalies in log entries"""
        anomalies = []
        
        for entry in log_entries:
            # Check against predefined patterns
            for pattern in self.parser.anomaly_patterns:
                if re.search(pattern.regex_pattern, entry.message, re.IGNORECASE):
                    anomaly = {
                        'timestamp': entry.timestamp,
                        'pattern_id': pattern.pattern_id,
                        'pattern_name': pattern.name,
                        'description': pattern.description,
                        'severity': pattern.severity.value,
                        'event_type': pattern.event_type.value,
                        'root_cause': pattern.root_cause,
                        'remediation': pattern.remediation,
                        'service': entry.service,
                        'message': entry.message,
                        'metadata': entry.metadata
                    }
                    anomalies.append(anomaly)
        
        # Detect statistical anomalies
        statistical_anomalies = self._detect_statistical_anomalies(log_entries)
        anomalies.extend(statistical_anomalies)
        
        # Detect temporal anomalies
        temporal_anomalies = self._detect_temporal_anomalies(log_entries)
        anomalies.extend(temporal_anomalies)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, log_entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in log patterns"""
        anomalies = []
        
        # Group by service and event type
        service_events = defaultdict(list)
        for entry in log_entries:
            key = f"{entry.service}_{entry.event_type.value}"
            service_events[key].append(entry)
        
        for key, entries in service_events.items():
            if len(entries) < 10:  # Need minimum data for statistical analysis
                continue
            
            # Check for unusual error rates
            error_count = sum(1 for entry in entries if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL])
            error_rate = error_count / len(entries)
            
            if error_rate > 0.1:  # More than 10% errors
                anomalies.append({
                    'timestamp': entries[-1].timestamp,
                    'pattern_id': 'high_error_rate',
                    'pattern_name': 'High Error Rate',
                    'description': f'High error rate detected: {error_rate:.2%}',
                    'severity': LogLevel.WARNING.value,
                    'event_type': EventType.PERFORMANCE_ISSUE.value,
                    'root_cause': 'System experiencing high error rate',
                    'remediation': 'Investigate and fix underlying issues',
                    'service': entries[0].service,
                    'message': f'Error rate: {error_rate:.2%}',
                    'metadata': {'error_rate': error_rate, 'total_entries': len(entries)}
                })
        
        return anomalies
    
    def _detect_temporal_anomalies(self, log_entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect temporal anomalies in log patterns"""
        anomalies = []
        
        # Group by time windows
        time_windows = defaultdict(list)
        for entry in log_entries:
            # Group by 5-minute windows
            window_key = entry.timestamp.replace(minute=entry.timestamp.minute // 5 * 5, second=0, microsecond=0)
            time_windows[window_key].append(entry)
        
        # Check for unusual activity patterns
        for window, entries in time_windows.items():
            if len(entries) > 100:  # Unusually high activity
                anomalies.append({
                    'timestamp': window,
                    'pattern_id': 'high_activity',
                    'pattern_name': 'High Activity Period',
                    'description': f'Unusually high activity: {len(entries)} events in 5 minutes',
                    'severity': LogLevel.WARNING.value,
                    'event_type': EventType.PERFORMANCE_ISSUE.value,
                    'root_cause': 'System experiencing high activity load',
                    'remediation': 'Monitor system resources and scale if necessary',
                    'service': 'system',
                    'message': f'High activity detected: {len(entries)} events',
                    'metadata': {'event_count': len(entries), 'time_window': '5 minutes'}
                })
        
        return anomalies

class RootCauseAnalyzer:
    """Analyzes root causes of anomalies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_rules = self._load_correlation_rules()
    
    def _load_correlation_rules(self) -> List[Dict[str, Any]]:
        """Load correlation rules for root cause analysis"""
        return [
            {
                'name': 'Database Connection Issues',
                'patterns': ['database_connection_failure', 'high_latency'],
                'root_cause': 'Database server issues causing system-wide problems',
                'confidence': 0.9
            },
            {
                'name': 'Memory Exhaustion Cascade',
                'patterns': ['memory_exhaustion', 'high_activity'],
                'root_cause': 'Memory exhaustion causing system instability',
                'confidence': 0.95
            },
            {
                'name': 'Security Attack Pattern',
                'patterns': ['security_breach', 'high_activity', 'high_error_rate'],
                'root_cause': 'Potential security attack causing system overload',
                'confidence': 0.85
            },
            {
                'name': 'ML Model Failure Impact',
                'patterns': ['ml_model_failure', 'agent_coordination_failure'],
                'root_cause': 'ML model failure affecting agent coordination',
                'confidence': 0.8
            }
        ]
    
    def analyze_root_cause(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze root cause of anomalies"""
        if not anomalies:
            return {'root_cause': 'No anomalies detected', 'confidence': 0.0}
        
        # Group anomalies by time window
        time_groups = defaultdict(list)
        for anomaly in anomalies:
            # Group by 10-minute windows
            window_key = anomaly['timestamp'].replace(minute=anomaly['timestamp'].minute // 10 * 10, second=0, microsecond=0)
            time_groups[window_key].append(anomaly)
        
        # Find the most significant time window
        max_anomalies = 0
        significant_window = None
        for window, window_anomalies in time_groups.items():
            if len(window_anomalies) > max_anomalies:
                max_anomalies = len(window_anomalies)
                significant_window = window
        
        if not significant_window:
            return {'root_cause': 'Isolated anomalies detected', 'confidence': 0.3}
        
        window_anomalies = time_groups[significant_window]
        
        # Apply correlation rules
        best_match = None
        best_confidence = 0.0
        
        for rule in self.correlation_rules:
            pattern_matches = sum(1 for anomaly in window_anomalies 
                                if anomaly['pattern_id'] in rule['patterns'])
            
            if pattern_matches >= len(rule['patterns']):
                confidence = rule['confidence'] * (pattern_matches / len(rule['patterns']))
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = rule
        
        if best_match:
            return {
                'root_cause': best_match['root_cause'],
                'confidence': best_confidence,
                'rule_name': best_match['name'],
                'anomaly_count': len(window_anomalies),
                'time_window': significant_window,
                'affected_services': list(set(anomaly['service'] for anomaly in window_anomalies))
            }
        else:
            # Fallback analysis
            severity_counts = Counter(anomaly['severity'] for anomaly in window_anomalies)
            most_common_severity = severity_counts.most_common(1)[0][0]
            
            return {
                'root_cause': f'Multiple {most_common_severity.lower()} issues detected',
                'confidence': 0.5,
                'anomaly_count': len(window_anomalies),
                'time_window': significant_window,
                'affected_services': list(set(anomaly['service'] for anomaly in window_anomalies))
            }

class AIOpsEngine:
    """Main AIOps engine for log analysis and root cause analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = LogParser()
        self.anomaly_detector = AnomalyDetector()
        self.root_cause_analyzer = RootCauseAnalyzer()
    
    async def analyze_logs(self, log_file_path: str) -> Dict[str, Any]:
        """Analyze logs from a file"""
        try:
            # Read log file
            log_entries = []
            async with aiofiles.open(log_file_path, 'r') as f:
                async for line in f:
                    entry = self.parser.parse_log_line(line.strip())
                    if entry:
                        log_entries.append(entry)
            
            self.logger.info(f"Parsed {len(log_entries)} log entries")
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(log_entries)
            self.logger.info(f"Detected {len(anomalies)} anomalies")
            
            # Analyze root cause
            root_cause_analysis = self.root_cause_analyzer.analyze_root_cause(anomalies)
            
            # Generate summary
            summary = {
                'total_log_entries': len(log_entries),
                'anomalies_detected': len(anomalies),
                'anomaly_rate': len(anomalies) / len(log_entries) if log_entries else 0,
                'root_cause_analysis': root_cause_analysis,
                'anomalies': anomalies,
                'log_entries': log_entries
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to analyze logs: {e}")
            raise
    
    async def analyze_realtime_logs(self, log_stream) -> Dict[str, Any]:
        """Analyze real-time log stream"""
        try:
            log_entries = []
            anomalies = []
            
            async for line in log_stream:
                entry = self.parser.parse_log_line(line.strip())
                if entry:
                    log_entries.append(entry)
                    
                    # Check for immediate anomalies
                    entry_anomalies = self.anomaly_detector.detect_anomalies([entry])
                    anomalies.extend(entry_anomalies)
                    
                    # If anomalies found, analyze root cause
                    if entry_anomalies:
                        root_cause_analysis = self.root_cause_analyzer.analyze_root_cause(anomalies)
                        
                        yield {
                            'timestamp': datetime.now(),
                            'anomalies': entry_anomalies,
                            'root_cause_analysis': root_cause_analysis,
                            'total_anomalies': len(anomalies)
                        }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze real-time logs: {e}")
            raise
    
    def generate_incident_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate incident report from analysis"""
        report = f"""
# Incident Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Log Entries**: {analysis_result['total_log_entries']}
- **Anomalies Detected**: {analysis_result['anomalies_detected']}
- **Anomaly Rate**: {analysis_result['anomaly_rate']:.2%}

## Root Cause Analysis
- **Root Cause**: {analysis_result['root_cause_analysis']['root_cause']}
- **Confidence**: {analysis_result['root_cause_analysis']['confidence']:.2%}
- **Affected Services**: {', '.join(analysis_result['root_cause_analysis'].get('affected_services', []))}

## Anomalies Detected
"""
        
        for i, anomaly in enumerate(analysis_result['anomalies'][:10], 1):  # Show first 10
            report += f"""
### {i}. {anomaly['pattern_name']}
- **Severity**: {anomaly['severity']}
- **Service**: {anomaly['service']}
- **Description**: {anomaly['description']}
- **Root Cause**: {anomaly['root_cause']}
- **Remediation**: {anomaly['remediation']}
- **Timestamp**: {anomaly['timestamp']}
"""
        
        return report

# Example usage and testing
async def main():
    """Example usage of AIOps engine"""
    logging.basicConfig(level=logging.INFO)
    
    # Create AIOps engine
    aiops = AIOpsEngine()
    
    # Generate sample log data
    sample_logs = [
        "[2023-12-01 10:00:00] INFO api_server: API request received",
        "[2023-12-01 10:00:01] WARNING api_server: High latency detected: 150ms",
        "[2023-12-01 10:00:02] ERROR database: Connection failed to database server",
        "[2023-12-01 10:00:03] CRITICAL system: Memory exhausted, out of memory",
        "[2023-12-01 10:00:04] ERROR ml_agent: Model training failed",
        "[2023-12-01 10:00:05] WARNING security: Suspicious activity detected",
        "[2023-12-01 10:00:06] ERROR agent_coordinator: Agent coordination failed",
        "[2023-12-01 10:00:07] INFO api_server: API response sent",
        "[2023-12-01 10:00:08] WARNING monitoring: High CPU usage detected",
        "[2023-12-01 10:00:09] ERROR database: Query timeout exceeded"
    ]
    
    # Write sample logs to file
    with open('sample_logs.txt', 'w') as f:
        for log in sample_logs:
            f.write(log + '\n')
    
    # Analyze logs
    result = await aiops.analyze_logs('sample_logs.txt')
    
    print("AIOps Analysis Results:")
    print(f"Total log entries: {result['total_log_entries']}")
    print(f"Anomalies detected: {result['anomalies_detected']}")
    print(f"Root cause: {result['root_cause_analysis']['root_cause']}")
    print(f"Confidence: {result['root_cause_analysis']['confidence']:.2%}")
    
    # Generate incident report
    report = aiops.generate_incident_report(result)
    print("\nIncident Report:")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
