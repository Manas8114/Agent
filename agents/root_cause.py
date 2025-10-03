#!/usr/bin/env python3
"""
GPT-powered Root-Cause Analysis for Enhanced Telecom AI System
Ingests logs, alarms, events and provides intelligent RCA
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import json
import re
import hashlib
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# GPT/LLM imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers")

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    source: str
    message: str
    context: Dict[str, Any]
    raw_log: str

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    source: str
    message: str
    context: Dict[str, Any]
    resolved: bool = False

@dataclass
class RootCauseAnalysis:
    """Root cause analysis result"""
    analysis_id: str
    timestamp: datetime
    primary_cause: str
    contributing_factors: List[str]
    confidence_score: float
    evidence: List[str]
    recommendations: List[str]
    severity: AlertSeverity
    estimated_impact: str
    resolution_steps: List[str]

class LogParser:
    """Advanced log parser for telecom systems"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Log patterns for different telecom components
        self.log_patterns = {
            'gnb': {
                'pattern': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] gNB-(\d+): (.+)',
                'groups': ['timestamp', 'level', 'gnb_id', 'message']
            },
            'ue': {
                'pattern': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] UE-(\w+): (.+)',
                'groups': ['timestamp', 'level', 'ue_id', 'message']
            },
            'core': {
                'pattern': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (\w+): (.+)',
                'groups': ['timestamp', 'level', 'component', 'message']
            },
            'api': {
                'pattern': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] API: (.+)',
                'groups': ['timestamp', 'level', 'message']
            }
        }
        
        # Alert patterns
        self.alert_patterns = {
            'high_latency': r'latency.*(\d+\.?\d*)\s*ms.*threshold',
            'packet_loss': r'packet.*loss.*(\d+\.?\d*)\s*%',
            'connection_failure': r'connection.*failed|timeout|unreachable',
            'resource_exhaustion': r'memory.*exhausted|cpu.*overload|disk.*full',
            'security_breach': r'security.*alert|unauthorized.*access|intrusion'
        }
    
    def parse_log_entry(self, log_line: str) -> Optional[LogEntry]:
        """Parse a single log entry"""
        try:
            # Try different patterns
            for component, pattern_info in self.log_patterns.items():
                match = re.match(pattern_info['pattern'], log_line)
                if match:
                    groups = match.groups()
                    parsed_data = dict(zip(pattern_info['groups'], groups))
                    
                    # Parse timestamp
                    timestamp = datetime.strptime(parsed_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                    
                    # Parse log level
                    level = LogLevel(parsed_data['level'].upper())
                    
                    # Extract context
                    context = {k: v for k, v in parsed_data.items() if k not in ['timestamp', 'level', 'message']}
                    
                    return LogEntry(
                        timestamp=timestamp,
                        level=level,
                        source=component,
                        message=parsed_data.get('message', ''),
                        context=context,
                        raw_log=log_line
                    )
            
            # If no pattern matches, create a generic entry
            return LogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                source='unknown',
                message=log_line,
                context={},
                raw_log=log_line
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse log entry: {e}")
            return None
    
    def detect_alerts(self, log_entry: LogEntry) -> List[Alert]:
        """Detect alerts from log entry"""
        alerts = []
        
        for alert_type, pattern in self.alert_patterns.items():
            if re.search(pattern, log_entry.message, re.IGNORECASE):
                # Determine severity based on log level and content
                severity = self._determine_alert_severity(log_entry, alert_type)
                
                alert = Alert(
                    alert_id=f"alert_{hashlib.md5(log_entry.raw_log.encode()).hexdigest()[:8]}",
                    timestamp=log_entry.timestamp,
                    severity=severity,
                    source=log_entry.source,
                    message=log_entry.message,
                    context=log_entry.context
                )
                alerts.append(alert)
        
        return alerts
    
    def _determine_alert_severity(self, log_entry: LogEntry, alert_type: str) -> AlertSeverity:
        """Determine alert severity based on log entry"""
        # Base severity on log level
        severity_mapping = {
            LogLevel.DEBUG: AlertSeverity.LOW,
            LogLevel.INFO: AlertSeverity.LOW,
            LogLevel.WARNING: AlertSeverity.MEDIUM,
            LogLevel.ERROR: AlertSeverity.HIGH,
            LogLevel.CRITICAL: AlertSeverity.CRITICAL
        }
        
        base_severity = severity_mapping.get(log_entry.level, AlertSeverity.MEDIUM)
        
        # Adjust based on alert type
        if alert_type in ['security_breach', 'connection_failure']:
            if base_severity == AlertSeverity.LOW:
                return AlertSeverity.MEDIUM
            elif base_severity == AlertSeverity.MEDIUM:
                return AlertSeverity.HIGH
        
        return base_severity

class GPTRootCauseAnalyzer:
    """GPT-powered root cause analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # GPT configuration
        self.openai_api_key = self.config.get('openai_api_key')
        self.model_name = self.config.get('model_name', 'gpt-3.5-turbo')
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.3)
        
        # Local LLM configuration
        self.use_local_llm = self.config.get('use_local_llm', False)
        self.local_model_name = self.config.get('local_model_name', 'microsoft/DialoGPT-medium')
        self.local_pipeline = None
        
        # Initialize LLM
        self._initialize_llm()
        
        # Analysis history
        self.analysis_history = []
    
    def _initialize_llm(self):
        """Initialize language model"""
        if self.use_local_llm and TRANSFORMERS_AVAILABLE:
            try:
                self.local_pipeline = pipeline(
                    "text-generation",
                    model=self.local_model_name,
                    tokenizer=self.local_model_name,
                    max_length=500,
                    temperature=self.temperature
                )
                self.logger.info(f"Local LLM initialized: {self.local_model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize local LLM: {e}")
                self.local_pipeline = None
        
        if OPENAI_AVAILABLE and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.logger.info("OpenAI API configured")
    
    def analyze_root_cause(self, logs: List[LogEntry], alerts: List[Alert], 
                          context: Dict[str, Any] = None) -> RootCauseAnalysis:
        """Perform root cause analysis"""
        try:
            # Prepare analysis input
            analysis_input = self._prepare_analysis_input(logs, alerts, context)
            
            # Generate analysis using LLM
            analysis_result = self._generate_analysis(analysis_input)
            
            # Parse and structure the result
            rca = self._parse_analysis_result(analysis_result, logs, alerts)
            
            # Store in history
            self.analysis_history.append(rca)
            
            return rca
            
        except Exception as e:
            self.logger.error(f"Root cause analysis failed: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(logs, alerts)
    
    def _prepare_analysis_input(self, logs: List[LogEntry], alerts: List[Alert], 
                               context: Dict[str, Any] = None) -> str:
        """Prepare input for LLM analysis"""
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)
        
        # Create analysis prompt
        prompt = f"""
        Perform root cause analysis for the following telecom system issues:
        
        CONTEXT:
        {json.dumps(context or {}, indent=2)}
        
        ALERTS ({len(alerts)}):
        """
        
        for alert in alerts:
            prompt += f"""
        - {alert.timestamp.isoformat()} [{alert.severity.value.upper()}] {alert.source}: {alert.message}
        """
        
        prompt += f"""
        
        RECENT LOGS ({len(sorted_logs)}):
        """
        
        for log in sorted_logs[-50:]:  # Last 50 logs
            prompt += f"""
        - {log.timestamp.isoformat()} [{log.level.value}] {log.source}: {log.message}
        """
        
        prompt += """
        
        Please provide:
        1. Primary root cause
        2. Contributing factors
        3. Confidence score (0-1)
        4. Evidence supporting the analysis
        5. Recommended actions
        6. Estimated impact
        7. Resolution steps
        
        Format your response as JSON.
        """
        
        return prompt
    
    def _generate_analysis(self, prompt: str) -> str:
        """Generate analysis using LLM"""
        if self.use_local_llm and self.local_pipeline:
            return self._generate_with_local_llm(prompt)
        elif OPENAI_AVAILABLE and self.openai_api_key:
            return self._generate_with_openai(prompt)
        else:
            return self._generate_fallback_analysis(prompt)
    
    def _generate_with_local_llm(self, prompt: str) -> str:
        """Generate analysis using local LLM"""
        try:
            result = self.local_pipeline(
                prompt,
                max_length=min(len(prompt.split()) + 200, 500),
                num_return_sequences=1,
                temperature=self.temperature,
                do_sample=True
            )
            return result[0]['generated_text']
        except Exception as e:
            self.logger.error(f"Local LLM generation failed: {e}")
            return self._generate_fallback_analysis(prompt)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate analysis using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert telecom network analyst specializing in root cause analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API generation failed: {e}")
            return self._generate_fallback_analysis(prompt)
    
    def _generate_fallback_analysis(self, prompt: str) -> str:
        """Generate fallback analysis without LLM"""
        return json.dumps({
            "primary_cause": "System performance degradation",
            "contributing_factors": ["High network load", "Resource constraints"],
            "confidence_score": 0.6,
            "evidence": ["Multiple error logs", "Performance metrics"],
            "recommendations": ["Monitor system resources", "Check network connectivity"],
            "severity": "medium",
            "estimated_impact": "Service degradation",
            "resolution_steps": ["Restart affected services", "Scale resources"]
        })
    
    def _parse_analysis_result(self, analysis_result: str, logs: List[LogEntry], 
                              alerts: List[Alert]) -> RootCauseAnalysis:
        """Parse LLM analysis result"""
        try:
            # Try to parse as JSON
            if analysis_result.strip().startswith('{'):
                result_data = json.loads(analysis_result)
            else:
                # Extract JSON from text
                json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in analysis result")
            
            # Create RCA object
            rca = RootCauseAnalysis(
                analysis_id=f"rca_{hashlib.md5(analysis_result.encode()).hexdigest()[:8]}",
                timestamp=datetime.now(),
                primary_cause=result_data.get('primary_cause', 'Unknown'),
                contributing_factors=result_data.get('contributing_factors', []),
                confidence_score=float(result_data.get('confidence_score', 0.5)),
                evidence=result_data.get('evidence', []),
                recommendations=result_data.get('recommendations', []),
                severity=AlertSeverity(result_data.get('severity', 'medium')),
                estimated_impact=result_data.get('estimated_impact', 'Unknown'),
                resolution_steps=result_data.get('resolution_steps', [])
            )
            
            return rca
            
        except Exception as e:
            self.logger.warning(f"Failed to parse analysis result: {e}")
            return self._create_fallback_analysis(logs, alerts)
    
    def _create_fallback_analysis(self, logs: List[LogEntry], alerts: List[Alert]) -> RootCauseAnalysis:
        """Create fallback analysis when LLM fails"""
        # Analyze logs and alerts to create basic RCA
        error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        high_severity_alerts = [alert for alert in alerts if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]]
        
        primary_cause = "System error detected"
        if error_logs:
            primary_cause = f"Multiple errors in {len(error_logs)} log entries"
        if high_severity_alerts:
            primary_cause = f"Critical alerts: {len(high_severity_alerts)}"
        
        return RootCauseAnalysis(
            analysis_id=f"rca_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            primary_cause=primary_cause,
            contributing_factors=["System overload", "Resource constraints"],
            confidence_score=0.3,
            evidence=[f"{len(error_logs)} error logs", f"{len(high_severity_alerts)} critical alerts"],
            recommendations=["Check system resources", "Review error logs", "Contact system administrator"],
            severity=AlertSeverity.HIGH if high_severity_alerts else AlertSeverity.MEDIUM,
            estimated_impact="Service degradation",
            resolution_steps=["Restart affected services", "Scale resources", "Investigate root cause"]
        )
    
    def get_analysis_history(self) -> List[RootCauseAnalysis]:
        """Get analysis history"""
        return self.analysis_history
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses"""
        if not self.analysis_history:
            return {"total_analyses": 0}
        
        return {
            "total_analyses": len(self.analysis_history),
            "avg_confidence": np.mean([rca.confidence_score for rca in self.analysis_history]),
            "severity_distribution": {
                severity.value: len([rca for rca in self.analysis_history if rca.severity == severity])
                for severity in AlertSeverity
            },
            "common_causes": self._get_common_causes(),
            "recent_analyses": [
                {
                    "analysis_id": rca.analysis_id,
                    "timestamp": rca.timestamp.isoformat(),
                    "primary_cause": rca.primary_cause,
                    "severity": rca.severity.value,
                    "confidence": rca.confidence_score
                }
                for rca in self.analysis_history[-10:]  # Last 10 analyses
            ]
        }
    
    def _get_common_causes(self) -> List[Tuple[str, int]]:
        """Get most common root causes"""
        cause_counts = {}
        for rca in self.analysis_history:
            cause = rca.primary_cause
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        
        return sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]

class RootCauseAnalysisManager:
    """Manager for root cause analysis operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.log_parser = LogParser(config)
        self.gpt_analyzer = GPTRootCauseAnalyzer(config)
        
        # Data storage
        self.logs = []
        self.alerts = []
        self.analyses = []
        
        # Processing threads
        self.processing_thread = None
        self.is_processing = False
    
    def ingest_logs(self, log_data: List[str]) -> List[LogEntry]:
        """Ingest and parse log data"""
        parsed_logs = []
        
        for log_line in log_data:
            log_entry = self.log_parser.parse_log_entry(log_line)
            if log_entry:
                parsed_logs.append(log_entry)
                self.logs.append(log_entry)
                
                # Detect alerts from log
                alerts = self.log_parser.detect_alerts(log_entry)
                self.alerts.extend(alerts)
        
        self.logger.info(f"Ingested {len(parsed_logs)} log entries, detected {len(self.alerts)} alerts")
        return parsed_logs
    
    def analyze_current_issues(self, context: Dict[str, Any] = None) -> RootCauseAnalysis:
        """Analyze current issues and provide RCA"""
        # Use recent logs and alerts
        recent_logs = self.logs[-100:] if len(self.logs) > 100 else self.logs
        recent_alerts = self.alerts[-20:] if len(self.alerts) > 20 else self.alerts
        
        # Perform analysis
        rca = self.gpt_analyzer.analyze_root_cause(recent_logs, recent_alerts, context)
        self.analyses.append(rca)
        
        return rca
    
    def start_continuous_analysis(self, interval_seconds: int = 300):
        """Start continuous RCA analysis"""
        if self.is_processing:
            self.logger.warning("Continuous analysis already running")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._continuous_analysis_loop,
            args=(interval_seconds,)
        )
        self.processing_thread.start()
        
        self.logger.info(f"Started continuous analysis with {interval_seconds}s interval")
    
    def stop_continuous_analysis(self):
        """Stop continuous RCA analysis"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
        
        self.logger.info("Stopped continuous analysis")
    
    def _continuous_analysis_loop(self, interval_seconds: int):
        """Continuous analysis loop"""
        while self.is_processing:
            try:
                # Analyze current issues
                rca = self.analyze_current_issues()
                
                # Log high-severity issues
                if rca.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                    self.logger.warning(f"High-severity issue detected: {rca.primary_cause}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Continuous analysis error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on RCA"""
        if not self.analyses:
            return {"status": "unknown", "message": "No analysis data available"}
        
        recent_analyses = self.analyses[-10:]  # Last 10 analyses
        critical_issues = [rca for rca in recent_analyses if rca.severity == AlertSeverity.CRITICAL]
        high_issues = [rca for rca in recent_analyses if rca.severity == AlertSeverity.HIGH]
        
        if critical_issues:
            return {
                "status": "critical",
                "message": f"{len(critical_issues)} critical issues detected",
                "recent_issues": [rca.primary_cause for rca in critical_issues[-3:]]
            }
        elif high_issues:
            return {
                "status": "warning",
                "message": f"{len(high_issues)} high-severity issues detected",
                "recent_issues": [rca.primary_cause for rca in high_issues[-3:]]
            }
        else:
            return {
                "status": "healthy",
                "message": "No critical issues detected",
                "recent_analyses": len(recent_analyses)
            }
    
    def get_analysis_dashboard_data(self) -> Dict[str, Any]:
        """Get data for RCA dashboard"""
        return {
            "total_logs": len(self.logs),
            "total_alerts": len(self.alerts),
            "total_analyses": len(self.analyses),
            "system_health": self.get_system_health(),
            "analysis_summary": self.gpt_analyzer.get_analysis_summary(),
            "recent_analyses": [
                {
                    "analysis_id": rca.analysis_id,
                    "timestamp": rca.timestamp.isoformat(),
                    "primary_cause": rca.primary_cause,
                    "severity": rca.severity.value,
                    "confidence": rca.confidence_score,
                    "recommendations": rca.recommendations[:3]  # Top 3 recommendations
                }
                for rca in self.analyses[-20:]  # Last 20 analyses
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test root cause analysis
    print("Testing GPT-powered Root Cause Analysis...")
    
    # Create RCA manager
    rca_manager = RootCauseAnalysisManager({
        'use_local_llm': True,  # Use local LLM for testing
        'local_model_name': 'microsoft/DialoGPT-medium'
    })
    
    # Sample log data
    sample_logs = [
        "2024-01-15 10:30:15 [ERROR] gNB-001: High latency detected: 150ms (threshold: 100ms)",
        "2024-01-15 10:30:20 [WARNING] gNB-001: CPU usage at 95%",
        "2024-01-15 10:30:25 [ERROR] gNB-001: Connection timeout to UE-12345",
        "2024-01-15 10:30:30 [CRITICAL] gNB-001: Memory exhausted, service restarting",
        "2024-01-15 10:30:35 [INFO] gNB-001: Service restarted successfully"
    ]
    
    # Ingest logs
    parsed_logs = rca_manager.ingest_logs(sample_logs)
    print(f"Parsed {len(parsed_logs)} log entries")
    
    # Analyze issues
    rca = rca_manager.analyze_current_issues()
    print(f"Root cause analysis: {rca.primary_cause}")
    print(f"Confidence: {rca.confidence_score}")
    print(f"Recommendations: {rca.recommendations}")
    
    # Get dashboard data
    dashboard_data = rca_manager.get_analysis_dashboard_data()
    print(f"Dashboard data: {dashboard_data}")
    
    print("Root cause analysis testing completed!")
