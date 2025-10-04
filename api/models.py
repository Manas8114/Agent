#!/usr/bin/env python3
"""
Pydantic models for Enhanced Telecom AI System API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    agents_status: Dict[str, str]
    system_metrics: Dict[str, float]
    components: Dict[str, str]

class KPIsResponse(BaseModel):
    latency_ms: float
    throughput_mbps: float
    jitter_ms: float
    packet_loss_rate: float
    connection_quality: float
    signal_strength: float
    user_count: int
    data_volume_gb: float
    error_count: int
    warning_count: int

class CoordinationResponse(BaseModel):
    coordination_score: float
    active_agents: int
    total_agents: int
    last_coordination: str
    coordination_metrics: Dict[str, float]

class OptimizationResponse(BaseModel):
    energy_savings_percent: float
    qos_improvement_percent: float
    resource_utilization: float
    optimization_score: float
    recommendations: List[str]
    last_optimization: str

class EventRequest(BaseModel):
    event_type: str
    data: Dict[str, Any]

class EventResponse(BaseModel):
    event_id: str
    event_type: str
    timestamp: str
    data: Dict[str, Any]
    status: str

class ExplainRequest(BaseModel):
    agent_type: str
    instance: List[float]
    context: Optional[Dict[str, Any]] = None

class ExplainResponse(BaseModel):
    agent_type: str
    instance: List[float]
    prediction: float
    explanation: str
    feature_importance: Dict[str, Any]
    recommendations: List[str]
    timestamp: str

# AI 4.0 Models
class IntentRequest(BaseModel):
    description: str
    intent_type: str
    constraints: Dict[str, Any]
    priority: int

class IntentResponse(BaseModel):
    intent_id: str
    description: str
    status: str
    enforcement_logs: List[str]

class ZTAStatusResponse(BaseModel):
    pipeline_id: str
    name: str
    status: str
    updates_count: int
    execution_logs: List[str]
    active_pipelines: List[Dict[str, Any]]
    deployment_metrics: Dict[str, Any]

class QuantumStatusResponse(BaseModel):
    pqc_encryptions_total: int
    pqc_decryptions_total: int
    pqc_signatures_total: int
    pqc_verifications_total: int
    pqc_encryption_success_rate: float
    pqc_verification_success_rate: float
    security_level: str
    algorithms: List[str]
    threat_detection: Dict[str, Any]

class FederationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    total_nodes: int
    active_nodes: int
    model_updates_shared: int = Field(alias="updates_shared")
    model_aggregations_total: int = Field(alias="aggregations_total")
    avg_model_accuracy: float
    cooperative_scenarios_handled: int

class SelfEvolutionResponse(BaseModel):
    agent_id: str
    evolution_round: int
    architecture_improvement: float
    hyperparameter_optimization: Dict[str, Any]
    performance_improvement: float
    evolution_status: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str