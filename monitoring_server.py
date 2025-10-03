#!/usr/bin/env python3
"""
Simple Monitoring Server for Enhanced Telecom AI System
Provides basic metrics and monitoring without Docker dependencies
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Telecom AI Monitoring",
    description="Simple monitoring server for the Enhanced Telecom AI System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock metrics data
def get_system_metrics() -> Dict[str, Any]:
    """Generate mock system metrics."""
    import random
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_usage": round(random.uniform(20, 80), 2),
            "memory_usage": round(random.uniform(30, 90), 2),
            "disk_usage": round(random.uniform(10, 70), 2),
            "uptime": "2h 15m 30s"
        },
        "ai_agents": {
            "qos_anomaly": {
                "status": "healthy",
                "accuracy": round(random.uniform(85, 98), 2),
                "predictions_today": random.randint(100, 500)
            },
            "failure_prediction": {
                "status": "healthy", 
                "accuracy": round(random.uniform(88, 95), 2),
                "predictions_today": random.randint(50, 200)
            },
            "traffic_forecast": {
                "status": "healthy",
                "accuracy": round(random.uniform(82, 92), 2),
                "predictions_today": random.randint(200, 800)
            },
            "energy_optimize": {
                "status": "healthy",
                "accuracy": round(random.uniform(90, 97), 2),
                "predictions_today": random.randint(80, 300)
            },
            "security_detection": {
                "status": "healthy",
                "accuracy": round(random.uniform(85, 96), 2),
                "predictions_today": random.randint(150, 600)
            },
            "data_quality": {
                "status": "healthy",
                "accuracy": round(random.uniform(88, 95), 2),
                "predictions_today": random.randint(100, 400)
            }
        },
        "business_metrics": {
            "energy_savings_kwh": round(random.uniform(1000, 5000), 2),
            "cost_reduction_percent": round(random.uniform(15, 35), 2),
            "service_quality_score": round(random.uniform(85, 98), 2),
            "anomalies_detected": random.randint(5, 25),
            "failures_prevented": random.randint(2, 8)
        }
    }

@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Enhanced Telecom AI Monitoring",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "metrics": "/metrics",
            "health": "/health",
            "dashboard": "/dashboard"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def metrics():
    """Get system metrics."""
    return get_system_metrics()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple HTML dashboard."""
    metrics_data = get_system_metrics()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Telecom AI Monitoring</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
            .agent-status {{ display: flex; justify-content: space-between; align-items: center; margin: 10px 0; }}
            .status-healthy {{ color: #27ae60; }}
            .status-warning {{ color: #f39c12; }}
            .status-error {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Enhanced Telecom AI Monitoring Dashboard</h1>
                <p>Real-time monitoring of AI agents and system performance</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">System Performance</div>
                    <div>CPU Usage: <span class="metric-value">{metrics_data['system']['cpu_usage']}%</span></div>
                    <div>Memory Usage: <span class="metric-value">{metrics_data['system']['memory_usage']}%</span></div>
                    <div>Disk Usage: <span class="metric-value">{metrics_data['system']['disk_usage']}%</span></div>
                    <div>Uptime: <span class="metric-value">{metrics_data['system']['uptime']}</span></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">AI Agents Status</div>
                    {''.join([f'''
                    <div class="agent-status">
                        <span>{agent.replace('_', ' ').title()}</span>
                        <span class="status-healthy">✓ {data['status']} ({data['accuracy']}%)</span>
                    </div>
                    ''' for agent, data in metrics_data['ai_agents'].items()])}
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Business Impact</div>
                    <div>Energy Savings: <span class="metric-value">{metrics_data['business_metrics']['energy_savings_kwh']} kWh</span></div>
                    <div>Cost Reduction: <span class="metric-value">{metrics_data['business_metrics']['cost_reduction_percent']}%</span></div>
                    <div>Service Quality: <span class="metric-value">{metrics_data['business_metrics']['service_quality_score']}%</span></div>
                    <div>Anomalies Detected: <span class="metric-value">{metrics_data['business_metrics']['anomalies_detected']}</span></div>
                    <div>Failures Prevented: <span class="metric-value">{metrics_data['business_metrics']['failures_prevented']}</span></div>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
                <p>Enhanced Telecom AI System - Monitoring Dashboard</p>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
        
        <script>
            // Auto-refresh every 30 seconds
            setTimeout(() => location.reload(), 30000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Telecom AI Monitoring Server...")
    print("📍 Monitoring Dashboard: http://localhost:9090")
    print("📊 Metrics API: http://localhost:9090/metrics")
    print("🔍 Health Check: http://localhost:9090/health")
    print("\n" + "="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        log_level="info"
    )
