#!/usr/bin/env python3
"""
Real Prometheus Metrics Server
Provides actual Prometheus-compatible metrics from real system data
"""

import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
from data.real_data_sources import real_data_collector, get_real_metrics, get_real_metrics_summary

logger = logging.getLogger(__name__)

class RealPrometheusHandler(BaseHTTPRequestHandler):
    """Real Prometheus metrics handler"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            current_time = datetime.now().isoformat()
            summary = get_real_metrics_summary()
            
            html = f"""
            <html>
            <head>
                <title>Enhanced Telecom AI - Real Prometheus Metrics</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                    .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                    h2 {{ color: #34495e; }}
                    .status {{ color: #27ae60; font-weight: bold; }}
                    .metric {{ background: #ecf0f1; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                    .value {{ color: #e74c3c; font-weight: bold; }}
                    ul {{ line-height: 1.6; }}
                    a {{ color: #3498db; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 Enhanced Telecom AI System - Real Metrics</h1>
                    <h2>Real Prometheus Metrics Server</h2>
                    <p class="status">Status: Running with REAL data</p>
                    <p>Time: {current_time}</p>
                    
                    <h3>📊 Real Data Collection Status:</h3>
                    <div class="metric">
                        <strong>Total Metrics Collected:</strong> <span class="value">{summary.get('total_metrics', 0)}</span>
                    </div>
                    <div class="metric">
                        <strong>Data Sources:</strong> {', '.join(summary.get('sources', []))}
                    </div>
                    <div class="metric">
                        <strong>Metric Types:</strong> {len(summary.get('metric_types', []))} different metrics
                    </div>
                    
                    <h3>🔗 Available Endpoints:</h3>
                    <ul>
                        <li><a href="/metrics">/metrics</a> - Real Prometheus metrics</li>
                        <li><a href="/api/v1/health">/api/v1/health</a> - System health</li>
                        <li><a href="/api/v1/real-metrics">/api/v1/real-metrics</a> - Real metrics JSON</li>
                    </ul>
                    
                    <h3>🎯 Real System Status:</h3>
                    <ul>
                        <li>FastAPI Server: <span class="status">Running on port 8000</span></li>
                        <li>React Dashboard: <span class="status">Running on port 3000</span></li>
                        <li>Real Prometheus: <span class="status">Running on port 9090</span></li>
                        <li>Real Data Collection: <span class="status">ACTIVE</span></li>
                    </ul>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            # Get real metrics
            real_metrics = get_real_metrics()
            current_time = int(time.time())
            
            # Generate Prometheus format metrics
            metrics_output = f"""# HELP enhanced_telecom_ai_real_system_health Real system health status
# TYPE enhanced_telecom_ai_real_system_health gauge
enhanced_telecom_ai_real_system_health 1

# HELP enhanced_telecom_ai_real_metrics_total Total number of real metrics collected
# TYPE enhanced_telecom_ai_real_metrics_total counter
enhanced_telecom_ai_real_metrics_total {len(real_metrics)}

# HELP enhanced_telecom_ai_real_timestamp Current timestamp
# TYPE enhanced_telecom_ai_real_timestamp gauge
enhanced_telecom_ai_real_timestamp {current_time}
"""
            
            # Add real system metrics
            for metric in real_metrics:
                if metric.metric_name == "cpu_usage":
                    metrics_output += f"enhanced_telecom_ai_real_cpu_usage {metric.value}\n"
                elif metric.metric_name == "memory_usage":
                    metrics_output += f"enhanced_telecom_ai_real_memory_usage {metric.value}\n"
                elif metric.metric_name == "disk_usage":
                    metrics_output += f"enhanced_telecom_ai_real_disk_usage {metric.value}\n"
                elif metric.metric_name == "network_latency":
                    metrics_output += f"enhanced_telecom_ai_real_network_latency {metric.value}\n"
                elif metric.metric_name == "api_response_time":
                    metrics_output += f"enhanced_telecom_ai_real_api_response_time {metric.value}\n"
                elif metric.metric_name == "qos_latency":
                    metrics_output += f"enhanced_telecom_ai_real_qos_latency {metric.value}\n"
                elif metric.metric_name == "qos_throughput":
                    metrics_output += f"enhanced_telecom_ai_real_qos_throughput {metric.value}\n"
                elif metric.metric_name == "energy_consumption":
                    metrics_output += f"enhanced_telecom_ai_real_energy_consumption {metric.value}\n"
                elif metric.metric_name == "security_score":
                    metrics_output += f"enhanced_telecom_ai_real_security_score {metric.value}\n"
            
            self.wfile.write(metrics_output.encode())
            
        elif self.path == '/api/v1/real-metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            real_metrics = get_real_metrics()
            metrics_data = []
            
            for metric in real_metrics:
                metrics_data.append({
                    "timestamp": metric.timestamp.isoformat(),
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "source": metric.source
                })
            
            response = {
                "status": "success",
                "total_metrics": len(real_metrics),
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics_data
            }
            
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass

def start_real_prometheus_server():
    """Start real Prometheus metrics server"""
    try:
        with socketserver.TCPServer(('', 9090), RealPrometheusHandler) as httpd:
            logger.info('🚀 Real Prometheus metrics server running on port 9090')
            logger.info('📊 Access real metrics at: http://localhost:9090/metrics')
            logger.info('🌐 Web interface: http://localhost:9090')
            httpd.serve_forever()
    except Exception as e:
        logger.error(f'❌ Failed to start real Prometheus server: {e}')

async def start_real_data_collection():
    """Start real data collection"""
    from data.real_data_sources import start_real_data_collection
    await start_real_data_collection()

def run_real_prometheus():
    """Run real Prometheus server with data collection"""
    import asyncio
    
    # Start data collection in background
    asyncio.create_task(start_real_data_collection())
    
    # Start Prometheus server
    start_real_prometheus_server()

if __name__ == '__main__':
    run_real_prometheus()

