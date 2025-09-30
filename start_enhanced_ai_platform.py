#!/usr/bin/env python3
"""
Enhanced Telecom System Startup Script with Safety & Governance
Starts the complete AI-native network management platform
"""

import subprocess
import time
import webbrowser
import os
import sys
import socket
import asyncio
import threading
from pathlib import Path

def find_free_port(start_port):
    """Finds a free port starting from start_port."""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1

def check_redis_connection():
    """Check if Redis is running"""
    try:
        import redis
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please start Redis server: redis-server")
        return False

def kill_existing_processes():
    """Kill existing Python processes"""
    print("🧹 Cleaning up existing processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                      capture_output=True, text=True)
        print("✅ Existing Python processes terminated")
    except Exception as e:
        print(f"ℹ️  No existing processes to kill: {e}")

def start_redis_server():
    """Start Redis server if not running"""
    try:
        import redis
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis server already running")
        return True
    except:
        print("🚀 Starting Redis server...")
        try:
            # Try to start Redis server
            subprocess.Popen(["redis-server"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(3)
            
            # Check if it started
            import redis
            r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
            r.ping()
            print("✅ Redis server started successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to start Redis server: {e}")
            print("Please install and start Redis manually")
            return False

def start_control_api(api_port):
    """Start the Control API server"""
    print(f"🔧 Starting Control API on port {api_port}...")
    try:
        env = os.environ.copy()
        env["CONTROL_API_PORT"] = str(api_port)
        env["CONTROL_API_KEY"] = "secret"
        
        process = subprocess.Popen([
            sys.executable, "control_api.py"
        ], env=env)
        
        print(f"✅ Control API started on port {api_port} (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start Control API: {e}")
        return None

def start_coordinator():
    """Start the Safety Governance Coordinator"""
    print("🛡️  Starting Safety Governance Coordinator...")
    try:
        env = os.environ.copy()
        env["REDIS_URL"] = "redis://127.0.0.1:6379/0"
        env["POLICY_FILE"] = "policy.yaml"
        
        process = subprocess.Popen([
            sys.executable, "coordinator.py"
        ], env=env)
        
        print(f"✅ Coordinator started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start Coordinator: {e}")
        return None

def start_executor():
    """Start the Safe Action Executor"""
    print("⚡ Starting Safe Action Executor...")
    try:
        env = os.environ.copy()
        env["REDIS_URL"] = "redis://127.0.0.1:6379/0"
        env["CONTROL_API_URL"] = "http://127.0.0.1:5001"
        env["CONTROL_API_KEY"] = "secret"
        
        process = subprocess.Popen([
            sys.executable, "executor.py"
        ], env=env)
        
        print(f"✅ Executor started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start Executor: {e}")
        return None

def start_enhanced_system(api_port):
    """Start the enhanced telecom production system"""
    print("🚀 Starting Enhanced Telecom Production System...")
    try:
        script_path = os.path.join(os.path.dirname(__file__), "enhanced_telecom_system.py")
        env = os.environ.copy()
        env["API_PORT"] = str(api_port)
        env["REDIS_URL"] = "redis://127.0.0.1:6379/0"
        
        process = subprocess.Popen([sys.executable, script_path], env=env)
        print(f"✅ Enhanced Telecom System started on port {api_port} (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start Enhanced Telecom System: {e}")
        return None

def start_dashboard_server(dashboard_port):
    """Start a simple HTTP server to serve the enhanced dashboard"""
    print(f"🌐 Starting Enhanced Dashboard Server on port {dashboard_port}...")
    try:
        script_dir = os.path.dirname(__file__)
        command = [sys.executable, "-m", "http.server", str(dashboard_port)]
        process = subprocess.Popen(command, cwd=script_dir)
        print(f"✅ Enhanced Dashboard server started at http://localhost:{dashboard_port}")
        return process
    except Exception as e:
        print(f"❌ Failed to start dashboard server: {e}")
        return None

def test_system_integration():
    """Test the system integration"""
    print("🧪 Testing system integration...")
    
    try:
        import redis
        import requests
        
        # Test Redis
        r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection test passed")
        
        # Test Control API
        response = requests.get("http://127.0.0.1:5001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Control API test passed")
        else:
            print("❌ Control API test failed")
        
        # Test Enhanced Telecom System
        response = requests.get("http://127.0.0.1:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ Enhanced Telecom System test passed")
        else:
            print("❌ Enhanced Telecom System test failed")
        
        print("✅ System integration tests completed")
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")

def create_systemd_services():
    """Create systemd service files for production deployment"""
    services = {
        'ai-coordinator.service': '''[Unit]
Description=AI Safety Governance Coordinator
After=network.target redis.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ai_agents
ExecStart=/usr/bin/python3 coordinator.py
Restart=always
RestartSec=10
Environment=REDIS_URL=redis://127.0.0.1:6379/0
Environment=POLICY_FILE=/opt/ai_agents/policy.yaml

[Install]
WantedBy=multi-user.target''',
        
        'ai-executor.service': '''[Unit]
Description=AI Safe Action Executor
After=network.target redis.service ai-coordinator.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ai_agents
ExecStart=/usr/bin/python3 executor.py
Restart=always
RestartSec=10
Environment=REDIS_URL=redis://127.0.0.1:6379/0
Environment=CONTROL_API_URL=http://127.0.0.1:5001
Environment=CONTROL_API_KEY=secret

[Install]
WantedBy=multi-user.target''',
        
        'control-api.service': '''[Unit]
Description=Control API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ai_agents
ExecStart=/usr/bin/python3 control_api.py
Restart=always
RestartSec=10
Environment=CONTROL_API_PORT=5001
Environment=CONTROL_API_KEY=secret

[Install]
WantedBy=multi-user.target'''
    }
    
    print("📋 Creating systemd service files...")
    for service_name, service_content in services.items():
        try:
            with open(service_name, 'w') as f:
                f.write(service_content)
            print(f"✅ Created {service_name}")
        except Exception as e:
            print(f"❌ Failed to create {service_name}: {e}")

def main():
    print("🚀 Starting Enhanced Telecom AI-Native Network Management Platform")
    print("=" * 80)
    
    # Kill existing processes
    kill_existing_processes()
    time.sleep(2)
    
    # Check/Start Redis
    if not start_redis_server():
        print("❌ Cannot proceed without Redis. Please install and start Redis.")
        return
    
    # Find free ports
    api_port = find_free_port(8080)
    dashboard_port = find_free_port(3000)
    control_api_port = find_free_port(5001)
    
    processes = []
    
    # Start Control API
    control_api_process = start_control_api(control_api_port)
    if control_api_process:
        processes.append(('Control API', control_api_process))
    
    # Give Control API time to start
    time.sleep(3)
    
    # Start Coordinator
    coordinator_process = start_coordinator()
    if coordinator_process:
        processes.append(('Coordinator', coordinator_process))
    
    # Start Executor
    executor_process = start_executor()
    if executor_process:
        processes.append(('Executor', executor_process))
    
    # Give safety components time to start
    time.sleep(5)
    
    # Start Enhanced Telecom System
    telecom_process = start_enhanced_system(api_port)
    if telecom_process:
        processes.append(('Enhanced Telecom System', telecom_process))
    
    # Give backend time to start
    time.sleep(8)
    
    # Start Dashboard
    dashboard_process = start_dashboard_server(dashboard_port)
    if dashboard_process:
        processes.append(('Dashboard Server', dashboard_process))
    
    # Test system integration
    time.sleep(5)
    test_system_integration()
    
    # Create systemd services for production
    create_systemd_services()
    
    dashboard_url = f"http://localhost:{dashboard_port}/enhanced_dashboard.html"
    
    print("=" * 80)
    print("🎯 Enhanced AI-Native Network Management Platform Started!")
    print("=" * 80)
    print(f"📡 Enhanced API Server: http://localhost:{api_port}")
    print(f"🔧 Control API: http://localhost:{control_api_port}")
    print(f"🔍 Health Check: http://localhost:{api_port}/health")
    print(f"📊 System Status: http://localhost:{api_port}/status")
    print(f"📈 Telecom Metrics: http://localhost:{api_port}/telecom/metrics")
    print(f"🚨 QoS Alerts: http://localhost:{api_port}/telecom/alerts")
    print(f"🔮 Failure Predictions: http://localhost:{api_port}/telecom/predictions")
    print(f"📊 Traffic Forecasts: http://localhost:{api_port}/telecom/forecasts")
    print(f"⚡ Energy Optimization: http://localhost:{api_port}/telecom/energy")
    print(f"🔒 Security Events: http://localhost:{api_port}/telecom/security")
    print(f"📋 Data Quality: http://localhost:{api_port}/telecom/quality")
    print("=" * 80)
    print("🤖 Advanced AI Agents with Safety & Governance:")
    print("   ✅ Enhanced QoS Anomaly Detection")
    print("      • Root-cause analysis (congestion, poor RF, QoS misconfig)")
    print("      • Dynamic thresholds per cell/time of day")
    print("      • User impact scoring & QoE degradation")
    print("      • Self-healing recommendations")
    print()
    print("   ✅ Advanced Failure Prediction Agent")
    print("      • Predictive alarms for gNB/AMF failures")
    print("      • Explainable AI with feature importance")
    print("      • Scenario simulation ('what if' analysis)")
    print("      • Automated ticket creation")
    print()
    print("   ✅ Traffic Forecast Agent")
    print("      • Multi-timescale forecasting (5min, 1hr, daily)")
    print("      • Event-aware predictions")
    print("      • Capacity planning recommendations")
    print("      • Network slicing demand forecasting")
    print()
    print("   ✅ Energy Optimization Agent")
    print("      • Dynamic sleep modes & micro-sleep")
    print("      • Green score & CO₂ savings calculation")
    print("      • Adaptive thresholds learning")
    print("      • Cross-agent integration")
    print()
    print("   ✅ Security & Intrusion Detection Agent")
    print("      • Fake UE detection & SIM cloning")
    print("      • DoS attempts & signaling storms")
    print("      • Behavior analysis with DBSCAN")
    print("      • Multi-level threat analysis")
    print()
    print("   ✅ Data Quality Monitoring Agent")
    print("      • Completeness, accuracy & consistency checks")
    print("      • Automated validation pipeline")
    print("      • Quality recommendations")
    print("=" * 80)
    print("🛡️  Safety & Governance Framework:")
    print("   ✅ Auto-mode toggles (global & per-action)")
    print("   ✅ Confidence thresholds & policy enforcement")
    print("   ✅ Rate limiting & circuit breakers")
    print("   ✅ Canary deployments & validation")
    print("   ✅ Automatic rollback & audit trails")
    print("   ✅ Operator override & emergency stop")
    print("=" * 80)
    print("🔄 Redis Message Bus Channels:")
    print("   • anomalies.alerts - QoS & security alerts")
    print("   • optimization.commands - Energy & traffic optimization")
    print("   • actions.approved - Coordinator approvals")
    print("   • actions.executed - Executor results")
    print("   • actions.feedback - Success/failure feedback")
    print("   • operator.commands - Manual overrides")
    print("=" * 80)
    print(f"💡 Enhanced Dashboard URL: {dashboard_url}")
    print("🔄 Auto-refresh every 5 seconds with advanced visualizations")
    print("⏹️  Press Ctrl+C to stop the enhanced system")
    print("=" * 80)
    
    # Open the enhanced dashboard
    webbrowser.open(dashboard_url)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enhanced AI-native platform...")
        
        for name, process in processes:
            if process:
                process.terminate()
                process.wait()
                print(f"✅ {name} stopped.")
        
        print("✅ Enhanced AI-native platform shut down gracefully.")
        print("🎯 Thank you for using the Enhanced Telecom AI-Native Network Management Platform!")

if __name__ == "__main__":
    main()
