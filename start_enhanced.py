#!/usr/bin/env python3
"""
🚀 Enhanced Telecom Production System Startup Script
Starts the enhanced system with 6 advanced AI agents
"""

import subprocess
import time
import webbrowser
import os
import sys
import socket

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

def kill_existing_processes():
    """Kill existing Python processes"""
    print("🧹 Cleaning up existing processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                      capture_output=True, text=True)
        print("✅ Existing Python processes terminated")
    except Exception as e:
        print(f"ℹ️  No existing processes to kill: {e}")

def start_enhanced_system(api_port):
    """Starts the enhanced telecom production system"""
    print("🚀 Starting Enhanced Telecom Production System...")
    try:
        script_path = os.path.join(os.path.dirname(__file__), "enhanced_telecom_system.py")
        env = os.environ.copy()
        env["API_PORT"] = str(api_port)
        process = subprocess.Popen([sys.executable, script_path], env=env)
        print(f"✅ Enhanced Telecom System started on port {api_port} (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start Enhanced Telecom System: {e}")
        return None

def start_dashboard_server(dashboard_port):
    """Starts a simple HTTP server to serve the enhanced dashboard"""
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

def main():
    print("🚀 Starting Enhanced Telecom Production System with 6 AI Agents")
    print("=" * 80)
    
    # Kill existing processes
    kill_existing_processes()
    time.sleep(2)
    
    # Find free ports
    api_port = find_free_port(8080)
    dashboard_port = find_free_port(3000)
    
    # Start the enhanced backend system
    telecom_process = start_enhanced_system(api_port)
    if not telecom_process:
        print("❌ Aborting startup due to backend failure.")
        return
    
    # Give the backend time to start
    time.sleep(8)
    
    # Start the enhanced dashboard server
    dashboard_process = start_dashboard_server(dashboard_port)
    if not dashboard_process:
        telecom_process.terminate()
        print("❌ Aborting startup due to dashboard server failure.")
        return
    
    dashboard_url = f"http://localhost:{dashboard_port}/enhanced_dashboard.html"
    print(f"📡 Enhanced API Server: http://localhost:{api_port}")
    print(f"🔍 Health Check: http://localhost:{api_port}/health")
    print(f"📊 System Status: http://localhost:{api_port}/status")
    print(f"📈 Telecom Metrics: http://localhost:{api_port}/telecom/metrics")
    print(f"🚨 QoS Alerts: http://localhost:{api_port}/telecom/alerts")
    print(f"🔮 Failure Predictions: http://localhost:{api_port}/telecom/predictions")
    print(f"📊 Traffic Forecasts: http://localhost:{api_port}/telecom/forecasts")
    print(f"⚡ Energy Optimization: http://localhost:{api_port}/telecom/energy")
    print(f"🔒 Security Events: http://localhost:{api_port}/telecom/security")
    print(f"📋 Data Quality: http://localhost:{api_port}/telecom/quality")
    print(f"📱 Telecom Events: http://localhost:{api_port}/telecom/events")
    print("=" * 80)
    print("🤖 Advanced AI Agents:")
    print("   ✅ Enhanced QoS Anomaly Detection")
    print("      • Isolation Forest + LSTM Autoencoder")
    print("      • Adaptive thresholds & feature importance")
    print("      • Model explainability with SHAP-like analysis")
    print()
    print("   ✅ Advanced Failure Prediction Agent")
    print("      • Random Forest with adaptive learning")
    print("      • UE session tracking & pattern analysis")
    print("      • Feedback loop for continuous improvement")
    print()
    print("   ✅ Traffic Forecast Agent")
    print("      • Time series analysis with Prophet/LSTM")
    print("      • Capacity planning & overload warnings")
    print("      • 5-15 minute forecasting windows")
    print()
    print("   ✅ Energy Optimization Agent")
    print("      • Intelligent gNB management")
    print("      • Sleep mode & power reduction recommendations")
    print("      • Energy savings calculations")
    print()
    print("   ✅ Security & Intrusion Detection Agent")
    print("      • Behavior analysis with DBSCAN clustering")
    print("      • Brute force & SIM cloning detection")
    print("      • Suspicious mobility pattern analysis")
    print()
    print("   ✅ Data Quality Monitoring Agent")
    print("      • Completeness, accuracy & consistency checks")
    print("      • Automated data validation")
    print("      • Quality recommendations")
    print("=" * 80)
    print("🎯 Enhanced Dashboard Features:")
    print("   ✅ Real-time AI performance monitoring")
    print("   ✅ Model explainability & confidence scores")
    print("   ✅ Interactive charts for all 6 agents")
    print("   ✅ Feature importance visualization")
    print("   ✅ Adaptive threshold monitoring")
    print("   ✅ Security threat heatmaps")
    print("   ✅ Energy optimization recommendations")
    print("   ✅ Traffic forecasting with capacity alerts")
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
        print("\nShutting down enhanced system...")
        if telecom_process:
            telecom_process.terminate()
            telecom_process.wait()
            print("Enhanced Telecom Production System stopped.")
        if dashboard_process:
            dashboard_process.terminate()
            dashboard_process.wait()
            print("Enhanced Dashboard server stopped.")
        print("Enhanced system shut down gracefully.")

if __name__ == "__main__":
    main()
