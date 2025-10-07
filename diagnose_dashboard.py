#!/usr/bin/env python3
"""
Diagnostic script to check dashboard components
"""

import requests
import json
import time
from urllib.parse import urljoin

def check_endpoint(base_url, endpoint, description):
    """Check if an endpoint is accessible"""
    try:
        url = urljoin(base_url, endpoint)
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {description}: OK")
            return True
        else:
            print(f"❌ {description}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {description}: {e}")
        return False

def check_frontend_content():
    """Check if frontend contains expected content"""
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            content = response.text
            
            # Check for key components
            checks = [
                ("Enhanced Telecom AI System", "Main title"),
                ("UserExperiencePanel", "User Experience Panel"),
                ("YouTubeDemoPanel", "YouTube Demo Panel"),
                ("QuantumSafePanel", "Quantum Security Panel"),
                ("framer-motion", "Animation library"),
                ("lucide-react", "Icon library")
            ]
            
            print("\n🔍 Frontend Content Analysis:")
            for search_term, description in checks:
                if search_term in content:
                    print(f"✅ {description}: Found")
                else:
                    print(f"❌ {description}: Not found")
            
            return True
        else:
            print(f"❌ Frontend: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend: {e}")
        return False

def check_api_endpoints():
    """Check API endpoints"""
    print("\n🔍 API Endpoints Check:")
    
    endpoints = [
        ("/api/v1/health", "Health Check"),
        ("/api/v1/telecom/kpis", "KPIs"),
        ("/api/v1/telecom/quantum-status", "Quantum Status"),
        ("/api/v1/quantum/health", "Quantum Health"),
        ("/api/v1/quantum/status", "Quantum Status")
    ]
    
    results = []
    for endpoint, description in endpoints:
        result = check_endpoint("http://localhost:8000", endpoint, description)
        results.append(result)
    
    return results

def check_ports():
    """Check if required ports are open"""
    print("\n🔍 Port Availability Check:")
    
    import socket
    
    ports = [
        (8000, "Backend API"),
        (3000, "Frontend Dashboard"),
        (9090, "Prometheus")
    ]
    
    results = []
    for port, description in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"✅ {description} (Port {port}): Open")
                results.append(True)
            else:
                print(f"❌ {description} (Port {port}): Closed")
                results.append(False)
        except Exception as e:
            print(f"❌ {description} (Port {port}): Error - {e}")
            results.append(False)
    
    return results

def main():
    """Run all diagnostics"""
    print("🔍 Telecom AI 4.0 Dashboard Diagnostics")
    print("=" * 50)
    
    # Check ports
    port_results = check_ports()
    
    # Check API endpoints
    api_results = check_api_endpoints()
    
    # Check frontend content
    frontend_ok = check_frontend_content()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Diagnostic Summary:")
    
    port_ok = all(port_results)
    api_ok = any(api_results)  # At least some APIs working
    frontend_ok = frontend_ok
    
    print(f"   Ports: {'✅ OK' if port_ok else '❌ Issues'}")
    print(f"   APIs: {'✅ OK' if api_ok else '❌ Issues'}")
    print(f"   Frontend: {'✅ OK' if frontend_ok else '❌ Issues'}")
    
    if port_ok and api_ok and frontend_ok:
        print("\n🎉 Dashboard should be working!")
        print("\n🌐 Access URLs:")
        print("   Main Dashboard: http://localhost:3000")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
    else:
        print("\n⚠️ Issues detected. Check the errors above.")
        
        if not port_ok:
            print("   - Start missing services")
        if not api_ok:
            print("   - Check backend server")
        if not frontend_ok:
            print("   - Check frontend compilation")

if __name__ == "__main__":
    main()




