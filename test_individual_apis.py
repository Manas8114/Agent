#!/usr/bin/env python3
"""
Test individual API endpoints with longer timeouts
"""

import requests
import json
import time

def test_endpoint(url, name, timeout=10):
    """Test individual endpoint with longer timeout"""
    try:
        print(f"Testing {name}...")
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {name}: OK - {response.status_code} ({(end_time-start_time):.2f}s)")
            return True, data
        else:
            print(f"❌ {name}: Error - {response.status_code}")
            return False, None
    except requests.exceptions.Timeout:
        print(f"⏰ {name}: Timeout after {timeout}s")
        return False, None
    except requests.exceptions.ConnectionError:
        print(f"🔌 {name}: Connection Error")
        return False, None
    except Exception as e:
        print(f"❌ {name}: Error - {str(e)}")
        return False, None

def main():
    print("🔍 Individual API Endpoint Testing")
    print("=" * 50)
    
    endpoints = [
        ('http://localhost:8000/api/v1/health', 'Health Check'),
        ('http://localhost:8000/api/v1/telecom/kpis', 'KPIs'),
        ('http://localhost:8000/api/v1/telecom/federation', 'Federation'),
        ('http://localhost:8000/api/v1/telecom/self-evolution', 'Self-Evolution'),
        ('http://localhost:8000/api/v1/telecom/quantum-status', 'Quantum Security'),
        ('http://localhost:8000/api/v1/telecom/zta-status', 'ZTA Status')
    ]
    
    working_count = 0
    total_count = len(endpoints)
    
    for url, name in endpoints:
        success, data = test_endpoint(url, name, timeout=15)
        if success:
            working_count += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {working_count}/{total_count} endpoints working")
    
    if working_count == total_count:
        print("🎉 All API endpoints are working!")
    elif working_count > total_count // 2:
        print("⚠️  Most API endpoints are working")
    else:
        print("❌ Many API endpoints have issues")

if __name__ == "__main__":
    main()
