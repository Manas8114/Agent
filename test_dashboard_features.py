#!/usr/bin/env python3
"""
Test script to verify all dashboard features are working
"""

import requests
import json
import time

def test_backend_health():
    """Test backend health endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend Health: OK")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Agents: {len(data.get('agents_status', {}))}")
            return True
        else:
            print(f"❌ Backend Health: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend Health: {e}")
        return False

def test_quantum_endpoints():
    """Test quantum security endpoints"""
    try:
        # Test quantum health endpoint
        response = requests.get("http://localhost:8000/api/v1/quantum/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Quantum Security: OK")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Quantum Safe: {data.get('quantum_safe', False)}")
            return True
        else:
            print(f"❌ Quantum Security: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Quantum Security: {e}")
        return False

def test_frontend_access():
    """Test frontend accessibility"""
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "Enhanced Telecom AI System" in content:
                print("✅ Frontend Dashboard: OK")
                print("   Title: Enhanced Telecom AI System")
                return True
            else:
                print("❌ Frontend Dashboard: Missing expected content")
                return False
        else:
            print(f"❌ Frontend Dashboard: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend Dashboard: {e}")
        return False

def test_telecom_endpoints():
    """Test telecom-specific endpoints"""
    endpoints = [
        "/api/v1/telecom/kpis",
        "/api/v1/telecom/quantum-status", 
        "/api/v1/telecom/zta-status",
        "/api/v1/telecom/federation",
        "/api/v1/telecom/self-evolution"
    ]
    
    success_count = 0
    for endpoint in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            if response.status_code == 200:
                success_count += 1
                print(f"✅ {endpoint}: OK")
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
    
    print(f"   Telecom Endpoints: {success_count}/{len(endpoints)} working")
    return success_count == len(endpoints)

def main():
    """Run all tests"""
    print("🚀 Testing Telecom AI 4.0 Dashboard Features")
    print("=" * 50)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Frontend Access", test_frontend_access),
        ("Quantum Security", test_quantum_endpoints),
        ("Telecom Endpoints", test_telecom_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All features are working! Dashboard is ready.")
        print("\n🌐 Access URLs:")
        print("   Main Dashboard: http://localhost:3000")
        print("   Backend API: http://localhost:8000")
        print("   Quantum Security: http://localhost:8000/api/v1/quantum")
        print("   Prometheus: http://localhost:9090")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()




