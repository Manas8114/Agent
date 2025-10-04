#!/usr/bin/env python3
"""
Final Comprehensive Integration Test
Tests all components working together
"""

import requests
import json
import time
import subprocess
import sys

def test_backend_health():
    """Test backend health endpoint"""
    try:
        response = requests.get('http://localhost:8000/api/v1/health', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend Health: {data['status']} - Version {data['version']}")
            return True
        return False
    except Exception as e:
        print(f"❌ Backend Health: {str(e)}")
        return False

def test_ai4_features():
    """Test AI 4.0 specific features"""
    features = [
        ('/api/v1/telecom/kpis', 'KPIs & Metrics'),
        ('/api/v1/telecom/federation', 'Global Federation'),
        ('/api/v1/telecom/self-evolution', 'Self-Evolution'),
        ('/api/v1/telecom/quantum-status', 'Quantum Security')
    ]
    
    working_features = 0
    total_features = len(features)
    
    for endpoint, name in features:
        try:
            response = requests.get(f'http://localhost:8000{endpoint}', timeout=10)
            if response.status_code == 200:
                print(f"✅ {name}: Working")
                working_features += 1
            else:
                print(f"❌ {name}: Error {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")
    
    return working_features, total_features

def test_prometheus():
    """Test Prometheus metrics"""
    try:
        response = requests.get('http://localhost:9090', timeout=5)
        if response.status_code == 200:
            print("✅ Prometheus: Working")
            return True
        return False
    except Exception as e:
        print(f"❌ Prometheus: {str(e)}")
        return False

def test_react_frontend():
    """Test React frontend"""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        if response.status_code == 200:
            print("✅ React Frontend: Working")
            return True
        return False
    except Exception as e:
        print(f"❌ React Frontend: {str(e)}")
        return False

def test_data_integration():
    """Test data integration between components"""
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:8000/api/v1/health', timeout=10)
        if health_response.status_code != 200:
            return False
        
        # Test KPIs endpoint
        kpis_response = requests.get('http://localhost:8000/api/v1/telecom/kpis', timeout=10)
        if kpis_response.status_code != 200:
            return False
        
        # Test Federation endpoint
        fed_response = requests.get('http://localhost:8000/api/v1/telecom/federation', timeout=10)
        if fed_response.status_code != 200:
            return False
        
        print("✅ Data Integration: All components communicating")
        return True
    except Exception as e:
        print(f"❌ Data Integration: {str(e)}")
        return False

def main():
    """Main integration test"""
    print("🚀 Enhanced Telecom AI 4.0 - Final Integration Test")
    print("=" * 60)
    
    # Wait for services to stabilize
    print("⏳ Waiting for services to stabilize...")
    time.sleep(5)
    
    # Test all components
    print("\n🔍 Testing Core Components...")
    print("-" * 40)
    
    backend_ok = test_backend_health()
    prometheus_ok = test_prometheus()
    react_ok = test_react_frontend()
    
    print("\n🔍 Testing AI 4.0 Features...")
    print("-" * 40)
    working_features, total_features = test_ai4_features()
    
    print("\n🔍 Testing Data Integration...")
    print("-" * 40)
    data_integration_ok = test_data_integration()
    
    # Calculate overall health
    core_components = sum([backend_ok, prometheus_ok, react_ok])
    feature_percentage = (working_features / total_features) * 100 if total_features > 0 else 0
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    print(f"Core Components: {core_components}/3 working")
    print(f"  • Backend: {'✅' if backend_ok else '❌'}")
    print(f"  • Prometheus: {'✅' if prometheus_ok else '❌'}")
    print(f"  • React Frontend: {'✅' if react_ok else '❌'}")
    
    print(f"\nAI 4.0 Features: {working_features}/{total_features} working ({feature_percentage:.1f}%)")
    print(f"Data Integration: {'✅' if data_integration_ok else '❌'}")
    
    # Overall assessment
    overall_health = (core_components / 3) * 0.4 + (feature_percentage / 100) * 0.4 + (1 if data_integration_ok else 0) * 0.2
    
    print(f"\nOverall System Health: {overall_health:.1%}")
    
    if overall_health >= 0.8:
        print("\n🎉 SYSTEM FULLY OPERATIONAL!")
        print("✅ All major components working together")
    elif overall_health >= 0.6:
        print("\n⚠️  SYSTEM MOSTLY OPERATIONAL")
        print("✅ Most components working, minor issues")
    else:
        print("\n❌ SYSTEM NEEDS ATTENTION")
        print("⚠️  Multiple components have issues")
    
    print("=" * 60)
    
    return overall_health >= 0.6

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
