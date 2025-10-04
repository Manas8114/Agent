#!/usr/bin/env python3
"""
Enhanced Telecom AI 4.0 - Integration Test
Tests all components working together
"""

import requests
import json
import sys
import time

def test_backend_apis():
    """Test all backend API endpoints"""
    print("🔍 Testing Backend API Endpoints...")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        ('/api/v1/health', 'Health Check'),
        ('/api/v1/telecom/kpis', 'KPIs'),
        ('/api/v1/telecom/federation', 'Federation'),
        ('/api/v1/telecom/self-evolution', 'Self-Evolution'),
        ('/api/v1/telecom/quantum-status', 'Quantum Security'),
        ('/api/v1/telecom/zta-status', 'ZTA Status')
    ]
    
    base_url = 'http://localhost:8000'
    all_working = True
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f'{base_url}{endpoint}', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f'✅ {name}: OK - Status {response.status_code}')
                if 'status' in data:
                    print(f'   Status: {data["status"]}')
            else:
                print(f'❌ {name}: Error - Status {response.status_code}')
                all_working = False
        except requests.exceptions.ConnectionError:
            print(f'❌ {name}: Connection Error - Backend not running')
            all_working = False
        except Exception as e:
            print(f'❌ {name}: Error - {str(e)}')
            all_working = False
    
    return all_working

def test_prometheus():
    """Test Prometheus metrics server"""
    print("\n🔍 Testing Prometheus Server...")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:9090', timeout=5)
        if response.status_code == 200:
            print('✅ Prometheus Server: OK - Status 200')
            return True
        else:
            print(f'❌ Prometheus Server: Error - Status {response.status_code}')
            return False
    except requests.exceptions.ConnectionError:
        print('❌ Prometheus Server: Connection Error - Not running')
        return False
    except Exception as e:
        print(f'❌ Prometheus Server: Error - {str(e)}')
        return False

def test_frontend():
    """Test React frontend"""
    print("\n🔍 Testing React Frontend...")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        if response.status_code == 200:
            print('✅ React Frontend: OK - Status 200')
            return True
        else:
            print(f'❌ React Frontend: Error - Status {response.status_code}')
            return False
    except requests.exceptions.ConnectionError:
        print('❌ React Frontend: Connection Error - Not running')
        return False
    except Exception as e:
        print(f'❌ React Frontend: Error - {str(e)}')
        return False

def test_data_flow():
    """Test data flow between components"""
    print("\n🔍 Testing Data Flow...")
    print("=" * 50)
    
    try:
        # Test health endpoint for system metrics
        health_response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print('✅ Health Data Flow: OK')
            
            # Test KPIs endpoint
            kpis_response = requests.get('http://localhost:8000/api/v1/telecom/kpis', timeout=5)
            if kpis_response.status_code == 200:
                kpis_data = kpis_response.json()
                print('✅ KPIs Data Flow: OK')
                
                # Test Federation endpoint
                fed_response = requests.get('http://localhost:8000/api/v1/telecom/federation', timeout=5)
                if fed_response.status_code == 200:
                    fed_data = fed_response.json()
                    print('✅ Federation Data Flow: OK')
                    
                    # Test Self-Evolution endpoint
                    evo_response = requests.get('http://localhost:8000/api/v1/telecom/self-evolution', timeout=5)
                    if evo_response.status_code == 200:
                        evo_data = evo_response.json()
                        print('✅ Self-Evolution Data Flow: OK')
                        return True
                    else:
                        print('❌ Self-Evolution Data Flow: Failed')
                        return False
                else:
                    print('❌ Federation Data Flow: Failed')
                    return False
            else:
                print('❌ KPIs Data Flow: Failed')
                return False
        else:
            print('❌ Health Data Flow: Failed')
            return False
    except Exception as e:
        print(f'❌ Data Flow Test: Error - {str(e)}')
        return False

def main():
    """Main integration test"""
    print("🚀 Enhanced Telecom AI 4.0 - Integration Test")
    print("=" * 60)
    
    # Wait a moment for services to start
    print("⏳ Waiting for services to initialize...")
    time.sleep(3)
    
    # Test all components
    backend_ok = test_backend_apis()
    prometheus_ok = test_prometheus()
    frontend_ok = test_frontend()
    data_flow_ok = test_data_flow()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    print(f"Backend APIs: {'✅ WORKING' if backend_ok else '❌ ISSUES'}")
    print(f"Prometheus: {'✅ WORKING' if prometheus_ok else '❌ ISSUES'}")
    print(f"React Frontend: {'✅ WORKING' if frontend_ok else '❌ ISSUES'}")
    print(f"Data Flow: {'✅ WORKING' if data_flow_ok else '❌ ISSUES'}")
    
    all_working = backend_ok and prometheus_ok and frontend_ok and data_flow_ok
    
    print("\n" + "=" * 60)
    if all_working:
        print("🎉 ALL COMPONENTS WORKING TOGETHER SUCCESSFULLY!")
        print("✅ Enhanced Telecom AI 4.0 System is fully operational")
    else:
        print("⚠️  SOME COMPONENTS HAVE ISSUES")
        print("❌ System integration needs attention")
    print("=" * 60)

if __name__ == "__main__":
    main()
