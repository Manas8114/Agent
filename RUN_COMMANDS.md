# Enhanced Telecom AI 4.0 - Complete Run Commands Guide

## 🚀 Quick Start Commands

### 1. Backend Services

#### Start FastAPI Backend Server
```bash
# Navigate to project root
cd C:\Users\msgok\Desktop\G_agent\6G\New\enhanced_telecom_ai

# Start FastAPI server (port 8000)
python run_server.py
```

#### Start Prometheus Metrics Server
```bash
# In a separate terminal
python prometheus_server.py
```

### 2. Frontend Services

#### Start React Dashboard
```bash
# Navigate to frontend directory
cd dashboard/frontend

# Install dependencies (first time only)
npm install

# Start React development server (port 3000)
npm start
```

### 3. Complete System Startup

#### Option A: Manual Startup (Recommended for Development)
```bash
# Terminal 1: Start FastAPI Backend
python run_server.py

# Terminal 2: Start Prometheus Server
python prometheus_server.py

# Terminal 3: Start React Frontend
cd dashboard/frontend
npm start
```

#### Option B: Background Services
```bash
# Start all services in background
python run_server.py &
python prometheus_server.py &
cd dashboard/frontend && npm start &
```

## 🔧 Development Commands

### Backend Development

#### Run with Auto-reload
```bash
# FastAPI with auto-reload for development
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```

#### Run Specific Tests
```bash
# Test individual components
python test_ibn.py
python test_zta.py
python test_quantum_safe.py
python test_self_evolving.py
python test_federation.py
python test_observability.py

# Run integration tests
python test_integration.py
python final_integration_test.py
```

#### Run Individual AI 4.0 Features
```bash
# Test Federation Manager
python run_federation.py

# Test Observability
python run_observability.py
```

### Frontend Development

#### React Development Commands
```bash
cd dashboard/frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint
```

#### Install Additional Dependencies
```bash
# Install required packages
npm install lucide-react
npm install framer-motion
npm install axios
```

## 🌐 Access Points

### Web Interfaces
- **Main Dashboard**: http://localhost:3000
- **Real-Data Dashboard**: http://localhost:3000/real-data
- **API Documentation**: http://localhost:8000/docs
- **Prometheus Metrics**: http://localhost:9090

### API Endpoints
- **Health Check**: http://localhost:8000/api/v1/health
- **KPIs**: http://localhost:8000/api/v1/telecom/kpis
- **Federation**: http://localhost:8000/api/v1/telecom/federation
- **Self-Evolution**: http://localhost:8000/api/v1/telecom/self-evolution
- **Quantum Security**: http://localhost:8000/api/v1/telecom/quantum-status
- **ZTA Status**: http://localhost:8000/api/v1/telecom/zta-status
- **Real Data**: http://localhost:8000/api/v1/real-data

## 🐳 Docker Commands

### Build and Run with Docker
```bash
# Build backend image
docker build -t telecom-ai-backend .

# Build frontend image
cd dashboard/frontend
docker build -t telecom-ai-frontend .

# Run with docker-compose
docker-compose up -d

# Run in production mode
docker-compose -f docker-compose.production.yml up -d
```

### Docker Development
```bash
# Run backend in container
docker run -p 8000:8000 telecom-ai-backend

# Run frontend in container
docker run -p 3000:3000 telecom-ai-frontend
```

## ☸️ Kubernetes Commands

### Deploy to Kubernetes
```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/telecom-ai-deployment.yaml

# Check deployment status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/telecom-ai-backend
kubectl logs -f deployment/telecom-ai-frontend
```

## 📊 Monitoring Commands

### Prometheus and Grafana
```bash
# Start Prometheus server
python prometheus_server.py

# Access Prometheus metrics
curl http://localhost:9090/metrics

# Start Grafana (if configured)
# Access at http://localhost:3001
```

### System Health Checks
```bash
# Check backend health
curl http://localhost:8000/api/v1/health

# Check all API endpoints
python test_individual_apis.py

# Run comprehensive integration test
python final_integration_test.py
```

## 🔍 Debugging Commands

### Backend Debugging
```bash
# Run with debug logging
python -m uvicorn api.server:app --reload --log-level debug

# Check backend logs
tail -f logs/backend.log

# Test specific endpoints
curl -v http://localhost:8000/api/v1/health
```

### Frontend Debugging
```bash
# Start with debug mode
REACT_APP_DEBUG=true npm start

# Check browser console for errors
# Open Developer Tools (F12)
```

## 🧪 Testing Commands

### Unit Tests
```bash
# Test individual components
python -m pytest tests/test_ibn.py
python -m pytest tests/test_zta.py
python -m pytest tests/test_quantum.py
python -m pytest tests/test_federation.py
python -m pytest tests/test_self_evolution.py
```

### Integration Tests
```bash
# Test all components together
python test_integration.py

# Test API endpoints
python test_individual_apis.py

# Test data flow
python final_integration_test.py
```

### Frontend Tests
```bash
cd dashboard/frontend

# Run React tests
npm test

# Run with coverage
npm test -- --coverage

# Test specific components
npm test -- --testNamePattern="AI4Dashboard"
```

## 🚀 Production Deployment

### Production Backend
```bash
# Run with production settings
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (Linux/Mac)
gunicorn api.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Production Frontend
```bash
cd dashboard/frontend

# Build for production
npm run build

# Serve with nginx
# Copy build files to nginx web root
```

## 🔧 Maintenance Commands

### Database Operations
```bash
# Initialize database
python -c "from data.data_manager import DataManager; dm = DataManager(); dm.initialize_schema()"

# Reset database
python -c "from data.data_manager import DataManager; dm = DataManager(); dm.reset_database()"
```

### Log Management
```bash
# View logs
tail -f logs/backend.log
tail -f logs/frontend.log

# Clear logs
rm logs/*.log
```

### System Cleanup
```bash
# Stop all services
pkill -f "python run_server.py"
pkill -f "python prometheus_server.py"
pkill -f "npm start"

# Clean up processes
netstat -ano | findstr :8000
netstat -ano | findstr :3000
netstat -ano | findstr :9090
```

## 📋 Environment Setup

### Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Node.js Environment
```bash
cd dashboard/frontend

# Install Node.js dependencies
npm install

# Update dependencies
npm update
```

## 🎯 Quick Commands Summary

### Essential Commands
```bash
# Start everything
python run_server.py & python prometheus_server.py & cd dashboard/frontend && npm start

# Test system
python final_integration_test.py

# Check status
curl http://localhost:8000/api/v1/health
curl http://localhost:3000
curl http://localhost:9090
```

### Emergency Commands
```bash
# Kill all processes
taskkill /f /im python.exe
taskkill /f /im node.exe

# Restart everything
python run_server.py
python prometheus_server.py
cd dashboard/frontend && npm start
```

## 📝 Notes

- **Ports Used**: 8000 (Backend), 3000 (Frontend), 9090 (Prometheus)
- **Default URLs**: 
  - Dashboard: http://localhost:3000
  - API Docs: http://localhost:8000/docs
  - Metrics: http://localhost:9090
- **Logs**: Check console output for real-time logs
- **Troubleshooting**: Run `python final_integration_test.py` to diagnose issues

## 🆘 Troubleshooting

### Common Issues
```bash
# Port already in use
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Module not found
pip install -r requirements.txt

# React build issues
cd dashboard/frontend
npm install
npm start
```

### Health Checks
```bash
# Check if services are running
curl http://localhost:8000/api/v1/health
curl http://localhost:3000
curl http://localhost:9090

# Run integration test
python final_integration_test.py
```

---

**🎉 Enhanced Telecom AI 4.0 System - All Run Commands Ready!**
