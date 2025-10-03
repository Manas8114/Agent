# Enhanced Telecom AI System - Deployment Guide

## 🚀 **Quick Start**

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Node.js 16+
- 8GB RAM minimum
- 20GB disk space

### 1. Clone and Setup
```bash
git clone <repository-url>
cd enhanced_telecom_ai
```

### 2. Install Dependencies
```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd dashboard/frontend
npm install
cd ../..
```

### 3. Start All Services
```bash
# Option 1: Use the provided script
./start_clean.bat  # Windows
./start_clean.sh    # Linux/Mac

# Option 2: Manual start
python simple_test_server.py &
python monitoring_server.py &
python grafana_server.py &
cd dashboard/frontend && npm start &
```

### 4. Access Services
- **Main Dashboard**: http://localhost:3000
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Monitoring**: http://localhost:9090
- **Analytics**: http://localhost:3001

## 🐳 **Docker Deployment**

### Single Container Deployment
```bash
# Build and run
docker build -t telecom-ai .
docker run -p 8000:8000 -p 3000:3000 telecom-ai
```

### Multi-Container Deployment
```bash
# Start all services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ☸️ **Kubernetes Deployment**

### 1. Create Namespace
```bash
kubectl create namespace telecom-ai
```

### 2. Deploy with Helm
```bash
# Add Helm repository
helm repo add telecom-ai ./helm-charts

# Install the chart
helm install telecom-ai telecom-ai/telecom-ai \
  --namespace telecom-ai \
  --set image.tag=latest \
  --set replicas.api=3 \
  --set replicas.agents=2
```

### 3. Access Services
```bash
# Get service URLs
kubectl get services -n telecom-ai

# Port forward for local access
kubectl port-forward svc/telecom-api 8000:8000 -n telecom-ai
kubectl port-forward svc/telecom-dashboard 3000:3000 -n telecom-ai
```

## 🔧 **Production Configuration**

### Environment Variables
```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Database Configuration
export DATABASE_URL=postgresql://user:pass@localhost:5432/telecom_ai
export REDIS_URL=redis://localhost:6379

# MLflow Configuration
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=telecom_ai_production

# Monitoring Configuration
export PROMETHEUS_URL=http://localhost:9090
export GRAFANA_URL=http://localhost:3001
```

### Scaling Configuration
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  api:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
  
  agents:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 📊 **Monitoring Setup**

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'telecom-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard Import
1. Access Grafana at http://localhost:3001
2. Go to "Import Dashboard"
3. Upload `monitoring/grafana_dashboards/telecom_ai_dashboard.json`
4. Configure data source to point to Prometheus

### Alert Configuration
```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@telecom-ai.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/'
```

## 🔒 **Security Configuration**

### SSL/TLS Setup
```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update nginx configuration
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:3000;
    }
}
```

### Authentication Setup
```python
# Add to api/server.py
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTAuthentication

# Configure JWT authentication
jwt_authentication = JWTAuthentication(
    secret=SECRET_KEY,
    lifetime_seconds=3600,
    tokenUrl="auth/jwt/login"
)
```

## 📈 **Performance Optimization**

### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_qos_data_timestamp ON qos_data(timestamp);
CREATE INDEX idx_qos_data_cell_id ON qos_data(cell_id);
CREATE INDEX idx_energy_data_gnb_id ON energy_data(gnb_id);
```

### Caching Configuration
```python
# Add Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Load Balancing
```nginx
# nginx.conf
upstream telecom_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location /api/ {
        proxy_pass http://telecom_api;
    }
}
```

## 🚨 **Troubleshooting**

### Common Issues

#### 1. Port Conflicts
```bash
# Check for port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000

# Kill processes using ports
sudo fuser -k 8000/tcp
sudo fuser -k 3000/tcp
```

#### 2. Database Connection Issues
```bash
# Check database status
docker ps | grep postgres
docker logs postgres_container

# Reset database
docker-compose down -v
docker-compose up -d
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits
docker run --memory=4g telecom-ai
```

#### 4. MLflow Connection Issues
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Check MLflow UI
open http://localhost:5000
```

### Log Analysis
```bash
# View application logs
docker-compose logs -f api
docker-compose logs -f agents

# View system logs
journalctl -u telecom-ai -f

# Analyze performance
docker exec -it telecom-ai-container top
```

## 📋 **Health Checks**

### API Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Agent Health Check
```bash
curl http://localhost:8000/api/v1/agents/status
```

### Database Health Check
```bash
curl http://localhost:8000/api/v1/telecom/kpis
```

### Monitoring Health Check
```bash
curl http://localhost:9090/health
curl http://localhost:3001/health
```

## 🔄 **Backup and Recovery**

### Database Backup
```bash
# Create backup
pg_dump telecom_ai > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
psql telecom_ai < backup_20231201_120000.sql
```

### Configuration Backup
```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  monitoring/ \
  docker-compose.yml \
  requirements.txt
```

### Model Backup
```bash
# Backup ML models
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  models/ \
  mlruns/
```

## 📞 **Support**

### Documentation
- [API Documentation](http://localhost:8000/docs)
- [System Architecture](docs/ARCHITECTURE.md)
- [User Guide](docs/USER_GUIDE.md)

### Contact
- **Technical Support**: support@telecom-ai.com
- **Documentation Issues**: docs@telecom-ai.com
- **Security Issues**: security@telecom-ai.com

### Community
- **GitHub Issues**: [Report Issues](https://github.com/telecom-ai/issues)
- **Discord**: [Join Community](https://discord.gg/telecom-ai)
- **Stack Overflow**: [Ask Questions](https://stackoverflow.com/questions/tagged/telecom-ai)
