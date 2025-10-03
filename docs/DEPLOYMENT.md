# Deployment Guide

This guide covers deployment options for the Enhanced Telecom AI System.

## 🐳 Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 20GB+ disk space

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd enhanced_telecom_ai

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Service Configuration

#### API Server
```yaml
api:
  build: .
  ports:
    - "8000:8000"
  environment:
    - ENVIRONMENT=production
    - DATABASE_URL=sqlite:///app/data/telecom_ai.db
  volumes:
    - ./data:/app/data
```

#### Dashboard
```yaml
dashboard:
  build: ./dashboard/frontend
  ports:
    - "3000:3000"
  environment:
    - REACT_APP_API_URL=http://localhost:8000/api/v1
```

#### Monitoring
```yaml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
  image: grafana/grafana:latest
  ports:
    - "3001:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Scaling Services
```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale with load balancer
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes 1.20+
- Helm 3.0+
- kubectl configured

### Helm Chart Installation
```bash
# Add Helm repository
helm repo add telecom-ai ./k8s/helm-chart

# Install chart
helm install telecom-ai telecom-ai/telecom-ai \
  --namespace telecom-ai \
  --create-namespace \
  --set image.tag=latest \
  --set replicaCount=3
```

### Custom Configuration
```yaml
# values.yaml
api:
  replicaCount: 3
  image:
    repository: telecom-ai
    tag: latest
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"

dashboard:
  replicaCount: 2
  image:
    repository: telecom-ai-dashboard
    tag: latest

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "admin"
```

### Production Deployment
```bash
# Create namespace
kubectl create namespace telecom-ai

# Install with production values
helm install telecom-ai ./k8s/helm-chart \
  --namespace telecom-ai \
  --values k8s/values-production.yaml
```

## 🌐 Cloud Deployment

### AWS Deployment

#### ECS with Fargate
```yaml
# ecs-task-definition.json
{
  "family": "telecom-ai",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-registry/telecom-ai:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

#### EKS Deployment
```bash
# Create EKS cluster
eksctl create cluster --name telecom-ai --region us-west-2

# Deploy application
helm install telecom-ai ./k8s/helm-chart \
  --namespace telecom-ai \
  --create-namespace
```

### Azure Deployment

#### AKS Deployment
```bash
# Create AKS cluster
az aks create --resource-group telecom-ai-rg --name telecom-ai-cluster

# Deploy application
helm install telecom-ai ./k8s/helm-chart \
  --namespace telecom-ai \
  --create-namespace
```

### Google Cloud Deployment

#### GKE Deployment
```bash
# Create GKE cluster
gcloud container clusters create telecom-ai-cluster \
  --zone us-central1-a \
  --num-nodes 3

# Deploy application
helm install telecom-ai ./k8s/helm-chart \
  --namespace telecom-ai \
  --create-namespace
```

## 🔧 Environment Configuration

### Development Environment
```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug
DATABASE_URL=sqlite:///app/data/telecom_ai.db
REDIS_URL=redis://localhost:6379
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Production Environment
```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
DATABASE_URL=postgresql://user:pass@db:5432/telecom_ai
REDIS_URL=redis://redis:6379
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/production) | `development` |
| `DATABASE_URL` | Database connection string | `sqlite:///app/data/telecom_ai.db` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI | `http://localhost:5000` |
| `LOG_LEVEL` | Logging level | `info` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'telecom-ai-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Grafana Dashboards
```bash
# Import dashboards
curl -X POST \
  http://admin:admin@localhost:3001/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/telecom-ai-dashboard.json
```

### Alert Rules
```yaml
# monitoring/rules/telecom-ai-alerts.yml
groups:
  - name: telecom-ai-alerts
    rules:
      - alert: APIDown
        expr: up{job="telecom-ai-api"} == 0
        for: 1m
        labels:
          severity: critical
```

## 🔒 Security Configuration

### SSL/TLS Setup
```nginx
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
}
```

### Security Headers
```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
```

### Authentication
```python
# API authentication
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    if not verify_jwt_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return token
```

## 📈 Performance Optimization

### Database Optimization
```sql
-- Create indexes
CREATE INDEX idx_timestamp ON qos_data(timestamp);
CREATE INDEX idx_agent_name ON agent_metrics(agent_name);
CREATE INDEX idx_metric_name ON metrics_history(metric_name);
```

### Caching Strategy
```python
# Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

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
# nginx load balancing
upstream api_backend {
    server api1:8000 weight=3;
    server api2:8000 weight=2;
    server api3:8000 weight=1;
}

server {
    location /api/ {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🚀 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push Docker images
        run: |
          docker build -t telecom-ai:${{ github.sha }} .
          docker push your-registry/telecom-ai:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          helm upgrade telecom-ai ./k8s/helm-chart \
            --set image.tag=${{ github.sha }}
```

### GitLab CI
```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build -t telecom-ai:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  script:
    - helm upgrade telecom-ai ./k8s/helm-chart \
        --set image.tag=$CI_COMMIT_SHA
```

## 🔍 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000

# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 instead
```

#### Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
# Or in docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G
```

#### Database Connection Issues
```bash
# Check database connectivity
docker-compose exec api python -c "
from enhanced_telecom_ai.data.data_manager import DataManager
dm = DataManager()
print('Database connected successfully')
"
```

#### Service Health Checks
```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check all services
docker-compose ps

# View logs
docker-compose logs api
docker-compose logs dashboard
```

### Performance Monitoring
```bash
# Monitor resource usage
docker stats

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=up

# View Grafana dashboards
open http://localhost:3001
```

## 📋 Maintenance

### Regular Tasks
- Monitor system health and performance
- Update dependencies and security patches
- Backup database and configuration
- Review and optimize resource usage
- Update monitoring dashboards

### Backup Strategy
```bash
# Database backup
docker-compose exec api python -c "
from enhanced_telecom_ai.data.data_manager import DataManager
dm = DataManager()
dm.backup_database('/backup/telecom_ai.db')
"

# Configuration backup
tar -czf config-backup.tar.gz docker-compose.yml monitoring/ nginx/
```

### Update Procedure
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart services
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify deployment
docker-compose ps
curl http://localhost:8000/api/v1/health
```

---

For more detailed information, see the [Architecture Guide](ARCHITECTURE.md) and [Monitoring Guide](MONITORING.md).
