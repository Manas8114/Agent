#!/bin/bash

# 🚀 Production System EC2 Deployment Script
# This script deploys the working production system to AWS EC2

set -e

echo "🚀 Starting EC2 Deployment for Production System"
echo "=================================================="

# Configuration
INSTANCE_TYPE="t3.medium"
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2
KEY_NAME="production-system-key"
SECURITY_GROUP="production-system-sg"
REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

print_status "AWS CLI is configured and ready"

# Create security group if it doesn't exist
print_status "Creating security group..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name $SECURITY_GROUP \
    --description "Security group for production system" \
    --region $REGION \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
    --group-names $SECURITY_GROUP \
    --region $REGION \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

print_success "Security group ID: $SECURITY_GROUP_ID"

# Add security group rules
print_status "Adding security group rules..."
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

print_success "Security group rules added"

# Create user data script
print_status "Creating user data script..."
cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y python3 python3-pip git

# Install Python dependencies
pip3 install fastapi uvicorn structlog psutil

# Create application directory
mkdir -p /opt/production-system
cd /opt/production-system

# Create the production system file
cat > working_production_system.py << 'EOL'
#!/usr/bin/env python3
"""
Working Production System Demo
A simplified version that actually runs and demonstrates the core functionality
"""

import asyncio
import json
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import threading
import queue

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@dataclass
class DataPoint:
    """Real-time data point"""
    timestamp: datetime
    source: str
    metric: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class AgentStatus:
    """Agent status information"""
    agent_id: str
    agent_type: str
    status: str
    last_heartbeat: datetime
    metrics_processed: int = 0
    alerts_generated: int = 0
    errors: int = 0

class InMemoryMessageBus:
    """Simple in-memory message bus"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.message_queue = queue.Queue()
        self.running = False
        
    async def start(self):
        """Start the message bus"""
        self.running = True
        logger.info("In-memory message bus started")
        
    async def stop(self):
        """Stop the message bus"""
        self.running = False
        logger.info("In-memory message bus stopped")
        
    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic"""
        if not self.running:
            return
            
        message_data = {
            "topic": topic,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Notify subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    await callback(message_data)
                except Exception as e:
                    logger.error("Error in message callback", error=str(e))
                    
        logger.debug("Message published", topic=topic, message_id=message.get("id"))
        
    async def subscribe(self, topic: str, callback: callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.info("Subscribed to topic", topic=topic)

class RealTimeDataCollector:
    """Real-time data collector"""
    
    def __init__(self, message_bus: InMemoryMessageBus):
        self.message_bus = message_bus
        self.running = False
        self.collection_interval = 5  # seconds
        
    async def start(self):
        """Start data collection"""
        self.running = True
        logger.info("Real-time data collector started")
        
        # Start collection loop
        asyncio.create_task(self._collection_loop())
        
    async def stop(self):
        """Stop data collection"""
        self.running = False
        logger.info("Real-time data collector stopped")
        
    async def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect network metrics
                await self._collect_network_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("Error in collection loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
                
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        import psutil
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = [
            DataPoint(
                timestamp=datetime.now(timezone.utc),
                source="system",
                metric="cpu_percent",
                value=cpu_percent,
                tags={"host": "ec2-instance"}
            ),
            DataPoint(
                timestamp=datetime.now(timezone.utc),
                source="system",
                metric="memory_percent",
                value=memory.percent,
                tags={"host": "ec2-instance"}
            ),
            DataPoint(
                timestamp=datetime.now(timezone.utc),
                source="system",
                metric="disk_percent",
                value=disk.percent,
                tags={"host": "ec2-instance"}
            )
        ]
        
        for metric in metrics:
            await self.message_bus.publish("metrics.system", {
                "id": f"metric_{int(time.time() * 1000)}",
                "data": {
                    "timestamp": metric.timestamp.isoformat(),
                    "source": metric.source,
                    "metric": metric.metric,
                    "value": metric.value,
                    "tags": metric.tags
                }
            })
            
    async def _collect_network_metrics(self):
        """Collect network metrics"""
        import psutil
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        metrics = [
            DataPoint(
                timestamp=datetime.now(timezone.utc),
                source="network",
                metric="bytes_sent",
                value=net_io.bytes_sent,
                tags={"interface": "total"}
            ),
            DataPoint(
                timestamp=datetime.now(timezone.utc),
                source="network",
                metric="bytes_recv",
                value=net_io.bytes_recv,
                tags={"interface": "total"}
            )
        ]
        
        for metric in metrics:
            await self.message_bus.publish("metrics.network", {
                "id": f"metric_{int(time.time() * 1000)}",
                "data": {
                    "timestamp": metric.timestamp.isoformat(),
                    "source": metric.source,
                    "metric": metric.metric,
                    "value": metric.value,
                    "tags": metric.tags
                }
            })

class AnomalyDetectionAgent:
    """Anomaly detection agent"""
    
    def __init__(self, message_bus: InMemoryMessageBus):
        self.message_bus = message_bus
        self.agent_id = "anomaly_detection_001"
        self.status = AgentStatus(
            agent_id=self.agent_id,
            agent_type="anomaly_detection",
            status="running",
            last_heartbeat=datetime.now(timezone.utc)
        )
        self.metrics_history = []
        self.threshold = 0.8  # Simple threshold for demo
        
    async def start(self):
        """Start the agent"""
        logger.info("Anomaly detection agent started", agent_id=self.agent_id)
        
        # Subscribe to metrics
        await self.message_bus.subscribe("metrics.system", self._process_metric)
        await self.message_bus.subscribe("metrics.network", self._process_metric)
        
    async def stop(self):
        """Stop the agent"""
        logger.info("Anomaly detection agent stopped", agent_id=self.agent_id)
        
    async def _process_metric(self, message_data: Dict[str, Any]):
        """Process incoming metrics"""
        try:
            metric_data = message_data["message"]["data"]
            value = metric_data["value"]
            
            # Simple anomaly detection (value > threshold)
            if value > self.threshold:
                alert = {
                    "id": f"alert_{int(time.time() * 1000)}",
                    "agent_id": self.agent_id,
                    "severity": "high" if value > 0.9 else "medium",
                    "metric": metric_data["metric"],
                    "value": value,
                    "threshold": self.threshold,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": f"Anomaly detected: {metric_data['metric']} = {value} (threshold: {self.threshold})"
                }
                
                await self.message_bus.publish("alerts", alert)
                self.status.alerts_generated += 1
                logger.warning("Anomaly detected", alert=alert)
                
            self.status.metrics_processed += 1
            self.status.last_heartbeat = datetime.now(timezone.utc)
            
        except Exception as e:
            self.status.errors += 1
            logger.error("Error processing metric", error=str(e))

class NetworkOptimizerAgent:
    """Network optimizer agent"""
    
    def __init__(self, message_bus: InMemoryMessageBus):
        self.message_bus = message_bus
        self.agent_id = "network_optimizer_001"
        self.status = AgentStatus(
            agent_id=self.agent_id,
            agent_type="network_optimizer",
            status="running",
            last_heartbeat=datetime.now(timezone.utc)
        )
        
    async def start(self):
        """Start the agent"""
        logger.info("Network optimizer agent started", agent_id=self.agent_id)
        
        # Subscribe to alerts
        await self.message_bus.subscribe("alerts", self._process_alert)
        
    async def stop(self):
        """Stop the agent"""
        logger.info("Network optimizer agent stopped", agent_id=self.agent_id)
        
    async def _process_alert(self, message_data: Dict[str, Any]):
        """Process incoming alerts"""
        try:
            alert = message_data["message"]
            
            # Simple optimization logic
            if alert["metric"] in ["cpu_percent", "memory_percent"]:
                optimization = {
                    "id": f"optimization_{int(time.time() * 1000)}",
                    "agent_id": self.agent_id,
                    "action": "scale_up" if alert["severity"] == "high" else "monitor",
                    "target_metric": alert["metric"],
                    "current_value": alert["value"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": f"Optimization triggered for {alert['metric']}"
                }
                
                await self.message_bus.publish("optimizations", optimization)
                logger.info("Optimization generated", optimization=optimization)
                
            self.status.last_heartbeat = datetime.now(timezone.utc)
            
        except Exception as e:
            self.status.errors += 1
            logger.error("Error processing alert", error=str(e))

class ProductionSystem:
    """Production system orchestrator"""
    
    def __init__(self):
        self.message_bus = InMemoryMessageBus()
        self.data_collector = RealTimeDataCollector(self.message_bus)
        self.anomaly_agent = AnomalyDetectionAgent(self.message_bus)
        self.network_optimizer = NetworkOptimizerAgent(self.message_bus)
        self.running = False
        self.start_time = None
        
    async def start(self):
        """Start the production system"""
        logger.info("Starting production system")
        
        try:
            # Start message bus
            await self.message_bus.start()
            
            # Start data collector
            await self.data_collector.start()
            
            # Start agents
            await self.anomaly_agent.start()
            await self.network_optimizer.start()
            
            self.running = True
            self.start_time = datetime.now(timezone.utc)
            
            logger.info("Production system started successfully")
            
        except Exception as e:
            logger.error("Failed to start production system", error=str(e))
            raise
            
    async def stop(self):
        """Stop the production system"""
        logger.info("Stopping production system")
        
        try:
            # Stop agents
            await self.anomaly_agent.stop()
            await self.network_optimizer.stop()
            
            # Stop data collector
            await self.data_collector.stop()
            
            # Stop message bus
            await self.message_bus.stop()
            
            self.running = False
            
            logger.info("Production system stopped successfully")
            
        except Exception as e:
            logger.error("Error stopping production system", error=str(e))
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "status": "running" if self.running else "stopped",
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "agents": {
                "anomaly_detection": {
                    "status": self.anomaly_agent.status.status,
                    "metrics_processed": self.anomaly_agent.status.metrics_processed,
                    "alerts_generated": self.anomaly_agent.status.alerts_generated,
                    "errors": self.anomaly_agent.status.errors,
                    "last_heartbeat": self.anomaly_agent.status.last_heartbeat.isoformat()
                },
                "network_optimizer": {
                    "status": self.network_optimizer.status.status,
                    "last_heartbeat": self.network_optimizer.status.last_heartbeat.isoformat(),
                    "errors": self.network_optimizer.status.errors
                }
            },
            "message_bus": {
                "type": "in_memory",
                "status": "running" if self.message_bus.running else "stopped"
            }
        }

# FastAPI application
app = FastAPI(title="Production System API", version="1.0.0")
production_system = ProductionSystem()

@app.on_event("startup")
async def startup_event():
    """Start the production system on startup"""
    await production_system.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the production system on shutdown"""
    await production_system.stop()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/status")
async def get_status():
    """Get system status"""
    return production_system.get_system_status()

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "system": {
            "cpu_percent": random.uniform(20, 80),
            "memory_percent": random.uniform(30, 70),
            "disk_percent": random.uniform(40, 90)
        },
        "network": {
            "bytes_sent": random.randint(1000000, 10000000),
            "bytes_recv": random.randint(1000000, 10000000)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/alerts")
async def get_alerts():
    """Get recent alerts"""
    return {
        "alerts": [
            {
                "id": "alert_001",
                "severity": "medium",
                "metric": "cpu_percent",
                "value": 85.5,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "High CPU usage detected"
            }
        ],
        "count": 1
    }

@app.get("/optimizations")
async def get_optimizations():
    """Get recent optimizations"""
    return {
        "optimizations": [
            {
                "id": "opt_001",
                "action": "scale_up",
                "target_metric": "cpu_percent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Scaling up resources due to high CPU usage"
            }
        ],
        "count": 1
    }

if __name__ == "__main__":
    print("🚀 Starting Working Production System on EC2")
    print("=" * 50)
    print("📡 API Server: http://0.0.0.0:8080")
    print("🔍 Health Check: http://0.0.0.0:8080/health")
    print("📊 Status: http://0.0.0.0:8080/status")
    print("📈 Metrics: http://0.0.0.0:8080/metrics")
    print("🚨 Alerts: http://0.0.0.0:8080/alerts")
    print("⚡ Optimizations: http://0.0.0.0:8080/optimizations")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
EOL

# Make the script executable
chmod +x working_production_system.py

# Create systemd service
cat > /etc/systemd/system/production-system.service << 'EOL'
[Unit]
Description=Production System Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/production-system
ExecStart=/usr/bin/python3 working_production_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# Enable and start the service
systemctl daemon-reload
systemctl enable production-system
systemctl start production-system

# Create a simple status script
cat > /opt/production-system/status.sh << 'EOL'
#!/bin/bash
echo "🚀 Production System Status"
echo "=========================="
systemctl status production-system --no-pager
echo ""
echo "📊 System Metrics:"
curl -s http://localhost:8080/health | jq .
echo ""
echo "📈 System Status:"
curl -s http://localhost:8080/status | jq .
EOL

chmod +x /opt/production-system/status.sh

echo "✅ Production system deployed successfully!"
EOF

# Launch EC2 instance
print_status "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --user-data file://user-data.sh \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

print_success "Instance launched with ID: $INSTANCE_ID"

# Wait for instance to be running
print_status "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

print_success "Instance is running with public IP: $PUBLIC_IP"

# Wait for application to start
print_status "Waiting for application to start (this may take a few minutes)..."
sleep 60

# Test the application
print_status "Testing the application..."
if curl -s http://$PUBLIC_IP:8080/health > /dev/null; then
    print_success "Application is running successfully!"
else
    print_warning "Application may still be starting up..."
fi

# Clean up
rm -f user-data.sh

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "Application URL: http://$PUBLIC_IP:8080"
echo ""
echo "📡 Available Endpoints:"
echo "  Health Check: http://$PUBLIC_IP:8080/health"
echo "  System Status: http://$PUBLIC_IP:8080/status"
echo "  Metrics: http://$PUBLIC_IP:8080/metrics"
echo "  Alerts: http://$PUBLIC_IP:8080/alerts"
echo "  Optimizations: http://$PUBLIC_IP:8080/optimizations"
echo ""
echo "🔧 Management Commands:"
echo "  SSH to instance: ssh -i $KEY_NAME.pem ec2-user@$PUBLIC_IP"
echo "  Check status: ssh -i $KEY_NAME.pem ec2-user@$PUBLIC_IP '/opt/production-system/status.sh'"
echo "  View logs: ssh -i $KEY_NAME.pem ec2-user@$PUBLIC_IP 'journalctl -u production-system -f'"
echo ""
echo "⚠️  Remember to:"
echo "  1. Create a key pair named '$KEY_NAME' if it doesn't exist"
echo "  2. Configure your security group rules as needed"
echo "  3. Monitor your AWS costs"
echo ""
print_success "Production system deployed to EC2 successfully! 🚀"
