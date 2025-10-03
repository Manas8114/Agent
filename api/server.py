"""
FastAPI Server for Enhanced Telecom AI System

This module provides the main FastAPI application for the Enhanced Telecom AI System.
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator
from datetime import datetime

from api.endpoints import router
from api.models import ErrorResponse
from core.coordinator import AICoordinator
from core.metrics import MetricsCollector
from data.data_manager import DataManager
from data.sample_data_generator import SampleDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
coordinator = None
metrics_collector = None
data_manager = None
sample_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    global coordinator, metrics_collector, data_manager, sample_generator
    
    # Startup
    logger.info("Starting Enhanced Telecom AI System...")
    
    try:
        # Initialize core components
        coordinator = AICoordinator()
        metrics_collector = MetricsCollector()
        data_manager = DataManager()
        sample_generator = SampleDataGenerator()
        
        # Start coordination
        coordinator.start_coordination()
        
        # Generate initial sample data if needed
        await generate_initial_data()
        
        logger.info("Enhanced Telecom AI System started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Telecom AI System...")
    
    try:
        if coordinator:
            coordinator.stop_coordination()
        
        logger.info("Enhanced Telecom AI System shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

async def generate_initial_data():
    """Generate initial sample data if needed."""
    try:
        # Check if data exists
        summary = data_manager.get_data_summary()
        
        # Generate data for each type if empty
        for data_type in ['qos', 'traffic', 'energy', 'security', 'failure', 'data_quality']:
            if summary['data_types'].get(data_type, {}).get('status') == 'empty':
                logger.info(f"Generating initial {data_type} data...")
                
                if data_type == 'qos':
                    data = sample_generator.generate_qos_data(1000)
                elif data_type == 'traffic':
                    data = sample_generator.generate_traffic_data(168)  # 1 week
                elif data_type == 'energy':
                    data = sample_generator.generate_energy_data(168)
                elif data_type == 'security':
                    data = sample_generator.generate_security_data(5000)
                elif data_type == 'failure':
                    data = sample_generator.generate_failure_data(1000)
                elif data_type == 'data_quality':
                    data = sample_generator.generate_data_quality_data(1000)
                
                # Ingest data
                data_manager.ingest_data(data, data_type)
                
                logger.info(f"Generated {len(data)} samples of {data_type} data")
        
    except Exception as e:
        logger.error(f"Failed to generate initial data: {e}")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Enhanced Telecom AI System",
        description="Production-ready Enhanced Telecom AI System with 6 AI agents for anomaly detection, failure prediction, traffic forecasting, energy optimization, security detection, and data quality monitoring.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Enhanced Telecom AI System",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    # Global exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """HTTP exception handler."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "detail": f"HTTP {exc.status_code} error",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return app

# Create the app instance
app = create_app()

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "enhanced_telecom_ai.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server(reload=True)
