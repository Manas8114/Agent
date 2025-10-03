#!/usr/bin/env python3
"""
Enhanced Telecom AI System - Server Startup Script
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables
os.environ.setdefault('PYTHONPATH', str(current_dir))

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced Telecom AI System...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔄 Dashboard: http://localhost:3000 (when running)")
        print("📊 Monitoring: http://localhost:9090 (Prometheus)")
        print("📈 Grafana: http://localhost:3001 (admin/admin)")
        print("\n" + "="*60)
        
        # Import and run the server
        from api.server import app
        import uvicorn
        
        uvicorn.run(
            "api.server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
