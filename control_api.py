#!/usr/bin/env python3
"""
Control API - Safe Command Execution Interface
Provides a secure API for executing system commands with authentication and logging
"""

from flask import Flask, request, jsonify
import subprocess
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Configuration
API_KEY = os.getenv("CONTROL_API_KEY", "secret")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
PORT = int(os.getenv("CONTROL_API_PORT", "5001"))

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Allowed commands for security
ALLOWED_COMMANDS = {
    'kubectl': [
        'scale deployment',
        'get deployment',
        'get pods',
        'describe deployment',
        'describe pod'
    ],
    'systemctl': [
        'restart',
        'start',
        'stop',
        'status',
        'enable',
        'disable'
    ],
    'echo': ['*'],  # Allow all echo commands for testing
    'docker': [
        'ps',
        'logs',
        'restart',
        'stop',
        'start'
    ]
}

def authenticate_request() -> bool:
    """Authenticate API request"""
    api_key = request.headers.get('X-API-Key')
    return api_key == API_KEY

def is_command_allowed(command: str) -> bool:
    """Check if command is in allowed list"""
    parts = command.split()
    if not parts:
        return False
    
    base_command = parts[0]
    if base_command not in ALLOWED_COMMANDS:
        return False
    
    allowed_subcommands = ALLOWED_COMMANDS[base_command]
    if '*' in allowed_subcommands:
        return True
    
    # Check if any allowed subcommand matches
    command_str = ' '.join(parts[1:])
    for allowed in allowed_subcommands:
        if command_str.startswith(allowed):
            return True
    
    return False

def execute_command(command: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute command safely with timeout"""
    try:
        logger.info(f"Executing command: {command}")
        
        # Check if command is allowed
        if not is_command_allowed(command):
            logger.warning(f"Command not allowed: {command}")
            return False, f"Command not allowed: {command}"
        
        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        success = result.returncode == 0
        output = result.stdout + "\n" + result.stderr
        
        if success:
            logger.info(f"Command succeeded: {command}")
        else:
            logger.error(f"Command failed: {command} - {output}")
        
        return success, output
        
    except subprocess.TimeoutExpired:
        logger.error(f"Command timeout: {command}")
        return False, "Command execution timeout"
    except Exception as e:
        logger.error(f"Command error: {command} - {str(e)}")
        return False, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/exec', methods=['POST'])
def exec_command():
    """Execute command endpoint"""
    if not authenticate_request():
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        command = data.get('command')
        timeout = data.get('timeout', 30)
        
        if not command:
            return jsonify({'success': False, 'error': 'No command provided'}), 400
        
        # Execute command
        success, output = execute_command(command, timeout)
        
        return jsonify({
            'success': success,
            'output': output,
            'command': command,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/kubectl/scale', methods=['POST'])
def kubectl_scale():
    """Kubernetes scale endpoint"""
    if not authenticate_request():
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        deployment = data.get('deployment')
        replicas = data.get('replicas')
        
        if not deployment or replicas is None:
            return jsonify({'success': False, 'error': 'Missing deployment or replicas'}), 400
        
        command = f"kubectl scale deployment {deployment} --replicas={replicas}"
        success, output = execute_command(command)
        
        return jsonify({
            'success': success,
            'output': output,
            'deployment': deployment,
            'replicas': replicas,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Kubectl scale error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/systemctl/<action>', methods=['POST'])
def systemctl_action(action):
    """Systemctl action endpoint"""
    if not authenticate_request():
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        data = request.json or {}
        service = data.get('service')
        
        if not service:
            return jsonify({'success': False, 'error': 'Missing service name'}), 400
        
        command = f"sudo systemctl {action} {service}"
        success, output = execute_command(command)
        
        return jsonify({
            'success': success,
            'output': output,
            'action': action,
            'service': service,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Systemctl {action} error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/allowed-commands', methods=['GET'])
def get_allowed_commands():
    """Get list of allowed commands"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    return jsonify({
        'allowed_commands': ALLOWED_COMMANDS,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics"""
    if not authenticate_request():
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get basic system metrics
        import psutil
        
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info(f"Starting Control API on port {PORT}")
    logger.info(f"API Key: {API_KEY}")
    logger.info(f"Allowed commands: {list(ALLOWED_COMMANDS.keys())}")
    
    app.run(host='127.0.0.1', port=PORT, debug=False)
