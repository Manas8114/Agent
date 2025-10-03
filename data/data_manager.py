"""
Data Manager for Enhanced Telecom AI System

This module provides data management utilities for the Enhanced Telecom AI System,
including data ingestion, validation, and storage.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataManager:
    """
    Data Manager for the Enhanced Telecom AI System.
    
    Handles data ingestion, validation, storage, and retrieval.
    """
    
    def __init__(self, data_dir: str = "data", db_url: str = None):
        """
        Initialize the Data Manager.
        
        Args:
            data_dir: Directory for data storage
            db_url: Database URL for persistent storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "sample").mkdir(exist_ok=True)
        
        # Initialize database
        if db_url is None:
            db_url = f"sqlite:///{self.data_dir}/telecom_ai.db"
        
        self.db_url = db_url
        self.engine = create_engine(db_url)
        
        # Initialize database schema
        self._initialize_database()
        
        logger.info("Data Manager initialized")
    
    def _initialize_database(self):
        """Initialize database schema."""
        try:
            with self.engine.connect() as conn:
                # Create tables for different data types
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS qos_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        cell_id INTEGER,
                        user_id INTEGER,
                        latency_ms REAL,
                        throughput_mbps REAL,
                        jitter_ms REAL,
                        packet_loss_rate REAL,
                        connection_quality REAL,
                        signal_strength REAL,
                        user_count INTEGER,
                        data_volume_gb REAL,
                        error_count INTEGER,
                        warning_count INTEGER,
                        session_duration_minutes REAL,
                        handover_count INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS traffic_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        traffic_volume REAL,
                        user_count INTEGER,
                        data_volume_gb REAL,
                        peak_hour_indicator BOOLEAN,
                        is_weekend BOOLEAN,
                        network_load_percent REAL,
                        connection_count INTEGER,
                        bandwidth_utilization REAL,
                        protocol_distribution TEXT,
                        geographic_region TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS energy_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        energy_consumption_kwh REAL,
                        temperature REAL,
                        humidity REAL,
                        wind_speed REAL,
                        traffic_load REAL,
                        user_count INTEGER,
                        data_volume REAL,
                        base_station_id INTEGER,
                        cell_load REAL,
                        neighbor_load REAL,
                        distance_to_users REAL,
                        power_efficiency REAL,
                        cooling_load REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS security_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        source_ip TEXT,
                        dest_ip TEXT,
                        source_port INTEGER,
                        dest_port INTEGER,
                        protocol TEXT,
                        packet_size INTEGER,
                        packet_count INTEGER,
                        flow_duration REAL,
                        bytes_sent INTEGER,
                        bytes_received INTEGER,
                        packet_rate REAL,
                        byte_rate REAL,
                        connection_state TEXT,
                        flag_count INTEGER,
                        urgent_packets INTEGER,
                        user_agent TEXT,
                        request_type TEXT,
                        threat_type TEXT,
                        severity TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS failure_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        equipment_id INTEGER,
                        equipment_type TEXT,
                        equipment_age_days INTEGER,
                        temperature_celsius REAL,
                        humidity_percent REAL,
                        cpu_usage_percent REAL,
                        memory_usage_percent REAL,
                        disk_usage_percent REAL,
                        network_load_percent REAL,
                        uptime_hours REAL,
                        restart_count INTEGER,
                        error_log_count INTEGER,
                        warning_log_count INTEGER,
                        failure_occurred BOOLEAN,
                        maintenance_due BOOLEAN,
                        last_maintenance_days REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS data_quality_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        latency_ms REAL,
                        throughput_mbps REAL,
                        jitter_ms REAL,
                        packet_loss_rate REAL,
                        connection_quality REAL,
                        signal_strength REAL,
                        user_count INTEGER,
                        data_volume_gb REAL,
                        error_count INTEGER,
                        warning_count INTEGER,
                        data_completeness REAL,
                        data_consistency REAL,
                        data_accuracy REAL,
                        data_timeliness REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                logger.info("Database schema initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def ingest_data(self, data: pd.DataFrame, data_type: str, 
                   batch_size: int = 1000) -> bool:
        """
        Ingest data into the database.
        
        Args:
            data: Data to ingest
            data_type: Type of data ('qos', 'traffic', 'energy', 'security', 'failure', 'data_quality')
            batch_size: Batch size for insertion
            
        Returns:
            True if successful, False otherwise
        """
        try:
            table_name = f"{data_type}_data"
            
            # Ensure timestamp column exists
            if 'timestamp' not in data.columns:
                data['timestamp'] = datetime.now()
            
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Insert data in batches
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                batch.to_sql(table_name, self.engine, if_exists='append', index=False)
            
            logger.info(f"Ingested {len(data)} records of {data_type} data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest {data_type} data: {e}")
            return False
    
    def retrieve_data(self, data_type: str, start_time: datetime = None, 
                     end_time: datetime = None, limit: int = None) -> pd.DataFrame:
        """
        Retrieve data from the database.
        
        Args:
            data_type: Type of data to retrieve
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records
            
        Returns:
            Retrieved data
        """
        try:
            table_name = f"{data_type}_data"
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            conditions = []
            params = {}
            
            if start_time:
                conditions.append("timestamp >= :start_time")
                params['start_time'] = start_time
            
            if end_time:
                conditions.append("timestamp <= :end_time")
                params['end_time'] = end_time
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                data = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.info(f"Retrieved {len(data)} records of {data_type} data")
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve {data_type} data: {e}")
            return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Args:
            data: Data to validate
            data_type: Type of data
            
        Returns:
            Validation results
        """
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'total_records': len(data),
            'validation_passed': True,
            'issues': [],
            'quality_score': 100
        }
        
        # Check for missing values
        missing_values = data.isnull().sum()
        for col, missing_count in missing_values.items():
            if missing_count > 0:
                missing_ratio = missing_count / len(data)
                validation_results['issues'].append({
                    'type': 'missing_values',
                    'column': col,
                    'count': missing_count,
                    'ratio': missing_ratio,
                    'severity': 'high' if missing_ratio > 0.1 else 'medium' if missing_ratio > 0.05 else 'low'
                })
                validation_results['quality_score'] -= missing_ratio * 20
        
        # Check for outliers
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'id':  # Skip ID column
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                
                if outliers > 0:
                    outlier_ratio = outliers / len(data)
                    validation_results['issues'].append({
                        'type': 'outliers',
                        'column': col,
                        'count': outliers,
                        'ratio': outlier_ratio,
                        'severity': 'high' if outlier_ratio > 0.1 else 'medium' if outlier_ratio > 0.05 else 'low'
                    })
                    validation_results['quality_score'] -= outlier_ratio * 10
        
        # Check for data type consistency
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if numeric data is stored as strings
                try:
                    pd.to_numeric(data[col], errors='raise')
                except:
                    validation_results['issues'].append({
                        'type': 'data_type_inconsistency',
                        'column': col,
                        'expected_type': 'numeric',
                        'actual_type': 'object',
                        'severity': 'medium'
                    })
                    validation_results['quality_score'] -= 5
        
        # Check for duplicate records
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            duplicate_ratio = duplicates / len(data)
            validation_results['issues'].append({
                'type': 'duplicates',
                'count': duplicates,
                'ratio': duplicate_ratio,
                'severity': 'high' if duplicate_ratio > 0.1 else 'medium' if duplicate_ratio > 0.05 else 'low'
            })
            validation_results['quality_score'] -= duplicate_ratio * 20
        
        # Check for timestamp consistency
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            if timestamps.isnull().any():
                validation_results['issues'].append({
                    'type': 'invalid_timestamps',
                    'count': timestamps.isnull().sum(),
                    'severity': 'high'
                })
                validation_results['quality_score'] -= 10
        
        # Determine overall validation status
        high_severity_issues = [issue for issue in validation_results['issues'] 
                               if issue['severity'] == 'high']
        if high_severity_issues:
            validation_results['validation_passed'] = False
        
        validation_results['quality_score'] = max(0, validation_results['quality_score'])
        
        return validation_results
    
    def clean_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Clean data based on validation results.
        
        Args:
            data: Data to clean
            data_type: Type of data
            
        Returns:
            Cleaned data
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'id':  # Skip ID column
                cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
        
        # Handle categorical missing values
        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            cleaned_data[col] = cleaned_data[col].fillna('unknown')
        
        # Remove duplicates
        cleaned_data = cleaned_data.drop_duplicates()
        
        # Handle outliers (cap at 99th percentile)
        for col in numeric_columns:
            if col != 'id':
                upper_bound = cleaned_data[col].quantile(0.99)
                lower_bound = cleaned_data[col].quantile(0.01)
                cleaned_data[col] = cleaned_data[col].clip(lower_bound, upper_bound)
        
        # Ensure timestamp consistency
        if 'timestamp' in cleaned_data.columns:
            cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'])
            # Remove records with invalid timestamps
            cleaned_data = cleaned_data.dropna(subset=['timestamp'])
        
        logger.info(f"Cleaned {data_type} data: {len(data)} -> {len(cleaned_data)} records")
        return cleaned_data
    
    def get_data_summary(self, data_type: str = None) -> Dict[str, Any]:
        """
        Get summary of stored data.
        
        Args:
            data_type: Specific data type to summarize (None for all)
            
        Returns:
            Data summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_types': {}
        }
        
        data_types = [data_type] if data_type else ['qos', 'traffic', 'energy', 'security', 'failure', 'data_quality']
        
        for dt in data_types:
            try:
                table_name = f"{dt}_data"
                
                with self.engine.connect() as conn:
                    # Get record count
                    count_result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table_name}"))
                    record_count = count_result.fetchone()[0]
                    
                    # Get date range
                    date_result = conn.execute(text(f"""
                        SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                        FROM {table_name}
                    """))
                    date_row = date_result.fetchone()
                    min_date = date_row[0] if date_row[0] else None
                    max_date = date_row[1] if date_row[1] else None
                    
                    summary['data_types'][dt] = {
                        'record_count': record_count,
                        'date_range': {
                            'min_date': min_date,
                            'max_date': max_date
                        },
                        'status': 'available' if record_count > 0 else 'empty'
                    }
                    
            except Exception as e:
                summary['data_types'][dt] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return summary
    
    def export_data(self, data_type: str, filepath: str, 
                   format: str = 'csv', start_time: datetime = None, 
                   end_time: datetime = None, limit: int = None) -> bool:
        """
        Export data to file.
        
        Args:
            data_type: Type of data to export
            filepath: Path to export file
            format: Export format ('csv', 'json', 'parquet')
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Retrieve data
            data = self.retrieve_data(data_type, start_time, end_time, limit)
            
            if data.empty:
                logger.warning(f"No data found for {data_type}")
                return False
            
            # Export based on format
            if format == 'csv':
                data.to_csv(filepath, index=False)
            elif format == 'json':
                data.to_json(filepath, orient='records', date_format='iso')
            elif format == 'parquet':
                data.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {len(data)} records of {data_type} data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export {data_type} data: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Backup the database.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # For SQLite, copy the database file
            if 'sqlite' in self.db_url:
                db_path = self.db_url.replace('sqlite:///', '')
                import shutil
                shutil.copy2(db_path, backup_path)
                logger.info(f"Database backed up to {backup_path}")
                return True
            else:
                logger.warning("Database backup not implemented for non-SQLite databases")
                return False
                
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore the database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # For SQLite, copy the backup file
            if 'sqlite' in self.db_url:
                db_path = self.db_url.replace('sqlite:///', '')
                import shutil
                shutil.copy2(backup_path, db_path)
                logger.info(f"Database restored from {backup_path}")
                return True
            else:
                logger.warning("Database restore not implemented for non-SQLite databases")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            return False
