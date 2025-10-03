"""
Traffic Forecasting Agent

This agent predicts network traffic patterns using:
- Historical traffic data
- Time series analysis
- Seasonal patterns
- External factors (events, weather, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, fallback to simpler methods if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available, using alternative forecasting methods")

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class TrafficForecastAgent(BaseAgent):
    """
    AI Agent for forecasting network traffic patterns.
    
    Uses time series analysis and machine learning for traffic prediction.
    """
    
    def __init__(self):
        super().__init__("traffic_forecast", "sklearn")
        self.scaler = StandardScaler()
        self.feature_columns = [
            'timestamp', 'hour', 'day_of_week', 'month', 'is_weekend',
            'traffic_volume', 'user_count', 'data_volume_gb', 'peak_hour_indicator'
        ]
        self.prophet_model = None
        self.use_prophet = PROPHET_AVAILABLE
    
    def train(self, data: pd.DataFrame, target_column: str = None, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the traffic forecasting model.
        
        Args:
            data: Historical traffic data
            target_column: Target traffic metric column
            validation_split: Fraction for validation
            
        Returns:
            Training metrics
        """
        logger.info("Training Traffic Forecast Agent...")
        
        target_col = target_column or 'traffic_volume'
        
        # Prepare time series data
        ts_data = self._prepare_time_series_data(data, target_col)
        
        if self.use_prophet:
            # Use Prophet for time series forecasting
            metrics = self._train_prophet_model(ts_data, validation_split)
        else:
            # Use traditional ML approach
            metrics = self._train_ml_model(data, target_col, validation_split)
        
        self.metrics = metrics
        self.is_trained = True
        
        # Log to MLflow
        self.log_metrics_to_mlflow(metrics)
        
        logger.info(f"Traffic Forecast Agent training completed. Metrics: {metrics}")
        return metrics
    
    def _train_prophet_model(self, data: pd.DataFrame, validation_split: float) -> Dict[str, float]:
        """
        Train Prophet model for time series forecasting.
        
        Args:
            data: Time series data with ds and y columns
            validation_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        # Split data
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Initialize and train Prophet
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        self.prophet_model.fit(train_data)
        
        # Make predictions on validation set
        future = self.prophet_model.make_future_dataframe(periods=len(val_data))
        forecast = self.prophet_model.predict(future)
        
        # Get validation predictions
        val_predictions = forecast['yhat'][-len(val_data):].values
        val_actual = val_data['y'].values
        
        # Calculate metrics
        mae = mean_absolute_error(val_actual, val_predictions)
        mse = mean_squared_error(val_actual, val_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(val_actual, val_predictions)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((val_actual - val_predictions) / val_actual)) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'model_type': 'prophet'
        }
        
        return metrics
    
    def _train_ml_model(self, data: pd.DataFrame, target_col: str, 
                       validation_split: float) -> Dict[str, float]:
        """
        Train traditional ML model for traffic forecasting.
        
        Args:
            data: Traffic data
            target_col: Target column name
            validation_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        # Prepare features
        features = self._prepare_features(data)
        target = data[target_col].values
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = target[:split_idx], target[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'model_type': 'random_forest'
        }
        
        return metrics
    
    def predict(self, data: pd.DataFrame, forecast_hours: int = 24) -> np.ndarray:
        """
        Predict future traffic patterns.
        
        Args:
            data: Current traffic data
            forecast_hours: Hours to forecast ahead
            
        Returns:
            Traffic predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.use_prophet and self.prophet_model is not None:
            return self._predict_prophet(forecast_hours)
        else:
            return self._predict_ml(data)
    
    def _predict_prophet(self, forecast_hours: int) -> np.ndarray:
        """
        Make predictions using Prophet model.
        
        Args:
            forecast_hours: Hours to forecast
            
        Returns:
            Predictions array
        """
        future = self.prophet_model.make_future_dataframe(periods=forecast_hours, freq='H')
        forecast = self.prophet_model.predict(future)
        
        # Return only the forecasted values
        return forecast['yhat'][-forecast_hours:].values
    
    def _predict_ml(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ML model.
        
        Args:
            data: Input data
            
        Returns:
            Predictions array
        """
        features = self._prepare_features(data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Evaluate the traffic forecasting model.
        
        Args:
            test_data: Test traffic data
            target_column: Target column name
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        target_col = target_column or 'traffic_volume'
        
        if self.use_prophet and self.prophet_model is not None:
            # Evaluate Prophet model
            ts_data = self._prepare_time_series_data(test_data, target_col)
            future = self.prophet_model.make_future_dataframe(periods=len(ts_data))
            forecast = self.prophet_model.predict(future)
            
            predictions = forecast['yhat'].values
            actual = ts_data['y'].values
            
        else:
            # Evaluate ML model
            features = self._prepare_features(test_data)
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            actual = test_data[target_col].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape
        }
    
    def _prepare_time_series_data(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Prepare data for Prophet time series model.
        
        Args:
            data: Raw traffic data
            target_col: Target column name
            
        Returns:
            Prophet-formatted data
        """
        ts_data = data.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in ts_data.columns:
            ts_data['timestamp'] = pd.date_range(
                start='2023-01-01', 
                periods=len(ts_data), 
                freq='H'
            )
        
        # Format for Prophet
        ts_data = ts_data.rename(columns={'timestamp': 'ds', target_col: 'y'})
        ts_data = ts_data[['ds', 'y']].dropna()
        
        return ts_data
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for ML model.
        
        Args:
            data: Raw traffic data
            
        Returns:
            Processed feature array
        """
        features = data.copy()
        
        # Ensure timestamp column
        if 'timestamp' not in features.columns:
            features['timestamp'] = pd.date_range(
                start='2023-01-01', 
                periods=len(features), 
                freq='H'
            )
        
        # Extract time features
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        features['month'] = pd.to_datetime(features['timestamp']).dt.month
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # Add lag features
        if 'traffic_volume' in features.columns:
            features['traffic_lag_1h'] = features['traffic_volume'].shift(1)
            features['traffic_lag_24h'] = features['traffic_volume'].shift(24)
            features['traffic_rolling_mean_24h'] = features['traffic_volume'].rolling(24).mean()
        
        # Handle missing values
        features = features.fillna(features.median())
        
        # Select relevant columns
        feature_cols = [col for col in self.feature_columns if col in features.columns]
        return features[feature_cols].values
    
    def get_recommendations(self, predictions: np.ndarray, 
                          confidence_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate traffic-specific recommendations.
        
        Args:
            predictions: Traffic predictions
            confidence_threshold: Confidence threshold
            
        Returns:
            List of traffic management recommendations
        """
        recommendations = []
        
        # Analyze prediction patterns
        mean_traffic = np.mean(predictions)
        max_traffic = np.max(predictions)
        traffic_std = np.std(predictions)
        
        # Peak traffic detection
        peak_threshold = mean_traffic + 2 * traffic_std
        peak_indices = np.where(predictions > peak_threshold)[0]
        
        for idx in peak_indices:
            recommendations.append({
                'index': idx,
                'type': 'traffic_forecast',
                'predicted_traffic': predictions[idx],
                'severity': 'high' if predictions[idx] > mean_traffic + 3 * traffic_std else 'medium',
                'actions': [
                    'Scale up network capacity',
                    'Activate additional servers',
                    'Implement traffic shaping',
                    'Notify operations team'
                ],
                'priority': 'high',
                'time_window': f"Hour {idx} to {idx + 1}"
            })
        
        # Capacity planning recommendations
        if max_traffic > mean_traffic * 1.5:
            recommendations.append({
                'index': -1,
                'type': 'capacity_planning',
                'predicted_traffic': max_traffic,
                'severity': 'high',
                'actions': [
                    'Plan capacity expansion',
                    'Review infrastructure scaling',
                    'Consider load balancing optimization',
                    'Schedule maintenance during low traffic'
                ],
                'priority': 'medium',
                'time_window': 'Next 24 hours'
            })
        
        return recommendations
    
    def forecast_seasonal_patterns(self, data: pd.DataFrame, 
                                 forecast_days: int = 7) -> Dict[str, Any]:
        """
        Forecast seasonal traffic patterns.
        
        Args:
            data: Historical traffic data
            forecast_days: Days to forecast
            
        Returns:
            Seasonal forecast results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        # Generate hourly forecasts for the specified days
        forecast_hours = forecast_days * 24
        predictions = self.predict(data, forecast_hours)
        
        # Analyze patterns
        daily_patterns = []
        for day in range(forecast_days):
            day_start = day * 24
            day_end = (day + 1) * 24
            day_traffic = predictions[day_start:day_end]
            
            daily_patterns.append({
                'day': day + 1,
                'peak_hour': np.argmax(day_traffic),
                'peak_traffic': np.max(day_traffic),
                'avg_traffic': np.mean(day_traffic),
                'min_traffic': np.min(day_traffic)
            })
        
        return {
            'forecast_days': forecast_days,
            'predictions': predictions.tolist(),
            'daily_patterns': daily_patterns,
            'overall_peak': np.max(predictions),
            'overall_avg': np.mean(predictions),
            'recommendations': self.get_recommendations(predictions)
        }
