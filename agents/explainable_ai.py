#!/usr/bin/env python3
"""
Explainable AI (XAI) Integration for Enhanced Telecom AI System
Implements SHAP and LIME for model explainability
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# LIME imports
try:
    from lime import lime_tabular
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

class ExplainableAIAnalyzer:
    """Main class for explainable AI analysis"""
    
    def __init__(self, model, feature_names: List[str], config: Dict[str, Any] = None):
        self.model = model
        self.feature_names = feature_names
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Explanation methods
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Explanation cache
        self.explanation_cache = {}
        
        # Initialize explainers
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            if SHAP_AVAILABLE:
                # Initialize SHAP explainer based on model type
                if hasattr(self.model, 'predict_proba'):
                    # For models with probability output
                    self.shap_explainer = shap.Explainer(self.model)
                else:
                    # For models with direct prediction
                    self.shap_explainer = shap.Explainer(self.model)
                
                self.logger.info("SHAP explainer initialized")
            
            if LIME_AVAILABLE:
                # Initialize LIME explainer
                self.lime_explainer = LimeTabularExplainer(
                    np.zeros((1, len(self.feature_names))),  # Dummy data
                    feature_names=self.feature_names,
                    mode='regression' if self.config.get('task_type') == 'regression' else 'classification',
                    discretize_continuous=True
                )
                self.logger.info("LIME explainer initialized")
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize explainers: {e}")
    
    def explain_prediction(self, instance: np.ndarray, method: str = 'both') -> Dict[str, Any]:
        """Explain a single prediction"""
        instance_id = hash(instance.tobytes())
        
        # Check cache first
        if instance_id in self.explanation_cache:
            return self.explanation_cache[instance_id]
        
        explanation = {
            'instance': instance.tolist(),
            'prediction': self._get_prediction(instance),
            'timestamp': datetime.now().isoformat(),
            'methods': {}
        }
        
        # SHAP explanation
        if method in ['shap', 'both'] and self.shap_explainer is not None:
            try:
                shap_explanation = self._get_shap_explanation(instance)
                explanation['methods']['shap'] = shap_explanation
            except Exception as e:
                self.logger.warning(f"SHAP explanation failed: {e}")
                explanation['methods']['shap'] = {'error': str(e)}
        
        # LIME explanation
        if method in ['lime', 'both'] and self.lime_explainer is not None:
            try:
                lime_explanation = self._get_lime_explanation(instance)
                explanation['methods']['lime'] = lime_explanation
            except Exception as e:
                self.logger.warning(f"LIME explanation failed: {e}")
                explanation['methods']['lime'] = {'error': str(e)}
        
        # Cache explanation
        self.explanation_cache[instance_id] = explanation
        
        return explanation
    
    def _get_prediction(self, instance: np.ndarray) -> Union[float, int]:
        """Get model prediction for instance"""
        if hasattr(self.model, 'predict'):
            return self.model.predict(instance.reshape(1, -1))[0]
        elif hasattr(self.model, 'forward'):
            with torch.no_grad():
                tensor_instance = torch.FloatTensor(instance).unsqueeze(0)
                return self.model(tensor_instance).item()
        else:
            return 0.0
    
    def _get_shap_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """Get SHAP explanation for instance"""
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer(instance.reshape(1, -1))
            
            # Extract feature importance
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0] if len(shap_values.values.shape) > 1 else shap_values.values
            else:
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            # Create feature importance mapping
            feature_importance = {}
            for i, (feature, value) in enumerate(zip(self.feature_names, values)):
                feature_importance[feature] = {
                    'importance': float(value),
                    'abs_importance': float(abs(value)),
                    'rank': i + 1
                }
            
            # Sort by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1]['abs_importance'],
                reverse=True
            )
            
            return {
                'shap_values': values.tolist() if hasattr(values, 'tolist') else list(values),
                'feature_importance': feature_importance,
                'top_features': [{'feature': f, 'importance': imp} for f, imp in sorted_features[:5]],
                'explanation_text': self._generate_shap_explanation_text(sorted_features[:3])
            }
            
        except Exception as e:
            return {'error': f"SHAP explanation failed: {str(e)}"}
    
    def _get_lime_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """Get LIME explanation for instance"""
        try:
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                instance,
                self.model.predict if hasattr(self.model, 'predict') else lambda x: self._get_prediction(x),
                num_features=len(self.feature_names)
            )
            
            # Extract feature importance
            feature_importance = {}
            for i, (feature, weight) in enumerate(explanation.as_list()):
                feature_importance[feature] = {
                    'importance': float(weight),
                    'abs_importance': float(abs(weight)),
                    'rank': i + 1
                }
            
            return {
                'feature_importance': feature_importance,
                'top_features': explanation.as_list()[:5],
                'explanation_text': self._generate_lime_explanation_text(explanation.as_list()[:3])
            }
            
        except Exception as e:
            return {'error': f"LIME explanation failed: {str(e)}"}
    
    def _generate_shap_explanation_text(self, top_features: List[Tuple[str, Dict]]) -> str:
        """Generate human-readable explanation from SHAP"""
        if not top_features:
            return "No significant features identified."
        
        explanation_parts = []
        for feature, importance in top_features:
            direction = "increases" if importance['importance'] > 0 else "decreases"
            explanation_parts.append(
                f"Feature '{feature}' {direction} the prediction by {abs(importance['importance']):.3f}"
            )
        
        return "The prediction is influenced by: " + "; ".join(explanation_parts) + "."
    
    def _generate_lime_explanation_text(self, top_features: List[Tuple[str, float]]) -> str:
        """Generate human-readable explanation from LIME"""
        if not top_features:
            return "No significant features identified."
        
        explanation_parts = []
        for feature, weight in top_features:
            direction = "supports" if weight > 0 else "opposes"
            explanation_parts.append(
                f"'{feature}' {direction} the prediction (weight: {weight:.3f})"
            )
        
        return "Local explanation: " + "; ".join(explanation_parts) + "."
    
    def explain_batch(self, instances: np.ndarray, method: str = 'shap') -> Dict[str, Any]:
        """Explain a batch of predictions"""
        explanations = []
        
        for i, instance in enumerate(instances):
            explanation = self.explain_prediction(instance, method)
            explanations.append(explanation)
        
        # Aggregate explanations
        aggregated = self._aggregate_explanations(explanations)
        
        return {
            'batch_size': len(instances),
            'method': method,
            'individual_explanations': explanations,
            'aggregated_insights': aggregated,
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_explanations(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple explanations"""
        if not explanations:
            return {}
        
        # Collect all feature importances
        all_features = set()
        feature_weights = {}
        
        for explanation in explanations:
            if 'methods' in explanation:
                for method, method_data in explanation['methods'].items():
                    if 'feature_importance' in method_data:
                        for feature, importance in method_data['feature_importance'].items():
                            all_features.add(feature)
                            if feature not in feature_weights:
                                feature_weights[feature] = []
                            feature_weights[feature].append(importance['importance'])
        
        # Calculate average importance for each feature
        avg_importance = {}
        for feature in all_features:
            if feature in feature_weights:
                weights = feature_weights[feature]
                avg_importance[feature] = {
                    'mean_importance': np.mean(weights),
                    'std_importance': np.std(weights),
                    'count': len(weights)
                }
        
        # Sort by mean importance
        sorted_features = sorted(
            avg_importance.items(),
            key=lambda x: abs(x[1]['mean_importance']),
            reverse=True
        )
        
        return {
            'top_features': sorted_features[:10],
            'feature_statistics': avg_importance,
            'total_features': len(all_features)
        }
    
    def get_model_insights(self, training_data: np.ndarray) -> Dict[str, Any]:
        """Get global model insights"""
        try:
            if self.shap_explainer is None:
                return {'error': 'SHAP explainer not available'}
            
            # Calculate SHAP values for training data
            shap_values = self.shap_explainer(training_data)
            
            # Global feature importance
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.mean(np.abs(values), axis=0)
            
            # Create feature importance ranking
            feature_importance = {}
            for i, (feature, importance) in enumerate(zip(self.feature_names, mean_shap_values)):
                feature_importance[feature] = {
                    'importance': float(importance),
                    'rank': i + 1
                }
            
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            )
            
            return {
                'global_feature_importance': feature_importance,
                'top_global_features': sorted_features[:10],
                'model_complexity': {
                    'num_features': len(self.feature_names),
                    'num_samples': len(training_data),
                    'feature_diversity': float(np.std(mean_shap_values))
                }
            }
            
        except Exception as e:
            return {'error': f"Model insights failed: {str(e)}"}

class ExplainableAgent:
    """Base class for explainable AI agents"""
    
    def __init__(self, agent_type: str, model, feature_names: List[str], config: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.model = model
        self.feature_names = feature_names
        self.config = config or {}
        self.logger = logging.getLogger(f"explainable_{agent_type}")
        
        # Initialize explainable AI analyzer
        self.xai_analyzer = ExplainableAIAnalyzer(model, feature_names, config)
        
        # Agent-specific explanation templates
        self.explanation_templates = self._get_explanation_templates()
    
    def _get_explanation_templates(self) -> Dict[str, str]:
        """Get agent-specific explanation templates"""
        templates = {
            'qos_anomaly': {
                'anomaly_detected': "Anomaly detected due to: {features}",
                'normal_behavior': "Normal behavior maintained by: {features}",
                'recommendation': "Recommended action: {action} based on {reasoning}"
            },
            'failure_prediction': {
                'failure_risk': "High failure risk due to: {features}",
                'low_risk': "Low failure risk maintained by: {features}",
                'recommendation': "Preventive action: {action} to reduce risk"
            },
            'traffic_forecast': {
                'high_traffic': "High traffic predicted due to: {features}",
                'low_traffic': "Low traffic expected due to: {features}",
                'recommendation': "Capacity planning: {action} for predicted load"
            },
            'energy_optimize': {
                'high_consumption': "High energy consumption due to: {features}",
                'efficient': "Energy efficient operation due to: {features}",
                'recommendation': "Energy optimization: {action} to reduce consumption"
            },
            'security_detection': {
                'threat_detected': "Security threat detected due to: {features}",
                'secure': "System secure due to: {features}",
                'recommendation': "Security action: {action} to mitigate threat"
            },
            'data_quality': {
                'quality_issue': "Data quality issue due to: {features}",
                'good_quality': "Good data quality maintained by: {features}",
                'recommendation': "Data correction: {action} to improve quality"
            }
        }
        
        return templates.get(self.agent_type, templates['qos_anomaly'])
    
    def explain_decision(self, instance: np.ndarray, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Explain agent decision with context"""
        context = context or {}
        
        # Get basic explanation
        explanation = self.xai_analyzer.explain_prediction(instance)
        
        # Add agent-specific context
        explanation['agent_type'] = self.agent_type
        explanation['context'] = context
        
        # Generate human-readable explanation
        explanation['human_readable'] = self._generate_human_explanation(explanation, context)
        
        # Add recommendations
        explanation['recommendations'] = self._generate_recommendations(explanation, context)
        
        return explanation
    
    def _generate_human_explanation(self, explanation: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate human-readable explanation"""
        prediction = explanation.get('prediction', 0)
        
        # Determine explanation type based on agent and prediction
        if self.agent_type == 'qos_anomaly':
            if prediction > 0.5:  # Anomaly detected
                template = self.explanation_templates['anomaly_detected']
            else:
                template = self.explanation_templates['normal_behavior']
        elif self.agent_type == 'failure_prediction':
            if prediction > 0.7:  # High failure risk
                template = self.explanation_templates['failure_risk']
            else:
                template = self.explanation_templates['low_risk']
        else:
            # Generic explanation
            template = "Model prediction: {prediction:.3f} based on feature analysis"
        
        # Extract top features for explanation
        top_features = []
        if 'methods' in explanation:
            for method, method_data in explanation['methods'].items():
                if 'top_features' in method_data:
                    top_features.extend([f['feature'] for f in method_data['top_features'][:3]])
        
        # Create feature list
        feature_list = ', '.join(set(top_features)[:3]) if top_features else 'multiple factors'
        
        # Format explanation
        return template.format(
            features=feature_list,
            prediction=prediction
        )
    
    def _generate_recommendations(self, explanation: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        prediction = explanation.get('prediction', 0)
        
        if self.agent_type == 'qos_anomaly':
            if prediction > 0.5:
                recommendations.extend([
                    "Investigate network congestion",
                    "Check for hardware issues",
                    "Consider traffic rerouting"
                ])
            else:
                recommendations.append("Continue monitoring current performance")
        
        elif self.agent_type == 'failure_prediction':
            if prediction > 0.7:
                recommendations.extend([
                    "Schedule preventive maintenance",
                    "Prepare backup resources",
                    "Monitor system health closely"
                ])
            else:
                recommendations.append("Continue regular monitoring")
        
        elif self.agent_type == 'traffic_forecast':
            if prediction > 0.8:
                recommendations.extend([
                    "Scale up network capacity",
                    "Prepare additional resources",
                    "Implement traffic management"
                ])
            else:
                recommendations.append("Maintain current capacity planning")
        
        return recommendations
    
    def explain_batch_decisions(self, instances: np.ndarray, contexts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Explain batch of decisions"""
        contexts = contexts or [{}] * len(instances)
        
        batch_explanations = []
        for instance, context in zip(instances, contexts):
            explanation = self.explain_decision(instance, context)
            batch_explanations.append(explanation)
        
        # Aggregate insights
        aggregated_insights = self._aggregate_batch_insights(batch_explanations)
        
        return {
            'batch_size': len(instances),
            'agent_type': self.agent_type,
            'individual_explanations': batch_explanations,
            'aggregated_insights': aggregated_insights,
            'timestamp': datetime.now().isoformat()
        }
    
    def _aggregate_batch_insights(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate insights from batch explanations"""
        if not explanations:
            return {}
        
        # Collect common features across explanations
        all_features = set()
        feature_counts = {}
        
        for explanation in explanations:
            if 'methods' in explanation:
                for method, method_data in explanation['methods'].items():
                    if 'top_features' in method_data:
                        for feature_info in method_data['top_features']:
                            feature = feature_info.get('feature', '')
                            if feature:
                                all_features.add(feature)
                                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Find most common features
        common_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Calculate prediction statistics
        predictions = [exp.get('prediction', 0) for exp in explanations]
        
        return {
            'common_features': common_features,
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            },
            'total_features_analyzed': len(all_features),
            'explanation_quality': 'high' if len(explanations) > 10 else 'medium'
        }

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test explainable AI
    print("Testing Explainable AI Integration...")
    
    # Create a simple model for testing
    class SimpleModel:
        def __init__(self):
            self.weights = np.random.random(5)
        
        def predict(self, X):
            return np.dot(X, self.weights)
    
    model = SimpleModel()
    feature_names = ['latency', 'throughput', 'jitter', 'packet_loss', 'signal_strength']
    
    # Test explainable AI analyzer
    xai_analyzer = ExplainableAIAnalyzer(model, feature_names)
    
    # Test single prediction explanation
    test_instance = np.random.random(5)
    explanation = xai_analyzer.explain_prediction(test_instance)
    print(f"Single prediction explanation: {explanation}")
    
    # Test explainable agent
    explainable_agent = ExplainableAgent('qos_anomaly', model, feature_names)
    decision_explanation = explainable_agent.explain_decision(test_instance)
    print(f"Agent decision explanation: {decision_explanation}")
    
    print("Explainable AI testing completed!")
