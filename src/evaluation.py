"""
Model evaluation module with multiple metrics.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import os
from typing import Dict, Any


class ModelEvaluator:
    """Handles model evaluation and comparison."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.results = {}
        
    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str = "model") -> Dict[str, float]:
        """
        Calculate evaluation metrics for a single model.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model for reference
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'r2_score': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
        
        self.results[model_name] = metrics
        
        return metrics
        
    def compare_models(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models on test data.
        
        Args:
            models_dict: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame containing comparison metrics
        """
        print("Evaluating models...")
        
        results_list = []
        
        for model_name, model in models_dict.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.evaluate_model(y_test, y_pred, model_name)
            results_list.append(metrics)
            
            # Print results
            print(f"\n{model_name}:")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
        
        # Create DataFrame for easy comparison
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('r2_score', ascending=False)
        
        print("\n" + "="*50)
        print("Model Comparison Summary:")
        print("="*50)
        print(results_df.to_string(index=False))
        
        return results_df
        
    def save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """
        Save evaluation metrics to file.
        
        Args:
            metrics: Dictionary or DataFrame of metrics
            filepath: Path to save metrics
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if isinstance(metrics, pd.DataFrame):
            # Save as CSV
            csv_path = os.path.splitext(filepath)[0] + '.csv'
            metrics.to_csv(csv_path, index=False)
            # Also save as JSON
            metrics.to_json(filepath, orient='records', indent=2)
        else:
            # Save dictionary as JSON
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
        
    def get_best_model(self, metric: str = 'r2_score') -> str:
        """
        Get the name of the best performing model.
        
        Args:
            metric: Metric to use for comparison (default: r2_score)
            
        Returns:
            Name of the best model
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        if metric in ['mse', 'mae', 'rmse']:
            # Lower is better for error metrics
            best_model = min(self.results.items(), key=lambda x: x[1][metric])[0]
        else:
            # Higher is better for R² score
            best_model = max(self.results.items(), key=lambda x: x[1][metric])[0]
        
        return best_model
        
    def print_summary(self) -> None:
        """Print a summary of all evaluated models."""
        if not self.results:
            print("No models have been evaluated yet")
            return
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for model_name, metrics in sorted(
            self.results.items(), 
            key=lambda x: x[1]['r2_score'], 
            reverse=True
        ):
            print(f"\n{model_name}:")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  MAE:      {metrics['mae']:.4f}")
            print(f"  RMSE:     {metrics['rmse']:.4f}")
        
        best_model = self.get_best_model()
        print(f"\n{'='*60}")
        print(f"Best Model (by R² score): {best_model}")
        print(f"R² Score: {self.results[best_model]['r2_score']:.4f}")
        print("="*60)
