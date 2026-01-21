"""
Model training module supporting multiple ML algorithms.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os
from typing import Dict, Any, Optional


class ModelTrainer:
    """Handles model training and saving."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model trainer with default models.
        
        Args:
            config: Optional configuration dictionary for model hyperparameters
        """
        self.config = config or {}
        
        # Initialize models with default or config parameters
        rf_params = self.config.get('random_forest', {})
        gb_params = self.config.get('gradient_boosting', {})
        dt_params = self.config.get('decision_tree', {})
        
        self.models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(
                random_state=dt_params.get('random_state', 42),
                max_depth=dt_params.get('max_depth', None)
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', None),
                random_state=rf_params.get('random_state', 42),
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=gb_params.get('n_estimators', 100),
                learning_rate=gb_params.get('learning_rate', 0.1),
                max_depth=gb_params.get('max_depth', 3),
                random_state=gb_params.get('random_state', 42)
            )
        }
        
        # Try to import and add XGBoost if available
        try:
            import xgboost as xgb
            xgb_params = self.config.get('xgboost', {})
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=xgb_params.get('n_estimators', 100),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                max_depth=xgb_params.get('max_depth', 5),
                random_state=xgb_params.get('random_state', 42),
                n_jobs=-1
            )
        except ImportError:
            print("XGBoost not available. Skipping XGBoost model.")
        
        self.trained_models = {}
        
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        print(f"{model_name} training complete!")
        
        return model
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        print(f"Training {len(self.models)} models...")
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
        
        print("All models trained successfully!")
        return self.trained_models
        
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model object
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
    def save_all_models(self, output_dir: str = 'models/') -> None:
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = os.path.join(output_dir, f"{model_name}.pkl")
            self.save_model(model, filepath)
        
        print(f"All models saved to {output_dir}")
        
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
