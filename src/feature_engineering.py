"""
Feature engineering module for creating new features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class FeatureEngineer:
    """Handles feature creation and transformation."""
    
    def __init__(self):
        """Initialize feature engineer with scaler."""
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.is_fitted = True
        print("Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled
        
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional interaction features
        """
        # This is a placeholder for domain-specific interaction features
        # In a real scenario, you would create meaningful interactions
        # based on agricultural domain knowledge
        
        print("Interaction features creation skipped (no domain-specific features defined)")
        return data
        
    def engineer_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         scale: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full feature engineering pipeline.
        
        Args:
            X_train: Training features
            X_test: Test features
            scale: Whether to apply feature scaling
            
        Returns:
            Tuple of transformed (X_train, X_test)
        """
        print("Starting feature engineering...")
        
        # Note: For tree-based models like Random Forest, scaling is not required
        # and can sometimes reduce interpretability. We'll make it optional.
        if scale:
            X_train, X_test = self.scale_features(X_train, X_test)
        else:
            print("Feature scaling skipped (not required for tree-based models)")
        
        print("Feature engineering complete!")
        
        return X_train, X_test
