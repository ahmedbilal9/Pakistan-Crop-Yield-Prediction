"""
Visualization module for EDA and model results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, List


class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, output_dir: str = 'results/figures/'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
    def plot_distribution(self, data: pd.DataFrame, column: str, 
                         save: bool = True) -> None:
        """
        Plot distribution of a variable.
        
        Args:
            data: DataFrame containing the data
            column: Column name to plot
            save: Whether to save the figure
        """
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        plt.subplot(1, 2, 1)
        data[column].hist(bins=30, edgecolor='black')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {column}')
        
        # Plot box plot
        plt.subplot(1, 2, 2)
        data.boxplot(column=[column])
        plt.ylabel(column)
        plt.title(f'Box Plot of {column}')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'distribution_{column}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plot to {filepath}")
        
        plt.close()
        
    def plot_correlation_heatmap(self, data: pd.DataFrame, 
                                 max_features: int = 20,
                                 save: bool = True) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            data: DataFrame containing the data
            max_features: Maximum number of features to display
            save: Whether to save the figure
        """
        # Select numerical columns only
        numerical_data = data.select_dtypes(include=[np.number])
        
        # If too many features, select top correlated with target
        if len(numerical_data.columns) > max_features:
            print(f"Limiting to top {max_features} features for readability")
            numerical_data = numerical_data.iloc[:, :max_features]
        
        # Calculate correlation matrix
        corr = numerical_data.corr()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'correlation_heatmap.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved correlation heatmap to {filepath}")
        
        plt.close()
        
    def plot_predictions_vs_actual(self, y_true: pd.Series, y_pred: np.ndarray,
                                   model_name: str = "Model",
                                   save: bool = True) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
            save: Whether to save the figure
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Predicted vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'predictions_{model_name.replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved predictions plot to {filepath}")
        
        plt.close()
        
    def plot_feature_importance(self, model, feature_names: List[str],
                               top_n: int = 20,
                               save: bool = True) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            top_n: Number of top features to display
            save: Whether to save the figure
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        feature_importance_df = feature_importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {filepath}")
        
        plt.close()
        
    def plot_model_comparison(self, metrics_df: pd.DataFrame,
                             save: bool = True) -> None:
        """
        Plot comparison of model performances.
        
        Args:
            metrics_df: DataFrame containing model metrics
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² Score comparison
        axes[0].bar(metrics_df['model_name'], metrics_df['r2_score'], color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² Score Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # MAE comparison
        axes[1].bar(metrics_df['model_name'], metrics_df['mae'], color='lightcoral', edgecolor='black')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # RMSE comparison
        axes[2].bar(metrics_df['model_name'], metrics_df['rmse'], color='lightgreen', edgecolor='black')
        axes[2].set_xlabel('Model')
        axes[2].set_ylabel('RMSE')
        axes[2].set_title('RMSE Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved model comparison plot to {filepath}")
        
        plt.close()
        
    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray,
                      model_name: str = "Model",
                      save: bool = True) -> None:
        """
        Plot residuals to check for patterns.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model
            save: Whether to save the figure
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name}: Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f'residuals_{model_name.replace(" ", "_")}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved residuals plot to {filepath}")
        
        plt.close()
