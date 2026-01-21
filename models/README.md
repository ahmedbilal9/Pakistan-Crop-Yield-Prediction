# Trained Models

## Overview
This directory contains trained machine learning models for crop yield prediction.

## Models

The pipeline trains the following regression models:

### 1. Linear Regression
- **File**: `linear_regression.pkl`
- **Description**: Simple baseline model using linear relationships
- **Use Case**: Understanding linear relationships between features and yield

### 2. Decision Tree Regressor
- **File**: `decision_tree.pkl`
- **Description**: Tree-based model that captures non-linear relationships
- **Hyperparameters**: 
  - max_depth: Configurable in config.yaml
  - random_state: 42

### 3. Random Forest Regressor
- **File**: `random_forest.pkl`
- **Description**: Ensemble of decision trees, typically performs well
- **Hyperparameters**:
  - n_estimators: 100 (default)
  - max_depth: None (grows to maximum depth)
  - random_state: 42
  - n_jobs: -1 (uses all CPU cores)

### 4. Gradient Boosting Regressor
- **File**: `gradient_boosting.pkl`
- **Description**: Sequential ensemble method
- **Hyperparameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 3
  - random_state: 42

### 5. XGBoost Regressor
- **File**: `xgboost.pkl`
- **Description**: Optimized gradient boosting implementation
- **Hyperparameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 5
  - random_state: 42
  - n_jobs: -1

## Loading Models

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer()
model = trainer.load_model('models/random_forest.pkl')

# Make predictions
predictions = model.predict(X_test)
```

## Model Performance

Model performance metrics are saved in `results/metrics/model_comparison.json`

Expected performance hierarchy:
1. XGBoost / Random Forest (Best)
2. Gradient Boosting
3. Decision Tree
4. Linear Regression (Baseline)

## Model Selection Guidelines

- **Random Forest**: Good default choice, robust to overfitting
- **XGBoost**: Often achieves best performance but may need tuning
- **Gradient Boosting**: Good balance between performance and training time
- **Decision Tree**: Fast training, interpretable, but may overfit
- **Linear Regression**: Fast, interpretable, but limited to linear relationships

## Storage
- Models are saved using `joblib` for efficient serialization
- File sizes vary based on model complexity (Random Forest is typically largest)

## Notes
- Model files are excluded from version control (.gitignore)
- Train models using `python main.py` or individual modules
- Models are versioned by their hyperparameters in config.yaml
