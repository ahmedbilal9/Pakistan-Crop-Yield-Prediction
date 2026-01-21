"""
Main orchestration script to run the complete ML pipeline.

This script demonstrates how to use the modular components to:
1. Load and preprocess data
2. Engineer features
3. Train multiple models
4. Evaluate and compare models
5. Visualize results
"""
import os

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer


def main():
    """Execute complete ML pipeline."""
    print("="*60)
    print("Pakistan Crop Yield Prediction Pipeline")
    print("="*60)
    
    # Configuration
    DATA_PATH = 'data/raw/crop_yield_data.csv'
    MODELS_DIR = 'models/'
    RESULTS_DIR = 'results/'
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\n⚠️  Data file not found at: {DATA_PATH}")
        print("\nPlease ensure you have the dataset at the correct location.")
        print("Expected location: data/raw/crop_yield_data.csv")
        print("\nAlternatively, you can use the Jupyter notebook in notebooks/")
        print("to explore the analysis interactively.")
        return
    
    try:
        # 1. Load and preprocess data
        print("\n" + "="*60)
        print("STEP 1: Data Preprocessing")
        print("="*60)
        preprocessor = DataPreprocessor(DATA_PATH)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        
        # 2. Feature engineering
        print("\n" + "="*60)
        print("STEP 2: Feature Engineering")
        print("="*60)
        engineer = FeatureEngineer()
        X_train, X_test = engineer.engineer_features(X_train, X_test, scale=False)
        
        # 3. Train models
        print("\n" + "="*60)
        print("STEP 3: Model Training")
        print("="*60)
        trainer = ModelTrainer()
        trained_models = trainer.train_all_models(X_train, y_train)
        
        # Save models
        trainer.save_all_models(MODELS_DIR)
        
        # 4. Evaluate models
        print("\n" + "="*60)
        print("STEP 4: Model Evaluation")
        print("="*60)
        evaluator = ModelEvaluator()
        results_df = evaluator.compare_models(trained_models, X_test, y_test)
        
        # Save metrics
        os.makedirs(f'{RESULTS_DIR}/metrics', exist_ok=True)
        evaluator.save_metrics(results_df, f'{RESULTS_DIR}/metrics/model_comparison.json')
        
        # 5. Visualize results
        print("\n" + "="*60)
        print("STEP 5: Visualization")
        print("="*60)
        viz = Visualizer(f'{RESULTS_DIR}/figures/')
        
        # Plot model comparison
        viz.plot_model_comparison(results_df)
        
        # Plot predictions for best model
        best_model_name = evaluator.get_best_model()
        best_model = trained_models[best_model_name]
        y_pred = best_model.predict(X_test)
        
        viz.plot_predictions_vs_actual(y_test, y_pred, best_model_name)
        viz.plot_residuals(y_test, y_pred, best_model_name)
        
        # Plot feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            viz.plot_feature_importance(best_model, X_train.columns.tolist())
        
        # Final summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        evaluator.print_summary()
        
        print("\n📊 Results saved to:")
        print(f"  - Models: {MODELS_DIR}")
        print(f"  - Metrics: {RESULTS_DIR}/metrics/")
        print(f"  - Figures: {RESULTS_DIR}/figures/")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the data file exists at the correct location.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
