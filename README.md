# 🌾 Pakistan Crop Yield Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-professional-brightgreen)]()

## 📋 Project Overview
A comprehensive machine learning project to predict crop yields in Pakistan using historical agricultural data from FAOSTAT. This project demonstrates end-to-end ML pipeline development with modular, production-ready code following software engineering best practices.

The project transforms raw agricultural data into actionable insights through data preprocessing, feature engineering, model training, and evaluation. Multiple regression algorithms are compared to identify the best performing model for crop yield prediction.

## 🎯 Objectives
- Analyze agricultural datasets to extract meaningful insights about crop production in Pakistan
- Build robust predictive models for crop yield estimation
- Compare multiple ML algorithms (Linear Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost)
- Provide interpretable, production-ready code that can be integrated into larger systems
- Demonstrate professional software engineering practices with modular architecture

## 🏗️ Project Structure
```
Pakistan-Crop-Yield-Prediction/
├── data/                           # Data directory
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Cleaned/processed data
│   └── README.md                   # Data documentation
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading and preprocessing
│   ├── feature_engineering.py      # Feature creation and transformation
│   ├── model_training.py           # Model training and saving
│   ├── evaluation.py               # Model evaluation and comparison
│   └── visualization.py            # Plotting and visualizations
├── models/                         # Trained model files
│   └── README.md                   # Models documentation
├── results/                        # Outputs and visualizations
│   ├── figures/                    # Generated plots
│   └── metrics/                    # Performance metrics
├── notebooks/                      # Jupyter notebooks
│   └── pakistan-crop-yield-prediction.ipynb
├── config/                         # Configuration files
│   └── config.yaml                 # Hyperparameters and paths
├── tests/                          # Unit tests
│   └── test_preprocessing.py
├── main.py                         # Main execution script
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ahmedbilal9/Pakistan-Crop-Yield-Prediction.git
cd Pakistan-Crop-Yield-Prediction
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package (optional)**
```bash
pip install -e .
```

### Quick Start

**Run the complete pipeline:**
```bash
python main.py
```

**Note**: The pipeline requires the dataset to be placed at `data/raw/crop_yield_data.csv`. If you don't have the dataset, you can still explore the modular code structure and use the Jupyter notebook for demonstration.

### Usage Examples

#### Using the Complete Pipeline
```bash
python main.py
```

#### Using Individual Modules
```python
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

# Load and preprocess data
preprocessor = DataPreprocessor('data/raw/crop_yield_data.csv')
X_train, X_test, y_train, y_test = preprocessor.preprocess()

# Train a specific model
trainer = ModelTrainer()
model = trainer.train_model('random_forest', X_train, y_train)

# Evaluate the model
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(y_test, model.predict(X_test), 'random_forest')
print(metrics)
```

#### Exploring with Jupyter Notebook
```bash
jupyter notebook notebooks/pakistan-crop-yield-prediction.ipynb
```

## 📊 Dataset

### Source
- **Provider**: FAOSTAT (Food and Agriculture Organization Corporate Statistical Database)
- **Coverage**: Pakistan agricultural records from 1961 to 2023
- **Format**: CSV

### Features
- **Year**: Year of agricultural record
- **Item**: Type of crop (Wheat, Rice, Cotton, etc.)
- **Value**: Yield (output per hectare)
- **Unit**: Measurement unit
- **Flag**: Data quality indicator

### Preprocessing Pipeline
1. **Column Filtering**: Remove redundant and non-unique columns
2. **Element Filtering**: Filter to yield-specific records
3. **Missing Values**: Handle incomplete records
4. **Categorical Encoding**: One-hot encode categorical variables
5. **Train-Test Split**: 80/20 split with stratification

## 🤖 Models Implemented

| Model | Description | Key Characteristics |
|-------|-------------|---------------------|
| **Linear Regression** | Baseline model | Fast, interpretable, assumes linear relationships |
| **Decision Tree** | Single tree regressor | Captures non-linearity, interpretable, may overfit |
| **Random Forest** | Ensemble of trees | Robust, handles non-linearity well, reduced overfitting |
| **Gradient Boosting** | Sequential ensemble | Strong performance, slower training |
| **XGBoost** | Optimized gradient boosting | State-of-the-art performance, efficient |

## 📈 Results

The models are evaluated using multiple regression metrics:
- **R² Score**: Coefficient of determination (higher is better)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)

**Expected Performance Hierarchy:**
1. Random Forest / XGBoost (Best performers)
2. Gradient Boosting
3. Decision Tree
4. Linear Regression (Baseline)

Results are automatically saved to:
- `results/metrics/model_comparison.json` - Detailed metrics
- `results/figures/` - Visualization plots

## 🔬 Methodology

### 1. Data Preprocessing
- Load raw agricultural data
- Filter irrelevant columns
- Handle missing values
- Encode categorical variables
- Split into training and testing sets

### 2. Feature Engineering
- Feature scaling (optional, for linear models)
- Domain-specific feature creation (extensible)

### 3. Model Training
- Train multiple regression algorithms
- Use cross-validation for hyperparameter tuning
- Save trained models for reuse

### 4. Model Evaluation
- Calculate performance metrics (R², MAE, RMSE)
- Compare models side-by-side
- Identify best performing model

### 5. Visualization
- Exploratory data analysis plots
- Model performance comparisons
- Prediction vs actual plots
- Feature importance analysis
- Residual analysis

## 🛠️ Development

### Code Quality Standards
- **PEP 8 Compliant**: Following Python style guidelines
- **Type Hints**: Enhanced code readability and IDE support
- **Docstrings**: Comprehensive documentation for all modules
- **Modular Design**: Separation of concerns with clear interfaces
- **Error Handling**: Robust exception handling

### Running Tests
```bash
pytest tests/ -v
```

### Project Configuration
Edit `config/config.yaml` to customize:
- Data paths
- Model hyperparameters
- Training parameters
- Output directories

## 📝 Future Improvements
- [ ] Incorporate weather and soil data for enhanced predictions
- [ ] Implement time-series forecasting using LSTM/ARIMA
- [ ] Add hyperparameter optimization (Grid Search, Bayesian Optimization)
- [ ] Deploy as REST API using Flask or FastAPI
- [ ] Create interactive dashboard using Streamlit or Dash
- [ ] Add support for multi-output prediction (multiple crops)
- [ ] Implement automated retraining pipeline
- [ ] Add data drift detection
- [ ] Create Docker containerization
- [ ] Add CI/CD pipeline

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## �� Author
**Ahmed Bilal**
- GitHub: [@ahmedbilal9](https://github.com/ahmedbilal9)

## 🙏 Acknowledgments
- FAOSTAT for providing comprehensive agricultural datasets
- The open-source ML community for excellent tools and libraries
- Contributors and users of this project

## 📚 References
- [FAOSTAT Database](http://www.fao.org/faostat/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**⭐ If you find this project helpful, please consider giving it a star!**
