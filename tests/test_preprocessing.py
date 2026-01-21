"""
Unit tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
import tempfile
import os


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample crop yield dataset for testing."""
        data = {
            'Year': [2020, 2021, 2022, 2020, 2021],
            'Item': ['Wheat', 'Wheat', 'Wheat', 'Rice', 'Rice'],
            'Element': ['Yield', 'Yield', 'Yield', 'Yield', 'Yield'],
            'Value': [3000, 3100, 3200, 2500, 2600],
            'Unit': ['kg/ha', 'kg/ha', 'kg/ha', 'kg/ha', 'kg/ha'],
            'Flag': ['A', 'A', 'A', 'A', 'A'],
            'Domain': ['Crops', 'Crops', 'Crops', 'Crops', 'Crops'],
            'Domain Code': ['QC', 'QC', 'QC', 'QC', 'QC'],
            'Area': ['Pakistan', 'Pakistan', 'Pakistan', 'Pakistan', 'Pakistan'],
            'Area Code (M49)': ['586', '586', '586', '586', '586'],
            'Note': [None, None, None, None, None],
            'Year Code': ['2020', '2021', '2022', '2020', '2021'],
            'Element Code': ['5419', '5419', '5419', '5419', '5419'],
            'Item Code (CPC)': ['0111', '0111', '0111', '0112', '0112'],
            'Flag Description': ['Aggregate', 'Aggregate', 'Aggregate', 'Aggregate', 'Aggregate']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create a temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        yield temp_file
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    def test_init(self, temp_csv_file):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(temp_csv_file)
        assert preprocessor.filepath == temp_csv_file
        assert preprocessor.data is None
        assert preprocessor.X is None
        assert preprocessor.y is None
    
    def test_load_data(self, temp_csv_file):
        """Test data loading."""
        preprocessor = DataPreprocessor(temp_csv_file)
        data = preprocessor.load_data()
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        assert 'Value' in data.columns
    
    def test_filter_irrelevant_columns(self, temp_csv_file):
        """Test filtering of irrelevant columns."""
        preprocessor = DataPreprocessor(temp_csv_file)
        preprocessor.load_data()
        preprocessor.filter_irrelevant_columns()
        
        # Check that irrelevant columns are removed
        irrelevant_cols = ['Domain Code', 'Domain', 'Area', 'Area Code (M49)', 
                          'Note', 'Year Code', 'Element Code', 'Item Code (CPC)', 
                          'Flag Description']
        for col in irrelevant_cols:
            assert col not in preprocessor.data.columns
    
    def test_filter_yield_data(self, temp_csv_file):
        """Test filtering to yield data only."""
        preprocessor = DataPreprocessor(temp_csv_file)
        preprocessor.load_data()
        preprocessor.filter_irrelevant_columns()
        preprocessor.filter_yield_data()
        
        # Element column should be removed after filtering
        assert 'Element' not in preprocessor.data.columns
        # All remaining rows should have been yield data
        assert len(preprocessor.data) == 5
    
    def test_handle_missing_values(self, temp_csv_file):
        """Test handling of missing values."""
        preprocessor = DataPreprocessor(temp_csv_file)
        preprocessor.load_data()
        preprocessor.handle_missing_values()
        
        # No missing values should remain
        assert preprocessor.data.isnull().sum().sum() == 0
    
    def test_encode_categorical(self, temp_csv_file):
        """Test categorical encoding."""
        preprocessor = DataPreprocessor(temp_csv_file)
        preprocessor.load_data()
        preprocessor.filter_irrelevant_columns()
        preprocessor.filter_yield_data()
        
        initial_cols = len(preprocessor.data.columns)
        preprocessor.encode_categorical()
        
        # Number of columns should increase due to one-hot encoding
        assert len(preprocessor.data.columns) >= initial_cols
    
    def test_split_data(self, temp_csv_file):
        """Test data splitting."""
        preprocessor = DataPreprocessor(temp_csv_file)
        preprocessor.load_data()
        preprocessor.filter_irrelevant_columns()
        preprocessor.filter_yield_data()
        preprocessor.encode_categorical()
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.2, random_state=42)
        
        # Check shapes
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check that Value column is not in features
        assert 'Value' not in X_train.columns
        assert 'Value' not in X_test.columns
    
    def test_preprocess_pipeline(self, temp_csv_file):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor(temp_csv_file)
        X_train, X_test, y_train, y_test = preprocessor.preprocess(test_size=0.2, random_state=42)
        
        # Check that all outputs are valid
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        # Check types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check that features and target are separated
        assert 'Value' not in X_train.columns
        assert len(X_train) + len(X_test) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
