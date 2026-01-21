"""
Data preprocessing module for Pakistan crop yield dataset.
Handles data loading, cleaning, and initial transformations.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self, filepath: str):
        """
        Initialize with data filepath.
        
        Args:
            filepath: Path to the CSV data file
        """
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Returns:
            Loaded pandas DataFrame
        """
        self.data = pd.read_csv(self.filepath)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
        
    def filter_irrelevant_columns(self) -> pd.DataFrame:
        """
        Remove irrelevant and redundant columns from dataset.
        
        Returns:
            DataFrame with irrelevant columns removed
        """
        # Columns to drop based on analysis
        cols_to_drop = [
            'Domain Code', 'Domain', 'Area', 'Area Code (M49)', 
            'Note', 'Year Code', 'Element Code', 'Item Code (CPC)', 
            'Flag Description'
        ]
        
        # Only drop columns that exist in the dataframe
        cols_to_drop = [col for col in cols_to_drop if col in self.data.columns]
        
        if cols_to_drop:
            self.data = self.data.drop(columns=cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} irrelevant columns")
        
        return self.data
    
    def filter_yield_data(self) -> pd.DataFrame:
        """
        Filter dataset to only include yield data.
        
        Returns:
            DataFrame filtered for yield element
        """
        if 'Element' in self.data.columns:
            self.data = self.data[self.data['Element'] == 'Yield']
            self.data = self.data.drop('Element', axis=1)
            print(f"Filtered to yield data: {self.data.shape[0]} rows")
        
        return self.data
        
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies.
        
        Returns:
            DataFrame with missing values handled
        """
        # Check for missing values
        missing_before = self.data.isnull().sum().sum()
        
        if missing_before > 0:
            # Drop rows with missing values in the target variable
            if 'Value' in self.data.columns:
                self.data = self.data.dropna(subset=['Value'])
            
            # Fill remaining missing values if any
            self.data = self.data.dropna()
            
            missing_after = self.data.isnull().sum().sum()
            print(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return self.data
        
    def encode_categorical(self) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Returns:
            DataFrame with encoded categorical variables
        """
        # Identify categorical columns (object dtype)
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if it's in the list
        if 'Value' in categorical_cols:
            categorical_cols.remove('Value')
        
        if categorical_cols:
            self.data = pd.get_dummies(self.data, columns=categorical_cols)
            print(f"Encoded {len(categorical_cols)} categorical columns")
        
        return self.data
        
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        if 'Value' not in self.data.columns:
            raise ValueError("Target column 'Value' not found in dataset")
        
        self.X = self.data.drop('Value', axis=1)
        self.y = self.data['Value']
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        return X_train, X_test, y_train, y_test
        
    def preprocess(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute full preprocessing pipeline.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Starting preprocessing pipeline...")
        
        self.load_data()
        self.filter_irrelevant_columns()
        self.filter_yield_data()
        self.handle_missing_values()
        self.encode_categorical()
        X_train, X_test, y_train, y_test = self.split_data(test_size, random_state)
        
        print("Preprocessing complete!")
        
        return X_train, X_test, y_train, y_test
