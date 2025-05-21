import pandas as pd
import numpy as np
import pytest
from src.data_preprocessing import load_data, preprocess_data

def test_load_data():
    data = load_data('data/mancity23-24.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert 'Date' in data.columns
    assert 'Result' in data.columns

def test_preprocess_data():
    data = load_data('data/mancity23-24.csv')
    processed_data = preprocess_data(data)
    
    # Check for missing values
    assert processed_data.isnull().sum().sum() == 0
    
    # Check data types
    assert processed_data['poss'].dtype == np.int64
    assert processed_data['passes'].dtype == np.int64
    
    # Check target column
    assert 'target' in processed_data.columns
    assert processed_data['target'].dtype == np.int64

    # Check if the target is binary
    assert set(processed_data['target'].unique()) == {0, 1}