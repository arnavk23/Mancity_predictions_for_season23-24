import pandas as pd
from src.features import extract_features

def test_extract_features():
    # Sample data for testing
    sample_data = pd.DataFrame({
        'poss': [55, 60, 45],
        'passes': [500, 600, 400],
        'Result': ['W', 'L', 'D']
    })
    
    # Expected output after feature extraction
    expected_output = pd.DataFrame({
        'poss': [55, 60, 45],
        'passes': [500, 600, 400],
        'target': [1, 0, 0]
    })
    
    # Run the feature extraction function
    output = extract_features(sample_data)
    
    # Assert that the output matches the expected output
    pd.testing.assert_frame_equal(output, expected_output)