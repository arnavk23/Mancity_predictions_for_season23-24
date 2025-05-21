import pandas as pd

def load_data(filepath):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath, index_col=0)
    return data

def preprocess_data(df):
    """Preprocess the dataset by handling missing values and converting data types."""
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    
    # Handle missing values
    df = df.dropna()  # or use df.fillna(method='ffill') based on requirements
    
    # Convert relevant columns to appropriate data types
    df['poss'] = df['poss'].astype(int)
    df['passes'] = df['passes'].astype(int)
    
    return df