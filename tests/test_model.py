import pytest
import pandas as pd
from src.model import RandomForestClassifierModel

def test_model_training():
    # Load sample data for testing
    data = pd.read_csv('data/mancity23-24.csv', index_col=0)
    data["target"] = (data["Result"] == "W").astype("int")
    
    # Split the data into training and testing sets
    train = data[data["Date"] < '4-11-2023']
    test = data[data["Date"] > '4-11-2023']
    
    # Initialize the model
    model = RandomForestClassifierModel(n_estimators=50, min_samples_split=10, random_state=1)
    
    # Train the model
    model.fit(train[["poss", "passes"]], train["target"])
    
    # Make predictions
    preds = model.predict(test[["poss", "passes"]])
    
    # Check if predictions are of the correct shape
    assert preds.shape[0] == test.shape[0], "Predictions shape does not match test set shape"

def test_model_accuracy():
    # Load sample data for testing
    data = pd.read_csv('data/mancity23-24.csv', index_col=0)
    data["target"] = (data["Result"] == "W").astype("int")
    
    # Split the data into training and testing sets
    train = data[data["Date"] < '4-11-2023']
    test = data[data["Date"] > '4-11-2023']
    
    # Initialize the model
    model = RandomForestClassifierModel(n_estimators=50, min_samples_split=10, random_state=1)
    
    # Train the model
    model.fit(train[["poss", "passes"]], train["target"])
    
    # Make predictions
    preds = model.predict(test[["poss", "passes"]])
    
    # Calculate accuracy
    accuracy = (preds == test["target"]).mean()
    
    # Check if accuracy is above a certain threshold
    assert accuracy > 0.5, "Model accuracy is below the expected threshold"