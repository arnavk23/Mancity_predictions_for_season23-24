import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_model(model_path):
    """Load the trained model from the specified path."""
    model = joblib.load(model_path)
    return model

def make_predictions(model, input_data):
    """Make predictions using the trained model."""
    predictions = model.predict(input_data)
    return predictions

def preprocess_input_data(data):
    """Preprocess the input data for predictions."""
    data["poss"] = data["poss"].astype("int")
    data["passes"] = data["passes"].astype("int")
    return data[["poss", "passes"]]

def predict(input_data, model_path):
    """Main function to load model and make predictions."""
    model = load_model(model_path)
    processed_data = preprocess_input_data(input_data)
    predictions = make_predictions(model, processed_data)
    return predictions