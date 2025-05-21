import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

class MancityModel:
    def __init__(self, n_estimators=50, min_samples_split=10, random_state=1):
        self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)

    def train(self, data, predictors, target):
        self.model.fit(data[predictors], data[target])

    def evaluate(self, test_data, predictors, target):
        preds = self.model.predict(test_data[predictors].values)
        accuracy = accuracy_score(test_data[target], preds)
        precision = precision_score(test_data[target], preds)
        return accuracy, precision

    def predict(self, input_data):
        return self.model.predict(input_data)