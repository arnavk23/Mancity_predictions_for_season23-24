def extract_features(data):
    data["target"] = (data["Result"] == "W").astype("int")
    data["poss"] = data["poss"].astype("int")
    data["passes"] = data["passes"].astype("int")
    return data[["poss", "passes", "target"]]

def feature_engineering(data):
    data = extract_features(data)
    return data

def get_feature_columns():
    return ["poss", "passes"]