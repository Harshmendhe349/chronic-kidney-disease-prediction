import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def encode_categorical_features(df):
    mappings = {
        'rbc': {'normal': 1, 'abnormal': 0},
        'pc': {'normal': 1, 'abnormal': 0},
        'pcc': {'present': 1, 'notpresent': 0},
        'ba': {'present': 1, 'notpresent': 0},
        'htn': {'yes': 1, 'no': 0},
        'dm': {'yes': 1, 'no': 0},
        'cad': {'yes': 1, 'no': 0},
        'appet': {'good': 1, 'poor': 0},
        'pe': {'yes': 1, 'no': 0},
        'ane': {'yes': 1, 'no': 0}
    }

    for feature, mapping in mappings.items():
        if feature in df.columns:
            df[feature] = df[feature].str.lower().map(mapping).fillna(0)

    return df

# Load saved model and preprocessing objects
model = joblib.load("best_ckd_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# Load test data
test_df = pd.read_csv("test_cases.csv")

# Prepare features and labels
X = test_df.drop(columns=["id", "classification"])
y = test_df["classification"].map({'ckd': 1, 'notckd': 0})  # <-- Fix

# Encode string-based categorical columns
X = encode_categorical_features(X)

# Preprocess numeric features and predict
X_preprocessed = selector.transform(scaler.transform(imputer.transform(X)))
y_pred = model.predict(X_preprocessed)

# Evaluation
print("Accuracy:", accuracy_score(y, y_pred))
print("Report:\n", classification_report(y, y_pred))
