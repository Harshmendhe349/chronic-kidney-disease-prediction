import joblib
import pandas as pd

model = joblib.load('model.pkl')
pipeline = joblib.load('pipeline.pkl')  # if you saved preprocessing steps

# Load a few rows from your training Excel/CSV
df = pd.read_csv('chronic_kidney_disease.csv')
df = df.dropna().sample(5)  # test on 5 clean rows

# Remove label if present
X = df.drop(columns=['classification'], errors='ignore')

# Predict
X_preprocessed = pipeline.transform(X)
preds = model.predict(X_preprocessed)

print("Sample Predictions:", preds)
