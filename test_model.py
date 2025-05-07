import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_curve, f1_score
)
from sklearn.calibration import calibration_curve

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

# Load test data (including edge cases)
test_df = pd.read_csv("test_cases.csv")

# Prepare features and labels
X = test_df.drop(columns=["id", "classification"])
y = test_df["classification"].map({'ckd': 1, 'notckd': 0})

# Encode string-based categorical columns
X = encode_categorical_features(X)

# Preprocess numeric features
X_preprocessed = selector.transform(scaler.transform(imputer.transform(X)))

# Predict probabilities
y_probs = model.predict_proba(X_preprocessed)[:, 1]
y_pred_default = (y_probs >= 0.5).astype(int)

# 1. Threshold tuning using PR/F1
precisions, recalls, thresholds = precision_recall_curve(y, y_probs)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"F1-Score at Optimal Threshold: {f1_scores[optimal_idx]:.3f}")

# Save optimal threshold for use in app.py
joblib.dump(optimal_threshold, "optimal_threshold.pkl")

# Predict using optimal threshold
y_pred_opt = (y_probs >= optimal_threshold).astype(int)

# 2. Evaluation
print("\n--- Evaluation at Default Threshold (0.5) ---")
print("Accuracy:", accuracy_score(y, y_pred_default))
print("Report:\n", classification_report(y, y_pred_default))

print("\n--- Evaluation at Optimal Threshold ---")
print("Accuracy:", accuracy_score(y, y_pred_opt))
print("Report:\n", classification_report(y, y_pred_opt))

# 3. Plot precision, recall, f1 vs threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
plt.axvline(optimal_threshold, color='k', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score vs Threshold')
plt.legend()
plt.grid(True)
plt.savefig('threshold_optimization.png')
plt.close()

# 4. Calibration curve
prob_true, prob_pred = calibration_curve(y, y_probs, n_bins=10, strategy='uniform')
plt.figure(figsize=(8,5))
plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.savefig('calibration_curve.png')
plt.close()

print("Saved plots: threshold_optimization.png, calibration_curve.png")
print("Tested on", len(y), "samples (including edge cases).")
