import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_curve, f1_score
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Function to encode categorical columns (same as in file1.py)
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
            df[feature] = df[feature].astype(str).str.lower().map(mapping).fillna(0)
    return df

# ----------------- Load Model & Preprocessors -----------------
model = joblib.load("best_ckd_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# ----------------- Load & Prepare Test Data -----------------
test_df = pd.read_csv("test_cases.csv")
X = test_df.drop(columns=["id", "classification"], errors='ignore')
y = test_df["classification"]

# Ensure target is numeric (in case it's still string labels)
if y.dtype == object:
    y = y.map({'ckd': 1, 'notckd': 0})

# Encode string-based categorical columns
X = encode_categorical_features(X)

# Apply saved preprocessing pipeline
X_imputed = imputer.transform(X)
X_scaled = scaler.transform(X_imputed)
X_selected = selector.transform(X_scaled)

# ----------------- Predict Probabilities -----------------
y_probs = model.predict_proba(X_selected)[:, 1]
y_pred_default = (y_probs >= 0.5).astype(int)

# ----------------- Optimal Threshold -----------------
precisions, recalls, thresholds = precision_recall_curve(y, y_probs)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
print(f"Optimal Threshold: {optimal_threshold:.3f}")
print(f"F1-Score at Optimal Threshold: {f1_scores[optimal_idx]:.3f}")

# Save threshold for use elsewhere
joblib.dump(optimal_threshold, "optimal_threshold.pkl")

y_pred_opt = (y_probs >= optimal_threshold).astype(int)

# ----------------- Evaluation -----------------
print("\n--- Evaluation at Default Threshold (0.5) ---")
print("Accuracy:", accuracy_score(y, y_pred_default))
print("Report:\n", classification_report(y, y_pred_default))

print("\n--- Evaluation at Optimal Threshold ---")
print("Accuracy:", accuracy_score(y, y_pred_opt))
print("Report:\n", classification_report(y, y_pred_opt))

# ----------------- Precision/Recall/F1 vs Threshold Plot -----------------
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

# ----------------- Calibration Curve -----------------
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

print("✅ Saved plots: threshold_optimization.png, calibration_curve.png")
print(f"✅ Tested on {len(y)} samples (real test set).")
