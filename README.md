# Chronic Kidney Disease (CKD) Predictor

A robust machine learning pipeline for early detection and risk assessment of Chronic Kidney Disease (CKD), featuring advanced data processing, synthetic data augmentation, ensemble modeling, and clinical explainability.

---

## Features

- **End-to-end CKD prediction**: Data cleaning, feature engineering, model training, and evaluation.
- **Synthetic data augmentation**: Uses CTGAN to generate additional CKD samples for improved class balance.
- **Multiple ML models**: XGBoost, LightGBM, CatBoost, AdaBoost with hyperparameter tuning.
- **Model explainability**: SHAP and LIME explanations for clinical transparency.
- **Interactive web app**: User-friendly input, clinical calculations (eGFR, CKD stage), and PDF report generation.
- **Personalized advice**: Health tips, dietary recommendations, and differential diagnosis.
- **Robust evaluation**: Calibration curves, threshold optimization, and feature importance plots.

---

## Project Structure

```bash
├── app.py # Flask web application
├── file1.py # Main training pipeline
├── test_model.py # Evaluation and threshold analysis
├── kidney_disease.csv # Raw CKD dataset
├── test_cases.csv # Cleaned, real test set
├── shap_summary.png # SHAP feature importance plot
├── calibration_curve.png # Model calibration plot
├── threshold_optimization.png # Threshold vs Precision/Recall/F1 plot
├── best_ckd_model.pkl # Trained model
├── imputer.pkl # Preprocessing artifact
├── scaler.pkl # Preprocessing artifact
├── selector.pkl # Feature selector artifact
├── X_train.pkl # Training data (post-preprocessing)
├── optimal_threshold.pkl # Best threshold for F1 score
├── static/ # Directory for generated plots and explanations
├── templates/
│ ├── index.html # Web form for user input
│ └── result.html # Results and explanations
└── README.md # Project documentation
```

---

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn, shap, lime, ctgan, flask, reportlab, matplotlib, joblib

Install dependencies:
```bash
pip install -r requirements.txt
```
---

## Data Pipeline

1. **Data Cleaning**  
   - Handles missing values and inconsistent labels.
   - Encodes categorical features and converts numerics.

2. **Test Set Isolation**  
   - Splits off a real, untouched test set for final evaluation (`test_cases.csv`).

3. **Synthetic Augmentation**  
   - Trains CTGAN on the training set to generate additional CKD samples, improving class balance.

4. **Preprocessing**  
   - Median imputation, standard scaling, and feature selection (using XGBoost feature importances).

5. **Balancing**  
   - Applies SMOTE to further address class imbalance.

---

## Model Training

- **Algorithms**: XGBoost, LightGBM, CatBoost, AdaBoost.
- **Hyperparameter Tuning**: Grid search with cross-validation for each model.
- **Feature Selection**: Median thresholding on XGBoost importances.
- **Artifacts Saved**: Best model, preprocessors, selector, and training data.

---

## Evaluation

- **Metrics**: Accuracy, classification report, confusion matrix.
- **Threshold Optimization**: Finds the probability cutoff maximizing F1 score.
- **Calibration Curve**: Assesses probability calibration (see `calibration_curve.png`).
- **Feature Importance**: SHAP summary plot (`shap_summary.png`).
- **Threshold Analysis**: Precision, recall, and F1 vs threshold (`threshold_optimization.png`).

---

## Explainability

- **SHAP**: Global and per-patient feature impact visualization.
- **LIME**: Local explanations for individual predictions (HTML export).
- **Clinical Context**: eGFR calculation, CKD staging, and personalized advice provided in the web app and PDF reports.

---

## Web Application

- **Framework**: Flask
- **Input**: Patient clinical data via a user-friendly form (`index.html`).
- **Output**: CKD probability, eGFR, CKD stage, confidence, clinical advice, top influencing features, differential diagnosis, and downloadable PDF reports.
- **Explainability**: SHAP and LIME explanations available for each prediction.

---

## Usage

### 1. **Training**
```bash
python file1.py
```

- Trains models, saves best artifacts, and generates plots.

### 2. **Testing/Evaluation**
```bash
python test_model.py
```

- Evaluates model on the real test set, generates calibration and threshold plots.

### 3. **Web App**
```bash
python app.py
```

- Runs the Flask app. Access via <http://localhost:5000/>.

---

## Key Results

- **Calibration**: Model probabilities closely match actual outcomes, indicating reliable risk estimates (`calibration_curve.png`).
- **Feature Importance**: Top features include hemoglobin, red blood cell count, and specific gravity (`shap_summary.png`).
- **Threshold Optimization**: Optimal threshold for F1 maximization is highlighted (`threshold_optimization.png`).

---

## Notes

- **Data**: Uses `kidney_disease.csv` (UCI CKD dataset).
- **Test Set**: Evaluation is always on a real, untouched test set (`test_cases.csv`).
- **Explainability**: All predictions are accompanied by clinical explanations and visualizations.

---

## License

This project is for educational and clinical research purposes. Not for direct diagnostic use.

---

## Acknowledgements

- UCI Machine Learning Repository (CKD dataset)
- SHAP, LIME, CTGAN authors
- OpenFDA for drug information API

---

**For questions or contributions, please open an issue or pull request.**
