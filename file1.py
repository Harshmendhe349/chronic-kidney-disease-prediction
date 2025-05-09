import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from lime import lime_tabular
from sklearn.utils import shuffle
from ctgan import CTGAN 
import joblib

# -------------------- Load and Clean Data --------------------
df = pd.read_csv('kidney_disease.csv')

def clean_data(df):
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    df.replace(['\t?', '?'], np.nan, inplace=True)

    num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    df[cat_cols] = df[cat_cols].apply(lambda x: x.str.lower())

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

df_clean = clean_data(df)
df_clean = df_clean[df_clean['classification'].notna()]  # Remove rows with NaN in label

# -------------------- Save Real Test Set --------------------
X_real = df_clean.drop(['id', 'classification'], axis=1)
y_real = df_clean['classification']

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, stratify=y_real, random_state=42
)

# Save real, untouched test set
test_df = X_test_real.copy()
test_df['classification'] = y_test_real
test_df.to_csv("test_cases.csv", index=False)
print("✅ Saved clean test set as 'test_cases.csv'")

# -------------------- Start Augmented Training Pipeline --------------------
# Use only X_train_real for synthetic augmentation
X = X_train_real.copy()
y = y_train_real.copy()

# Drop NaNs in training labels
mask = ~y.isna()
X = X[mask]
y = y[mask]

# ------------------ CTGAN SYNTHETIC DATA GENERATION ------------------
print("\n🔁 Training CTGAN to generate synthetic CKD samples...")
df_ctgan = X.copy()
df_ctgan['classification'] = y
df_ctgan_clean = df_ctgan.dropna()

ctgan = CTGAN(epochs=300)
ctgan.fit(df_ctgan_clean)

synthetic_data = ctgan.sample(len(df_ctgan_clean))
synthetic_ckd = synthetic_data[synthetic_data['classification'] == 1]

df_augmented = pd.concat([df_ctgan, synthetic_ckd], ignore_index=True)
df_augmented = shuffle(df_augmented, random_state=42)

X = df_augmented.drop('classification', axis=1)
y = df_augmented['classification']

# ------------------ Preprocessing Pipeline ------------------
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
selector = SelectFromModel(estimator=xgb, threshold='median')
X_selected = selector.fit_transform(X_scaled, y_res)
selected_features = X.columns[selector.get_support()]
print(f"✅ Selected features: {list(selected_features)}")

# ------------------ Model Training ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_res, test_size=0.2, random_state=42, stratify=y_res
)

joblib.dump(X_train, 'X_train.pkl')

models = {
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [100, 200]
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(),
        'params': {
            'num_leaves': [31, 50],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(verbose=0),
        'params': {
            'iterations': [100, 200],
            'depth': [4, 6],
            'learning_rate': [0.03, 0.1]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    }
}

best_models = {}
for name, config in models.items():
    print(f"\n🔍 Training {name}...")
    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"✅ Best params for {name}: {grid.best_params_}")
    print(f"✅ Best CV accuracy: {grid.best_score_:.2%}")

# ------------------ Evaluation ------------------
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f"\n📊 {name} Performance:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# ------------------ SHAP Explainability ------------------
explainer = shap.TreeExplainer(best_models['XGBoost'])
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=selected_features, plot_type='bar', show=False)
plt.title('Feature Importance using SHAP')
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()

# ------------------ LIME Explainability ------------------
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=selected_features,
    class_names=['No CKD', 'CKD'],
    mode='classification'
)

exp = explainer_lime.explain_instance(
    X_test[0],
    best_models['XGBoost'].predict_proba,
    num_features=10
)
exp.save_to_file('lime_explanation.html')

# ------------------ Save Final Artifacts ------------------
joblib.dump(best_models['XGBoost'], 'best_ckd_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')

print("\n✅ Model training and saving completed successfully.")
