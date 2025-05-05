from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('best_ckd_model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')

# Feature names (same order as training)
features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


@app.route('/')
def index():
    return render_template('index.html', features=features)


def preprocess_form_data(form):
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

    processed = []
    age = None
    sc = None
    for feature in features:
        value = form.get(feature)
        if feature in mappings:
            value = mappings[feature].get(value.lower(), 0)
        else:
            value = float(value) if value else 0
        processed.append(value)
        if feature == 'age':
            age = value
        if feature == 'sc':
            sc = value
    gender = form.get('gender', 'female').lower()
    return processed, age, sc, gender


def estimate_gfr(scr, age, gender='female'):
    if scr <= 0 or age <= 0:
        return None
    gfr = 186 * (scr ** -1.154) * (age ** -0.203)
    if gender == 'female':
        gfr *= 0.742
    return gfr


def ckd_gfr_stage(gfr):
    if gfr is None:
        return "Unknown"
    if gfr >= 90:
        return 'G1'
    elif gfr >= 60:
        return 'G2'
    elif gfr >= 45:
        return 'G3a'
    elif gfr >= 30:
        return 'G3b'
    elif gfr >= 15:
        return 'G4'
    else:
        return 'G5'


def personalized_advice(age, bp, bgr, rbc, pc, sc, gfr_stage):
    advice = []

    if bp > 140:
        advice.append("Consider reducing salt intake and monitor blood pressure regularly.")

    if bgr > 140:
        advice.append("Monitor blood sugar levels and consult with a doctor.")

    if rbc == 0:
        advice.append("Abnormal RBC count detected. Please consult a healthcare provider.")

    if pc == 0:
        advice.append("Abnormal pus cells detected. It may indicate an infection; consult a doctor.")

    # CKD Stage-specific advice
    if gfr_stage == 'G1':
        advice.append("Monitor kidney function regularly. Maintain a healthy lifestyle.")
    elif gfr_stage == 'G2':
        advice.append("Control blood pressure and blood sugar. Stay hydrated.")
    elif gfr_stage == 'G3a':
        advice.append("Consult a nephrologist. Limit salt, protein, and potassium.")
    elif gfr_stage == 'G3b':
        advice.append("Prepare for advanced care. Diet control is crucial.")
    elif gfr_stage == 'G4':
        advice.append("Consider dialysis planning. Frequent nephrologist visits needed.")
    elif gfr_stage == 'G5':
        advice.append("Kidney failure. Dialysis or transplant usually required.")

    return " ".join(advice)


@app.route('/predict', methods=['POST'])
def predict():
    input_data, age, sc, gender = preprocess_form_data(request.form)
    input_array = np.array(input_data).reshape(1, -1)

    # Preprocessing
    input_array = imputer.transform(input_array)
    input_array = scaler.transform(input_array)
    input_array = selector.transform(input_array)

    # Prediction
    prob_ckd = model.predict_proba(input_array)[0][1]
    threshold = 0.6
    prediction = 1 if prob_ckd > threshold else 0

    # GFR Calculation
    gfr = estimate_gfr(sc, age, gender)
    gfr_stage = ckd_gfr_stage(gfr)

    # Get personalized advice
    advice = personalized_advice(age, input_data[1], input_data[9], input_data[5], input_data[6], sc, gfr_stage)

    # Final result message
    if prediction == 1 and gfr is not None and gfr >= 90:
        result = f"⚠️ Model predicts CKD (Probability: {prob_ckd:.2%}), but GFR is healthy ({gfr:.1f} ml/min)."
    elif prediction == 0 and gfr is not None and gfr < 60:
        result = f"⚠️ Model predicts No CKD (Probability: {prob_ckd:.2%}), but GFR is low ({gfr:.1f} ml/min)."
    else:
        result = (
            f"Chronic Kidney Disease Detected (Probability: {prob_ckd:.2%})"
            if prediction == 1
            else f"No CKD Detected (Probability: {prob_ckd:.2%})"
        )

    return render_template(
        'result.html',
        result=result,
        gfr=gfr,
        gfr_stage=gfr_stage,
        advice=advice,
        confidence=prob_ckd * 100
    )


if __name__ == '__main__':
    app.run(debug=True)
