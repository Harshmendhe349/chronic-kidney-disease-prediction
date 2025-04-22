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
features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'rbc', 'pc', 
            'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

@app.route('/')
def index():
    return render_template('index.html', features=features)

def preprocess_form_data(form):
    # example mappings — use your actual encodings!
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
        
def stage_advice(stage):
    return {
        'G1': "Monitor kidney function regularly. Maintain healthy lifestyle.",
        'G2': "Control blood pressure and blood sugar. Stay hydrated.",
        'G3a': "Consult a nephrologist. Limit salt, protein, and potassium.",
        'G3b': "Prepare for advanced care. Diet control is crucial.",
        'G4': "Consider dialysis planning. Frequent nephrologist visits needed.",
        'G5': "Kidney failure. Dialysis or transplant usually required."
    }.get(stage, "Consult a healthcare professional.")

@app.route('/predict', methods=['POST'])
def predict():
    input_data, age, sc, gender = preprocess_form_data(request.form)
    input_array = np.array(input_data).reshape(1, -1)
    
    # Preprocess
    input_array = imputer.transform(input_array)
    input_array = scaler.transform(input_array)
    input_array = selector.transform(input_array)

    # Predict
    prediction = model.predict(input_array)[0]
    result = 'Chronic Kidney Disease Detected' if prediction == 1 else 'No CKD Detected'
    gfr = estimate_gfr(sc, age, gender)
    gfr_stage = ckd_gfr_stage(gfr)
    advice = stage_advice(gfr_stage)
    return render_template('result.html', result=result, gfr=gfr, gfr_stage=gfr_stage, advice=advice)




if __name__ == '__main__':
    app.run(debug=True)
