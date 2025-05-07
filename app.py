from flask import Flask, render_template, request, send_file, json
import numpy as np
import joblib
import shap
from lime import lime_tabular
import time
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import os
import json
from json import JSONDecodeError

app = Flask(__name__)

# Load model, preprocessors, and training data
model = joblib.load('best_ckd_model.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')
X_train = joblib.load('X_train.pkl')  # Make sure you saved this during training


# Feature names (same order as training)
features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
            'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


# Create static directory if not exists
if not os.path.exists('static'):
    os.makedirs('static')

# Initialize explainers
explainer_shap = shap.TreeExplainer(model)
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=features,
    class_names=['No CKD', 'CKD'],
    mode='classification',
    discretize_continuous=True
)


reverse_mappings = {
    'rbc': {1: 'normal', 0: 'abnormal'},
    'pc': {1: 'normal', 0: 'abnormal'},
    'pcc': {1: 'present', 0: 'notpresent'},
    'ba': {1: 'present', 0: 'notpresent'},
    'htn': {1: 'yes', 0: 'no'},
    'dm': {1: 'yes', 0: 'no'},
    'cad': {1: 'yes', 0: 'no'},
    'appet': {1: 'good', 0: 'poor'},
    'pe': {1: 'yes', 0: 'no'},
    'ane': {1: 'yes', 0: 'no'}
}

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

def differential_diagnosis(input_data):
    conditions = []
    bp = input_data[1]
    sc = input_data[11]
    bgr = input_data[9]
    
    if sc > 1.2:
        conditions.extend(["Acute Kidney Injury", "Dehydration"])
    if bp > 140:
        conditions.append("Hypertensive Nephropathy")
    if bgr > 140:
        conditions.append("Diabetic Nephropathy")
    
    return list(set(conditions))[:3] if conditions else ["No alternative conditions detected"]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Preprocess form data
        input_data, age, sc, gender = preprocess_form_data(request.form)
        input_array = np.array(input_data).reshape(1, -1)

        # Preprocessing pipeline
        input_array = imputer.transform(input_array)
        input_array = scaler.transform(input_array)
        input_array = selector.transform(input_array)

        # Generate predictions
        prob_ckd = model.predict_proba(input_array)[0][1]
        threshold = 0.6  # Can be dynamic based on test_cases.csv
        prediction = 1 if prob_ckd > threshold else 0

        # Clinical calculations
        gfr = estimate_gfr(sc, age, gender) if sc > 0 and age > 0 else None
        gfr_stage = ckd_gfr_stage(gfr)

        # Get top 3 influential features using SHAP
        shap_values = explainer_shap.shap_values(input_array)
        feature_importance = sorted(zip(features, shap_values[0]), 
                                key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Generate differential diagnosis
        diff_diag = differential_diagnosis(input_data)
        
        # SHAP visualization
        timestamp = int(time.time())
        shap_path = f"static/shap_{timestamp}.png"
        try:
            shap_values = explainer_shap.shap_values(input_array)
            plt.figure(figsize=(10,6))
            shap.summary_plot(shap_values, input_array, 
                            feature_names=features,
                            show=False)
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"SHAP Error: {str(e)}")
            shap_path = None

        # LIME explanation
        lime_path = f"static/lime_{timestamp}.html"
        try:
            exp = explainer_lime.explain_instance(
                input_array[0], 
                model.predict_proba, 
                num_features=10
            )
            exp.save_to_file(lime_path)
        except Exception as e:
            print(f"LIME Error: {str(e)}")
            lime_path = None    

        # Generate personalized advice
        advice = personalized_advice(
            age, input_data[1], input_data[9], 
            input_data[5], input_data[6], sc, gfr_stage
        )

        # Result message logic
        if prediction == 1 and gfr and gfr >= 90:
            result = f"⚠️ Model predicts CKD (Probability: {prob_ckd:.2%}), but GFR is healthy ({gfr:.1f} ml/min)."
        elif prediction == 0 and gfr and gfr < 60:
            result = f"⚠️ Model predicts No CKD (Probability: {prob_ckd:.2%}), but GFR is low ({gfr:.1f} ml/min)."
        else:
            result = (
                f"Chronic Kidney Disease Detected (Probability: {prob_ckd:.2%})"
                if prediction == 1 
                else f"No CKD Detected (Probability: {prob_ckd:.2%})"
            )
        
        user_data_dict = dict(zip(features, input_data))
        return render_template(
            'result.html',
            result=result,
            gfr=gfr,
            gfr_stage=gfr_stage,
            advice=advice,
            confidence=prob_ckd * 100,
            shap_plot=shap_path,
            lime_explanation=lime_path,
            feature_importance=feature_importance,
            diff_diag=diff_diag,
            user_data=user_data_dict,
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

# PDF Report Generation
@app.route('/download_report', methods=['POST'])
def download_report():
    print("RAW user_data from form:", request.form['user_data'])  # See what you get
    raw_data = request.form['user_data']
    app.logger.debug(f"Raw user_data: {raw_data}")
    try:
        
        # Get form data with proper encoding
        user_data = json.loads(request.form['user_data'].replace("'", '"'))
        
        # Handle legacy list format
        if isinstance(user_data, list):
            user_data = dict(zip(features, user_data))

        result = request.form['result']
        advice = request.form['advice']
        confidence = request.form['confidence']
        gfr = request.form.get('gfr', 'N/A')
        gfr_stage = request.form.get('gfr_stage', 'N/A')

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Custom style for tips
        tip_style = ParagraphStyle(
            'TipStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=6
        )

        # Title
        elements.append(Paragraph("<b>Chronic Kidney Disease Prediction Report</b>", 
                                styles['Title']))
        elements.append(Spacer(1, 24))
        

        # User Input Table
        elements.append(Paragraph("<b>Patient Input Data</b>", styles['Heading2']))
        table_data = [['Feature', 'Value']]
        
        # Add user input features
        feature_mapping = {
            'age': 'Age',
            'bp': 'Blood Pressure',
            'sg': 'Specific Gravity',
            'al': 'Albumin',
            'su': 'Sugar',
            'rbc': 'Red Blood Cells',
            'pc': 'Pus Cells',
            'pcc': 'Pus Clumps',
            'ba': 'Bacteria',
            'bgr': 'Blood Glucose',
            'bu': 'Blood Urea',
            'sc': 'Serum Creatinine',
            'sod': 'Sodium',
            'pot': 'Potassium',
            'hemo': 'Hemoglobin',
            'pcv': 'Packed Cell Volume',
            'wc': 'White Blood Cells',
            'rc': 'Red Blood Cell Count',
            'htn': 'Hypertension',
            'dm': 'Diabetes',
            'cad': 'Coronary Artery Disease',
            'appet': 'Appetite',
            'pe': 'Pedal Edema',
            'ane': 'Anemia'
        }
        
        for feature, value in user_data.items():
            display_name = feature_mapping.get(feature, feature.title())

            if feature in reverse_mappings:
                # Convert value to int for mapping (handles float inputs like 1.0)
                value = reverse_mappings[feature].get(int(float(value)), value)
            
            table_data.append([display_name, str(value)])
        
        # Create and style table
        table = Table(table_data, colWidths=[250, 250])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0078D4')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F0F0F0')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        elements.append(table)
        elements.append(PageBreak())

        # Prediction Results
        elements.append(Paragraph("<b>Prediction Results</b>", styles['Heading2']))
        results = [
            f"<b>Result:</b> {result}",
            f"<b>eGFR:</b> {gfr} ml/min/1.73m²",
            f"<b>CKD Stage:</b> {gfr_stage}",
            f"<b>Confidence:</b> {confidence}%"
        ]
        for res in results:
            elements.append(Paragraph(res, tip_style))
        elements.append(Spacer(1, 24))

        # Personalized Advice
        elements.append(Paragraph("<b>Personalized Health Advice</b>", styles['Heading2']))
        for tip in advice.split('. '):
            if tip.strip():
                elements.append(Paragraph(f"• {tip.strip()}", tip_style))
        elements.append(Spacer(1, 24))

        # Kidney-Friendly Diet Section
        elements.append(Paragraph("<b>Kidney-Friendly Diet Tips</b>", styles['Heading2']))
        diet_tips = [
            "Limit sodium intake to <2,000 mg/day (avoid processed foods)",
            "Choose low-potassium fruits (apples, berries) and vegetables",
            "Opt for lean protein sources (fish, egg whites)",
            "Monitor phosphorus intake (limit dairy, nuts, beans)",
            "Stay hydrated with water (consult doctor for fluid restrictions)"
        ]
        for tip in diet_tips:
            elements.append(Paragraph(f"• {tip}", tip_style))
        elements.append(Spacer(1, 24))

        # Exercise & Lifestyle Section
        elements.append(Paragraph("<b>Exercise & Lifestyle Recommendations</b>", styles['Heading2']))
        exercise_tips = [
            "Aim for 30 mins daily moderate activity (walking, cycling)",
            "Monitor blood pressure during exercise",
            "Practice stress-reduction techniques (yoga, meditation)",
            "Avoid NSAIDs (ibuprofen) without doctor approval",
            "Get annual kidney function tests"
        ]
        for tip in exercise_tips:
            elements.append(Paragraph(f"• {tip}", tip_style))

        # Generate PDF
        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, 
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name='CKD_Health_Report.pdf')
    
    except json.JSONDecodeError as e:
        return f"Invalid JSON data: {str(e)}", 400
    
    except Exception as e:
        app.logger.error(f"PDF Error: {str(e)}")
        raise e

if __name__ == '__main__':
    app.run(debug=True)
