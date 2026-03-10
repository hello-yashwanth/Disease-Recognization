import os
import json
import joblib
import pandas as pd
import numpy as np

# SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from .preprocess import load_severity_map
except Exception:
    from preprocess import load_severity_map


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------
# Medical tests mapping
# ---------------------------------------------------
def load_model_features():
    path = os.path.join(BASE_DIR, "models", "feature_importances.csv")
    df = pd.read_csv(path)
    return df["feature"].tolist()
def get_medical_tests_for_disease():

    return {
        "malaria": ["Blood Smear Test", "Rapid Diagnostic Test", "CBC"],
        "allergy": ["Skin Prick Test", "IgE Blood Test"],
        "hypothyroidism": ["TSH Test", "Free T4 Test"],
        "psoriasis": ["Skin Biopsy", "Physical Examination"],
        "gerd": ["Upper Endoscopy", "pH Monitoring"],
        "hepatitis a": ["HAV Antibody Test", "Liver Function Tests"],
        "diabetes": ["Fasting Blood Sugar", "HbA1c Test"],
        "hypertension": ["Blood Pressure Monitoring", "ECG"],
        "common cold": ["Physical Examination"],
        "chicken pox": ["Blood Test", "PCR Test"],
        "migraine": ["Neurological Exam", "MRI"],
        "bronchial asthma": ["Spirometry", "Chest X-Ray"],
        "dengue": ["NS1 Antigen Test", "CBC"],
        "heart attack": ["ECG", "Troponin Test"],
        "pneumonia": ["Chest X-Ray", "Sputum Culture"],
        "tuberculosis": ["Chest X-Ray", "Sputum Test"],
    }


# ---------------------------------------------------
# Prepare input vector
# ---------------------------------------------------

def prepare_input(symptom_presence, severity_map, model_features):

    # Normalize input symptoms
    normalized = {}
    for k, v in symptom_presence.items():
        normalized[str(k).lower().strip().replace(" ", "_")] = v

    # Create full feature vector (all features = 0)
    data = {f: 0.0 for f in model_features}

    for feature in model_features:

        symptom = feature.replace("__w", "")

        if symptom in normalized:

            val = normalized[symptom]

            if isinstance(val, dict):
                presence = float(val.get("presence", 1))
                duration = float(val.get("duration", 1))
            else:
                presence = float(val)
                duration = 1

            severity = severity_map.get(symptom.replace("_", " "), 1)

            data[feature] = presence * severity * duration

    # IMPORTANT: enforce correct column order
    X = pd.DataFrame([data])
    X = X[model_features]
    return X


# ---------------------------------------------------
# Prediction function
# ---------------------------------------------------

def predict_from_input(sample_input,
                       model_path=None,
                       encoder_path=None,
                       severity_path=None):

    if not sample_input:
        raise ValueError("No symptoms provided")

    if model_path is None:
        model_path = os.path.join(BASE_DIR, "models", "xgb_model.joblib")

    if encoder_path is None:
        encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.joblib")

    if severity_path is None:
        severity_path = os.path.join(BASE_DIR, "dataset", "Symptom-severity.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    # Load model
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    severity_map = load_severity_map(severity_path)

    medical_tests_map = get_medical_tests_for_disease()

    # Get feature names
    feature_names = load_model_features()

    # Prepare input
    X = prepare_input(sample_input, severity_map, feature_names)

    # Prediction
    proba = model.predict_proba(X)[0]

    idx = int(np.argmax(proba))

    confidence = float(proba[idx])

    pred_label = le.inverse_transform([idx])[0]

    # Input symptoms for display
    input_symptoms = [
        s.replace("_", " ").title() for s in sample_input.keys()
    ]

    # ---------------------------------------------------
    # SHAP Explanation
    # ---------------------------------------------------

    shap_features = []

    if SHAP_AVAILABLE:
        try:

            explainer = shap.TreeExplainer(model)

            shap_vals = explainer.shap_values(X)

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[idx]

            shap_vals = shap_vals[0]

            feature_map = dict(zip(feature_names, shap_vals))

            top = sorted(
                feature_map.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            shap_features = [
                f.replace("__w", "").replace("_", " ").title()
                for f, _ in top
            ]

        except Exception as e:
            print("SHAP failed:", e)

    if not shap_features:
        shap_features = input_symptoms[:5]

    # ---------------------------------------------------
    # Recommended tests
    # ---------------------------------------------------

    disease_key = pred_label.lower().strip()

    tests = medical_tests_map.get(disease_key, [])

    if not tests:
        tests = ["Physical Examination", "Blood Test"]

    result = {
        "prediction": pred_label,
        "confidence": confidence,
        "top_features_by_model": input_symptoms[:5],
        "top_features_by_shap": shap_features[:5],
        "recommended_tests": tests[:6],
    }

    return result


# ---------------------------------------------------
# CLI test
# ---------------------------------------------------

if __name__ == "__main__":

    example = {
        "fever": {"presence": 1, "duration": 3},
        "cough": {"presence": 1, "duration": 2}
    }

    print(json.dumps(predict_from_input(example), indent=2))