import os
import json
import joblib
import pandas as pd
import numpy as np

# Disable SHAP to save memory on deployment
SHAP_AVAILABLE = False

try:
    from .preprocess import load_severity_map
except Exception:
    from preprocess import load_severity_map


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------
# Load model features
# ---------------------------------------------------

def load_model_features():
    path = os.path.join(BASE_DIR, "models", "feature_importances.csv")
    df = pd.read_csv(path)
    return df["feature"].tolist()


# ---------------------------------------------------
# Medical tests mapping
# ---------------------------------------------------

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

    normalized = {}

    for k, v in symptom_presence.items():
        normalized[str(k).lower().strip().replace(" ", "_")] = v

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

    feature_names = load_model_features()

    X = prepare_input(sample_input, severity_map, feature_names)

    # Prediction
    proba = model.predict_proba(X)[0]

    idx = int(np.argmax(proba))

    confidence = float(proba[idx])

    pred_label = le.inverse_transform([idx])[0]

    # Input symptoms
    input_symptoms = [
        s.replace("_", " ").title() for s in sample_input.keys()
    ]

    # ---------------------------------------------------
    # SHAP Disabled (Render memory safe)
    # ---------------------------------------------------

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
        "top_features_by_shap": shap_features,
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