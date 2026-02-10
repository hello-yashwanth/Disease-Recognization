from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# -------------------- APP SETUP --------------------
app = Flask(__name__)

# ✅ Enable CORS globally
CORS(app)

# ✅ FORCE CORS HEADERS (VERY IMPORTANT FOR KOYEB)
@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


# -------------------- LOAD DATA --------------------
data = pd.read_csv("./Training.csv")
data.drop(["Unnamed: 133"], axis=1, inplace=True)

disease_map = {
    'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,
    'Hypertension ':10,'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,
    'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,
    'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,
    'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,
    'Hypothyroidism':31,'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,
    'Arthritis':35,'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,
    'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40
}

data.replace({"prognosis": disease_map}, inplace=True)

X = data.iloc[:, :131].values
Y = data.iloc[:, -1].values
x_train, _, _, _ = train_test_split(X, Y, test_size=0.3, random_state=42)

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("./Multiple_disease_prediction_rf.pkl", "rb"))

columns_dict = {col: i for i, col in enumerate(data.columns)}
predict_ans = {v: k for k, v in disease_map.items()}

def generate_input(selected_options):
    new_input = np.zeros((1, x_train.shape[1]))
    for symptom in selected_options:
        if symptom in columns_dict:
            new_input[0][columns_dict[symptom]] = 1
    return new_input


# -------------------- ROUTES --------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json()

        # accept both formats
        if isinstance(data, dict) and "options" in data:
            selected_options = data["options"]
        elif isinstance(data, list):
            selected_options = data
        else:
            return jsonify({"error": "Invalid input format"}), 400

        new_input = generate_input(selected_options)
        result = model.predict(new_input)
        disease = predict_ans[int(result[0])]

        return jsonify({"disease": disease})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def health_check():
    return jsonify({"status": "Backend running successfully"})


# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
