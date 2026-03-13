from flask import Flask, render_template, request, jsonify, url_for, redirect, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
import pandas as pd
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import json
import re

from src.database import db
from src.predict import predict_from_input
from src.admin.routes import admin_bp
from src.pdf_utils import generate_prediction_report

# =============================
# Load Environment Variables
# =============================

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

template_dir = os.path.join(BASE_DIR, "templates")
static_dir = os.path.join(BASE_DIR, "static")

# =============================
# Flask App Setup
# =============================

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# =============================
# Initialize Database
# =============================

db.init_db()

# =============================
# Register Admin Blueprint
# =============================

app.register_blueprint(admin_bp)

# =============================
# Flask Login Setup
# =============================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth"


class User(UserMixin):
    def __init__(self, user_id, username, email, role="user"):
        self.id = user_id
        self.username = username
        self.email = email
        self.role = role


@login_manager.user_loader
def load_user(user_id):
    user = db.get_user_by_id(int(user_id))
    if user:
        return User(user["id"], user["username"], user["email"], user.get("role", "user"))
    return None


@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for("auth"))


# =============================
# Context
# =============================

@app.context_processor
def inject_current_year():
    return {"current_year": datetime.now().year}


# =============================
# Page Routes
# =============================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/auth")
def auth():
    return render_template("auth.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# =============================
# Predictor Page
# =============================
@app.route('/history')
@login_required
def prediction_history():

    history = db.get_prediction_history(current_user.id)

    return render_template(
        "history.html",
        history=history,
        user=current_user
    )
@app.route('/settings')
@login_required
def settings():
    return render_template("settings.html", user=current_user)
@app.route("/predictor")
@login_required
def predictor():

    symptoms = []
    diseases = []

    try:
        ds_path = os.path.join(BASE_DIR, "dataset", "dataset.csv")

        if os.path.exists(ds_path):
            df = pd.read_csv(ds_path)

            symptom_cols = [c for c in df.columns if str(c).lower().startswith("symptom")]

            symptom_set = set()

            for col in symptom_cols:
                vals = df[col].dropna().astype(str).str.strip()
                symptom_set.update(vals.unique())

            symptoms = sorted(symptom_set)

    except Exception:
        pass

    try:
        diseases = db.get_all_diseases() or []
    except Exception:
        diseases = []

    return render_template("predictor.html", symptoms=symptoms, diseases=diseases)


# =============================
# Prediction API
# =============================

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():

    try:

        body = request.get_json() or {}

        symptoms = body.get("symptoms", {})
        patient_name = (body.get("patient_name") or "").strip()

        if not patient_name:
            patient_name = current_user.username

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        res = predict_from_input(symptoms)

        report_id = uuid.uuid4().hex[:8].upper()

        db.save_prediction(
            user_id=current_user.id,
            report_id=report_id,
            patient_name=patient_name,
            predicted_disease=res.get("prediction"),
            confidence=res.get("confidence", 0.0),
            symptoms=list(symptoms.keys()),
            recommended_tests=res.get("recommended_tests", []),
        )

        res["report_id"] = report_id
        res["patient_name"] = patient_name

        return jsonify(res)

    except Exception as e:
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/api/report/<report_id>/download", methods=["GET"])
@login_required
def download_report(report_id):

    record = db.get_prediction_by_report_id(report_id, current_user.id)

    if not record:
        return jsonify({"error": "Report not found"}), 404

    patient_name = record.get("patient_name") or "Unknown"
    predicted_disease = record.get("predicted_disease") or "Unknown"
    prediction_date = record.get("prediction_date")

    if isinstance(prediction_date, datetime):
        date_str = prediction_date.strftime("%d/%m/%Y %H:%M:%S")
    else:
        date_str = str(prediction_date)

    tests_raw = record.get("recommended_tests") or ""
    recommended_tests = [t.strip() for t in tests_raw.split(",") if t.strip()]

    symptoms_raw = record.get("symptoms") or "[]"

    try:
        symptoms_list = json.loads(symptoms_raw)
    except:
        symptoms_list = []

    pdf = generate_prediction_report(
        report_id=report_id,
        date_time=date_str,
        patient_name=patient_name,
        symptoms=symptoms_list,
        predicted_disease=predicted_disease,
        recommended_tests=recommended_tests
    )

    filename = f"report_{report_id}.pdf"
    path = os.path.join(BASE_DIR, filename)

    pdf.output(path)

    return send_file(path, as_attachment=True)

# =============================
# Contact API
# =============================

@app.route("/api/contact", methods=["POST"])
def api_contact():

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    message = data.get("message")

    if not name or not email or not message:
        return jsonify({"success": False, "message": "All fields required"}), 400

    success = db.save_contact_message(name, email, message)

    if success:
        return jsonify({"success": True})

    return jsonify({"success": False}), 500


# =============================
# Auth APIs
# =============================

@app.route("/api/signup", methods=["POST"])
def api_signup():

    data = request.get_json() or {}

    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not username or not email or not password:
        return jsonify({"success": False, "message": "All fields required"}), 400

    result = db.register_user(username, email, password, "user")

    return jsonify(result)


@app.route("/api/login", methods=["POST"])
def api_login():

    data = request.get_json() or {}

    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    result = db.login_user(username, password)

    if result["success"]:

        user = result["user"]

        user_obj = User(
            user_id=user["id"],
            username=user["username"],
            email=user["email"],
            role=user.get("role", "user"),
        )

        login_user(user_obj)

        return jsonify(result)

    return jsonify(result), 401


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")


# =============================
# Run Server
# =============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)