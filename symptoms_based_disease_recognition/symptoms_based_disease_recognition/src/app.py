from flask import Flask, render_template, request, jsonify, url_for, session, redirect, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
import pandas as pd
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import json
import re
# Load environment variables
load_dotenv()

try:
    from .predict import predict_from_input
    from .database import db
    from .pdf_utils import generate_prediction_report
    from .admin.routes import admin_bp
except Exception:
    from predict import predict_from_input
    from database import db
    from pdf_utils import generate_prediction_report
    from admin.routes import admin_bp

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Register admin blueprint (admin panel at /admin)
app.register_blueprint(admin_bp)


# =============================
# Flask Login Setup
# =============================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'


@login_manager.unauthorized_handler
def _unauthorized():
    # For API calls, return JSON instead of redirecting to HTML login page
    if request.path.startswith("/api/"):
        return jsonify({"error": "Login required"}), 401
    return redirect(url_for("auth"))


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


@app.context_processor
def inject_current_year():
    return {"current_year": datetime.now().year}


# =============================
# Page Routes
# =============================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/auth')
def auth():
    return render_template('auth.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html', user=current_user)


@app.route('/history')
@login_required
def prediction_history():
    history = db.get_prediction_history(current_user.id)
    return render_template('history.html', history=history, user=current_user)


# =============================
# Authentication APIs
# =============================

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json() or {}

    username = data.get('username','').strip()
    email = data.get('email','').strip()
    password = data.get('password','').strip()
    role = data.get('role','user')

    if not username or not email or not password:
        return jsonify({"success":False,"message":"All fields required"}),400

    # Email format validation
    email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
    if not re.match(email_pattern,email):
        return jsonify({"success":False,"message":"Invalid email format"}),400

    # Only allow gmail
    if not email.lower().endswith("@gmail.com"):
        return jsonify({
            "success":False,
            "message":"Only Gmail accounts are allowed"
        }),400

    if len(password) < 6:
        return jsonify({"success":False,"message":"Password must be at least 6 characters"}),400

    result = db.register_user(username,email,password,role)

    return jsonify(result)


@app.route('/api/login', methods=['POST'])
def api_login():

    data = request.get_json() or {}

    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400

    result = db.login_user(username, password)

    if result["success"]:

        user = result["user"]

        user_obj = User(
            user_id=user["id"],
            username=user["username"],
            email=user["email"],
            role=user.get("role", "user")
        )

        login_user(user_obj)

        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user.get("role", "user")
            }
        })

    return jsonify(result), 401


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


# =============================
# Prediction System
# =============================

@app.route('/predictor')
@login_required
def predictor():

    symptoms = []
    diseases = []

    try:
        ds_path = os.path.join(base_dir, 'dataset', 'dataset.csv')

        if os.path.exists(ds_path):
            df = pd.read_csv(ds_path)

            symptom_cols = [c for c in df.columns if str(c).lower().startswith('symptom')]

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

    return render_template('predictor.html', symptoms=symptoms, diseases=diseases)


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():

    body = request.get_json() or {}

    symptoms = body.get("symptoms", {})
    # If patient name is not provided in the form, default to the logged-in username
    patient_name = (body.get("patient_name") or "").strip()
    if not patient_name:
        patient_name = getattr(current_user, "username", "Unknown")

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    try:
        res = predict_from_input(symptoms)
    except Exception as e:
      import traceback
      print("PREDICTION ERROR:")
      print(traceback.format_exc())
      return jsonify({"error": str(e)}), 500

    user_id = current_user.id

    # Short receipt-like ID (matches the style shown in sample report)
    report_id = uuid.uuid4().hex[:8].upper()

    db.save_prediction(
        user_id=user_id,
        report_id=report_id,
        patient_name=patient_name,
        predicted_disease=res.get("prediction"),
        confidence=res.get("confidence", 0.0),
        symptoms=[k.replace("_", " ").title() for k in (symptoms or {}).keys()],
        recommended_tests=res.get("recommended_tests", []),
    )

    res["report_id"] = report_id
    res["patient_name"] = patient_name
    res["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return jsonify(res)


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
        date_str = prediction_date.strftime("%d/%m/%Y, %H:%M:%S")
    else:
        date_str = str(prediction_date) if prediction_date else datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    tests_raw = record.get("recommended_tests") or ""
    recommended_tests = [t.strip() for t in tests_raw.split(",") if t.strip()]

    symptoms_raw = record.get("symptoms") or "[]"
    try:
        symptoms_list = json.loads(symptoms_raw)
        if not isinstance(symptoms_list, list):
            symptoms_list = []
    except Exception:
        symptoms_list = []

    pdf = generate_prediction_report(
        report_id=report_id,
        date_time=date_str,
        patient_name=patient_name,
        symptoms=symptoms_list,
        predicted_disease=predicted_disease,
        recommended_tests=recommended_tests,
    )

    filename = f"report_{report_id}.pdf"
    out_path = os.path.join(base_dir, "tmp_" + filename)
    try:
        pdf.output(out_path)
        return send_file(out_path, as_attachment=True, download_name=filename, mimetype="application/pdf")
    finally:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass


# =============================
# Run Server
# =============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)