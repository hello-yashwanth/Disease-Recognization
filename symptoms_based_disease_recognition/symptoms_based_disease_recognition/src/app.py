from flask import Flask, render_template, request, jsonify, url_for, session, redirect, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
import pandas as pd
import os
from dotenv import load_dotenv
import uuid
from database import db
from datetime import datetime
import json
import re
import psycopg2


# =============================
# Load Environment Variables
# =============================

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))


# =============================
# Flask App Setup
# =============================

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')


# =============================
# PostgreSQL Connection (Render)
# =============================

DATABASE_URL = os.environ.get("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Create users table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    email VARCHAR(100),
    password VARCHAR(255),
    role VARCHAR(20) DEFAULT 'user'
);
""")

# Insert default admin user
cursor.execute("""
INSERT INTO users (username,email,password,role)
VALUES ('yash','admin@test.com','123456789','admin')
ON CONFLICT (username) DO NOTHING;
""")

conn.commit()


# =============================
# Import internal modules
# =============================

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


# Initialize database helper
db.init_db()

# Register admin blueprint
app.register_blueprint(admin_bp)


# =============================
# Flask Login Setup
# =============================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'


@login_manager.unauthorized_handler
def _unauthorized():
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


@app.route('/contact')
def contact():
    return render_template('contact.html')


# =============================
# Authentication APIs
# =============================

@app.route('/api/signup', methods=['POST'])
def api_signup():

    data = request.get_json() or {}

    username = data.get('username','').strip()
    email = data.get('email','').strip()
    password = data.get('password','').strip()
    role = "user"

    if not username or not email or not password:
        return jsonify({"success":False,"message":"All fields required"}),400

    email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'

    if not re.match(email_pattern,email):
        return jsonify({"success":False,"message":"Invalid email format"}),400

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
# Run Server
# =============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)