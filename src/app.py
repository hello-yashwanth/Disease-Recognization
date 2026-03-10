from flask import Flask, render_template, request, jsonify, url_for, session, redirect
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
try:
    from .predict import predict_from_input
    from .database import db
except Exception:
    from predict import predict_from_input
    from database import db

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')
app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, email):
        self.id = user_id
        self.username = username
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    user = db.get_user_by_id(int(user_id))
    if user:
        return User(user['id'], user['username'], user['email'])
    return None


@app.context_processor
def inject_current_year():
    from datetime import datetime
    return {'current_year': datetime.now().year}


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


@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    
    # Validation
    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
    
    result = db.register_user(username, email, password)
    return jsonify(result), 200 if result['success'] else 400


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400
    
    result = db.login_user(username, password)
    
    if result['success']:
        user = result.get('user')
        user_obj = User(user['id'], user['username'], user['email'])
        login_user(user_obj)
        return jsonify({'success': True, 'message': 'Login successful'}), 200
    
    return jsonify(result), 401


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')


@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html', user=current_user)


@app.route('/predictor')
def predictor():
    # derive symptom options from dataset (Symptom_1..Symptom_N columns)
    try:
        symptoms = set()
        # from main dataset columns Symptom_*
        ds_path = os.path.join(base_dir, 'dataset', 'dataset.csv')
        if os.path.exists(ds_path):
            df = pd.read_csv(ds_path)
            symptom_cols = [c for c in df.columns if str(c).lower().startswith('symptom')]
            for c in symptom_cols:
                vals = df[c].dropna().astype(str).str.strip()
                symptoms.update([v for v in vals.unique() if v and v.lower() != 'nan'])

        # from Symptom-severity.csv (column 'Symptom')
        sev_path = os.path.join(base_dir, 'dataset', 'Symptom-severity.csv')
        if os.path.exists(sev_path):
            sdf = pd.read_csv(sev_path)
            if 'Symptom' in sdf.columns:
                symptoms.update([str(v).strip() for v in sdf['Symptom'].dropna().unique() if str(v).strip()])

        symptoms = sorted(s for s in symptoms)
    except Exception:
        symptoms = []
    return render_template('predictor.html', symptoms=symptoms)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    body = request.get_json() or {}
    symptoms = body.get('symptoms', {})
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    try:
        res = predict_from_input(symptoms)
        return jsonify(res)
    except FileNotFoundError as e:
        return jsonify({'error': f'Required file not found: {str(e)}'}), 500
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")  # Log to console for debugging
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=True)
