
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
import os
import threading

try:
    # When imported as part of the package
    from ..database import db
    from .dataset_utils import add_disease_to_dataset
    from ..train import train as train_model
except Exception:
    # When running app.py as a plain script
    from database import db
    from admin.dataset_utils import add_disease_to_dataset
    from train import train as train_model

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route('/')
@login_required
def admin_dashboard():
    if not hasattr(current_user, 'role') or current_user.role != 'admin':
        flash('Access denied: Admins only.', 'danger')
        return redirect(url_for('dashboard'))
    diseases = db.get_all_diseases()
    return render_template('admin/dashboard.html', diseases=diseases)


@admin_bp.route('/add_disease', methods=['GET', 'POST'])
@login_required
def add_disease():
    if not hasattr(current_user, 'role') or current_user.role != 'admin':
        flash('Access denied: Admins only.', 'danger')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form.get('disease_name')
        description = request.form.get('description')
        symptoms = request.form.get('symptoms')
        severity = request.form.get('severity')  # overall severity (1-5, optional)
        days = request.form.get('duration_days')  # typical duration in days (optional)
        if not name:
            flash('Disease name is required.', 'danger')
            return render_template('admin/add_disease.html')
        if not symptoms:
            flash('Symptoms are required.', 'danger')
            return render_template('admin/add_disease.html')

        try:
            sev_val = float(severity) if severity else None
        except ValueError:
            sev_val = None

        try:
            days_val = int(days) if days else None
        except ValueError:
            days_val = None

        success = db.add_disease(name, description, sev_val, days_val)
        # Always add to dataset CSV, even if DB insert fails.
        # Base directory is project root (one level above src).
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        dataset_path = os.path.join(project_root, "dataset", "dataset.csv")
        add_disease_to_dataset(name, description, symptoms, dataset_path)
        if success:
            flash('Disease added successfully!', 'success')
        else:
            flash('Disease already exists in database, but symptoms were added to dataset CSV.', 'warning')
        return redirect(url_for('admin.admin_dashboard'))
    return render_template('admin/add_disease.html')


@admin_bp.route('/disease/<int:disease_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_disease(disease_id):
    if not hasattr(current_user, 'role') or current_user.role != 'admin':
        flash('Access denied: Admins only.', 'danger')
        return redirect(url_for('dashboard'))

    disease = db.get_disease_by_id(disease_id)
    if not disease:
        flash('Disease not found.', 'danger')
        return redirect(url_for('admin.admin_dashboard'))

    if request.method == 'POST':
        name = request.form.get('disease_name', '').strip()
        description = request.form.get('description', '').strip()
        severity = request.form.get('severity')
        days = request.form.get('duration_days')

        if not name:
            flash('Disease name is required.', 'danger')
            return render_template('admin/edit_disease.html', disease=disease)

        try:
            sev_val = float(severity) if severity else None
        except ValueError:
            sev_val = None

        try:
            days_val = int(days) if days else None
        except ValueError:
            days_val = None

        ok = db.update_disease(disease_id, name, description, sev_val, days_val)
        if ok:
            flash('Disease updated successfully.', 'success')
        else:
            flash('Failed to update disease. Check server logs.', 'danger')
        return redirect(url_for('admin.admin_dashboard'))

    return render_template('admin/edit_disease.html', disease=disease)


@admin_bp.route('/disease/<int:disease_id>/delete', methods=['POST'])
@login_required
def delete_disease(disease_id):
    if not hasattr(current_user, 'role') or current_user.role != 'admin':
        flash('Access denied: Admins only.', 'danger')
        return redirect(url_for('dashboard'))

    ok = db.delete_disease(disease_id)
    if ok:
        flash('Disease deleted successfully.', 'success')
    else:
        flash('Failed to delete disease. Check server logs.', 'danger')

    return redirect(url_for('admin.admin_dashboard'))

@admin_bp.route('/retrain', methods=['POST'])
@login_required
def retrain_model():
    if not hasattr(current_user, 'role') or current_user.role != 'admin':
        flash('Access denied: Admins only.', 'danger')
        return redirect(url_for('dashboard'))

    # Build absolute paths so retraining always uses the correct dataset/model dirs
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_path = os.path.join(project_root, "dataset", "dataset.csv")
    severity_path = os.path.join(project_root, "dataset", "Symptom-severity.csv")
    models_dir = os.path.join(project_root, "models")

    # Run training in a background thread so the request returns quickly
    def _run_training():
        try:
            print("Starting background model retraining...")
            train_model(dataset_path, severity_path, models_dir)
            print("Model retrained successfully.")
        except Exception as e:
            print("Error during model retraining:", e)

    threading.Thread(target=_run_training, daemon=True).start()
    flash('Model retraining started. This may take about a minute.', 'info')

    return redirect(url_for('admin.admin_dashboard'))
