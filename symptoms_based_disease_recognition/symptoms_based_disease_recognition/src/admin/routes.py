from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user
import os
import threading
from functools import wraps

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


# ================================
# Admin Access Decorator
# ================================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):

        if not current_user.is_authenticated:
            return redirect(url_for("auth"))

        if current_user.role != "admin":
            flash("Admin access only!", "danger")
            return redirect(url_for("dashboard"))

        return f(*args, **kwargs)

    return decorated_function


# ================================
# Admin Dashboard
# ================================
@admin_bp.route('/')
@login_required
@admin_required
def admin_dashboard():

    diseases = db.get_all_diseases()

    return render_template(
        "admin/dashboard.html",
        diseases=diseases
    )


# ================================
# Add Disease
# ================================
@admin_bp.route("/add_disease", methods=["GET", "POST"])
@login_required
@admin_required
def add_disease():

    if request.method == "POST":

        name = request.form.get("disease_name")
        description = request.form.get("description")
        symptoms = request.form.get("symptoms")

        severity = request.form.get("severity")
        days = request.form.get("duration_days")

        if not name:
            flash("Disease name is required.", "danger")
            return render_template("admin/add_disease.html")

        if not symptoms:
            flash("Symptoms are required.", "danger")
            return render_template("admin/add_disease.html")

        try:
            sev_val = float(severity) if severity else None
        except ValueError:
            sev_val = None

        try:
            days_val = int(days) if days else None
        except ValueError:
            days_val = None

        success = db.add_disease(name, description, sev_val, days_val)

        # Always update dataset
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        dataset_path = os.path.join(
            project_root,
            "dataset",
            "dataset.csv"
        )

        add_disease_to_dataset(
            name,
            description,
            symptoms,
            dataset_path
        )

        if success:
            flash("Disease added successfully!", "success")
        else:
            flash(
                "Disease already exists in database, but symptoms were added to dataset CSV.",
                "warning"
            )

        return redirect(url_for("admin.admin_dashboard"))

    return render_template("admin/add_disease.html")


# ================================
# Edit Disease
# ================================
@admin_bp.route("/disease/<int:disease_id>/edit", methods=["GET", "POST"])
@login_required
@admin_required
def edit_disease(disease_id):

    disease = db.get_disease_by_id(disease_id)

    if not disease:
        flash("Disease not found.", "danger")
        return redirect(url_for("admin.admin_dashboard"))

    if request.method == "POST":

        name = request.form.get("disease_name", "").strip()
        description = request.form.get("description", "").strip()

        severity = request.form.get("severity")
        days = request.form.get("duration_days")

        if not name:
            flash("Disease name is required.", "danger")
            return render_template(
                "admin/edit_disease.html",
                disease=disease
            )

        try:
            sev_val = float(severity) if severity else None
        except ValueError:
            sev_val = None

        try:
            days_val = int(days) if days else None
        except ValueError:
            days_val = None

        ok = db.update_disease(
            disease_id,
            name,
            description,
            sev_val,
            days_val
        )

        if ok:
            flash("Disease updated successfully.", "success")
        else:
            flash("Failed to update disease.", "danger")

        return redirect(url_for("admin.admin_dashboard"))

    return render_template(
        "admin/edit_disease.html",
        disease=disease
    )


# ================================
# Delete Disease
# ================================
@admin_bp.route("/disease/<int:disease_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_disease(disease_id):

    ok = db.delete_disease(disease_id)

    if ok:
        flash("Disease deleted successfully.", "success")
    else:
        flash("Failed to delete disease.", "danger")

    return redirect(url_for("admin.admin_dashboard"))


# ================================
# Retrain ML Model
# ================================
@admin_bp.route("/retrain", methods=["POST"])
@login_required
@admin_required
def retrain_model():

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    dataset_path = os.path.join(
        project_root,
        "dataset",
        "dataset.csv"
    )

    severity_path = os.path.join(
        project_root,
        "dataset",
        "Symptom-severity.csv"
    )

    models_dir = os.path.join(
        project_root,
        "models"
    )

    def _run_training():

        try:
            print("Starting background model retraining...")

            train_model(
                dataset_path,
                severity_path,
                models_dir
            )

            print("Model retrained successfully.")

        except Exception as e:

            print("Error during model retraining:", e)

    threading.Thread(
        target=_run_training,
        daemon=True
    ).start()

    flash(
        "Model retraining started. This may take about a minute.",
        "info"
    )

    return redirect(url_for("admin.admin_dashboard"))