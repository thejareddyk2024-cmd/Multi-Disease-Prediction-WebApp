"""
app.py
=======
Multi-Disease Prediction System – Flask Web Application

Routes:
    GET  /                  → Home page
    GET  /diabetes          → Diabetes form
    POST /predict/diabetes  → Diabetes prediction result
    GET  /heart             → Heart Disease form
    POST /predict/heart     → Heart Disease prediction result
    GET  /cancer            → Breast Cancer form
    POST /predict/cancer    → Breast Cancer prediction result
"""

import os
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# ── Load saved models once at startup ────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None  # Will show a friendly error if model not found

diabetes_model = load_model("diabetes_model.pkl")
heart_model    = load_model("heart_model.pkl")
cancer_model   = load_model("cancer_model.pkl")


# ── Helper ────────────────────────────────────────────────────────────────────
def risk_label(prediction: int) -> str:
    """Convert binary prediction to human-readable risk label."""
    return "High Risk ⚠️" if prediction == 1 else "Low Risk ✅"


# ── Home Page ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────────────────────
# DIABETES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/diabetes")
def diabetes_page():
    return render_template("diabetes.html")


@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["blood_pressure"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),          # Diabetes Pedigree Function
            float(request.form["age"]),
        ]
        X = np.array(features).reshape(1, -1)

        if diabetes_model is None:
            raise RuntimeError("Model file not found. Please run train_models.py first.")

        prediction = diabetes_model.predict(X)[0]
        probability = diabetes_model.predict_proba(X)[0][1] * 100
        result = risk_label(prediction)
        return render_template("diabetes.html",
                               result=result,
                               probability=f"{probability:.1f}%",
                               prediction=prediction)
    except Exception as e:
        return render_template("diabetes.html", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# HEART DISEASE
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/heart")
def heart_page():
    return render_template("heart.html")


@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    try:
        features = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"]),
        ]
        X = np.array(features).reshape(1, -1)

        if heart_model is None:
            raise RuntimeError("Model file not found. Please run train_models.py first.")

        prediction  = heart_model.predict(X)[0]
        probability = heart_model.predict_proba(X)[0][1] * 100
        result      = risk_label(prediction)
        return render_template("heart.html",
                               result=result,
                               probability=f"{probability:.1f}%",
                               prediction=prediction)
    except Exception as e:
        return render_template("heart.html", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# BREAST CANCER
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/cancer")
def cancer_page():
    return render_template("cancer.html")


@app.route("/predict/cancer", methods=["POST"])
def predict_cancer():
    try:
        # 10 mean features (most representative subset)
        features = [
            float(request.form["mean_radius"]),
            float(request.form["mean_texture"]),
            float(request.form["mean_perimeter"]),
            float(request.form["mean_area"]),
            float(request.form["mean_smoothness"]),
            float(request.form["mean_compactness"]),
            float(request.form["mean_concavity"]),
            float(request.form["mean_concave_points"]),
            float(request.form["mean_symmetry"]),
            float(request.form["mean_fractal_dimension"]),
            # SE features
            float(request.form["se_radius"]),
            float(request.form["se_texture"]),
            float(request.form["se_perimeter"]),
            float(request.form["se_area"]),
            float(request.form["se_smoothness"]),
            float(request.form["se_compactness"]),
            float(request.form["se_concavity"]),
            float(request.form["se_concave_points"]),
            float(request.form["se_symmetry"]),
            float(request.form["se_fractal_dimension"]),
            # Worst features
            float(request.form["worst_radius"]),
            float(request.form["worst_texture"]),
            float(request.form["worst_perimeter"]),
            float(request.form["worst_area"]),
            float(request.form["worst_smoothness"]),
            float(request.form["worst_compactness"]),
            float(request.form["worst_concavity"]),
            float(request.form["worst_concave_points"]),
            float(request.form["worst_symmetry"]),
            float(request.form["worst_fractal_dimension"]),
        ]
        X = np.array(features).reshape(1, -1)

        if cancer_model is None:
            raise RuntimeError("Model file not found. Please run train_models.py first.")

        prediction  = cancer_model.predict(X)[0]
        probability = cancer_model.predict_proba(X)[0][1] * 100
        result      = risk_label(prediction)
        return render_template("cancer.html",
                               result=result,
                               probability=f"{probability:.1f}%",
                               prediction=prediction)
    except Exception as e:
        return render_template("cancer.html", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
