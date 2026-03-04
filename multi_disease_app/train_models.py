"""
train_models.py
================
Multi-Disease Prediction System – Model Training Script
Trains Logistic Regression and Random Forest models for:
  1. Diabetes        (Pima Indians dataset via OpenML / manual CSV)
  2. Heart Disease   (sklearn Heart Disease dataset)
  3. Breast Cancer   (sklearn built-in)

Saves the best model (by accuracy) for each disease as a .pkl file.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: train, compare and save best model
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save(X_train, X_test, y_train, y_test, model_name: str, save_path: str):
    """
    Trains Logistic Regression and Random Forest (inside StandardScaler Pipelines),
    compares accuracy and saves the better model.
    Returns the winning pipeline.
    """
    # ── Model definitions ────────────────────────────────────────────────────
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
    }

    best_model  = None
    best_acc    = 0.0
    best_name   = ""

    print(f"\n{'='*55}")
    print(f"  Training models for: {model_name}")
    print(f"{'='*55}")

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"  [{name}]  Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["Low Risk", "High Risk"]))

        if acc > best_acc:
            best_acc   = acc
            best_model = pipeline
            best_name  = name

    print(f"  ✅ Best model: {best_name}  ({best_acc:.4f})")
    joblib.dump(best_model, save_path)
    print(f"  💾 Saved → {save_path}")
    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIABETES  ─ Pima Indians dataset (fetched from OpenML)
# ─────────────────────────────────────────────────────────────────────────────
def train_diabetes():
    """
    Features (8):
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    Target: 0 = No Diabetes, 1 = Diabetes
    """
    print("\n📌 Fetching Diabetes dataset (Pima Indians) from OpenML …")
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        # OpenML encodes target as 'tested_positive'/'tested_negative'
        df["class"] = (df["class"] == "tested_positive").astype(int)
        df.rename(columns={"class": "Outcome"}, inplace=True)
        feature_cols = [c for c in df.columns if c != "Outcome"]
        X = df[feature_cols].astype(float).values
        y = df["Outcome"].values
    except Exception as e:
        # Fallback: create a synthetic dataset that mimics Pima structure
        print(f"  ⚠  OpenML fetch failed ({e}). Using synthetic fallback data.")
        rng = np.random.default_rng(42)
        n   = 768
        X   = rng.uniform([0, 60, 40, 0, 0, 18, 0.07, 21],
                          [17, 200, 122, 99, 846, 67, 2.5, 81],
                          size=(n, 8))
        # Simple rule-based labels so the model learns something meaningful
        y = ((X[:, 1] > 140) | (X[:, 5] > 35)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_and_save(X_train, X_test, y_train, y_test,
                   "Diabetes", "models/diabetes_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 2. HEART DISEASE  ─ Cleveland dataset from OpenML
# ─────────────────────────────────────────────────────────────────────────────
def train_heart():
    """
    Features (13):
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    Target: 0 = No Disease, 1 = Disease
    """
    print("\n📌 Fetching Heart Disease dataset from UCI repository …")
    try:
        # Use the Cleveland Heart Disease dataset directly from UCI
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        col_names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak",
                     "slope", "ca", "thal", "target"]
        df = pd.read_csv(url, header=None, names=col_names, na_values="?")
        df = df.dropna()
        # Binarise target: 0=no disease, 1+=disease
        df["target"] = (df["target"] > 0).astype(int)
        X = df.drop(columns=["target"]).astype(float).values
        y = df["target"].values
        print(f"  📊 Loaded {len(df)} samples from UCI repository.")
    except Exception as e:
        print(f"  ⚠  UCI fetch failed ({e}). Using OpenML fallback …")
        try:
            from sklearn.datasets import fetch_openml
            from sklearn.preprocessing import LabelEncoder
            data = fetch_openml(name="heart-c", version=1, as_frame=True, parser="auto")
            df = data.frame.copy()
            # Encode all object/category columns to int
            for col in df.columns:
                if df[col].dtype == object or str(df[col].dtype) == "category":
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            df = df.dropna()
            target_col = "class" if "class" in df.columns else df.columns[-1]
            # Binarise: 0 = no disease (label 0), else = disease
            df[target_col] = (df[target_col].astype(float) > 0).astype(int)
            X = df.drop(columns=[target_col]).astype(float).values
            y = df[target_col].values
            print(f"  📊 Loaded {len(df)} samples from OpenML.")
        except Exception as e2:
            print(f"  ⚠  Both fetches failed ({e2}). Using synthetic fallback data.")
            rng = np.random.default_rng(42)
            n   = 303
            X   = rng.uniform(
                [29, 0, 0, 94, 126, 0, 0, 71, 0, 0, 0, 0, 0],
                [77, 1, 3, 200, 564, 1, 2, 202, 1, 6.2, 2, 3, 3],
                size=(n, 13))
            y = ((X[:, 7] < 140) & (X[:, 3] > 140)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_and_save(X_train, X_test, y_train, y_test,
                   "Heart Disease", "models/heart_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BREAST CANCER  ─ sklearn built-in
# ─────────────────────────────────────────────────────────────────────────────
def train_cancer():
    """
    Features (30): mean/se/worst of radius, texture, perimeter, area,
                   smoothness, compactness, concavity, concave points,
                   symmetry, fractal dimension
    Target: 0 = Malignant (High Risk), 1 = Benign (Low Risk)
    Note: We flip the target so 1 = High Risk (Malignant) for consistency.
    """
    print("\n📌 Loading Breast Cancer dataset (sklearn built-in) …")
    data = load_breast_cancer()
    X    = data.data
    # sklearn: 0=malignant, 1=benign → flip so 1=malignant (High Risk)
    y    = 1 - data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_and_save(X_train, X_test, y_train, y_test,
                   "Breast Cancer", "models/cancer_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_diabetes()
    train_heart()
    train_cancer()
    print("\n🎉 All models trained and saved successfully!\n")
