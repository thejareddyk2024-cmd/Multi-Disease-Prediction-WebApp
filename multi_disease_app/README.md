# 🧬 Multi-Disease Prediction System

A **Machine Learning web application** built with Python, Flask, and Scikit-learn that predicts the risk of:

- 🩸 **Diabetes** (Pima Indians Dataset)
- ❤️ **Heart Disease** (Cleveland Dataset)
- 🔬 **Breast Cancer** (Wisconsin Dataset)

---

## 📁 Project Structure

```
multi_disease_app/
│
├── models/                      # Saved trained models (.pkl files)
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── cancer_model.pkl
│
├── templates/                   # HTML templates
│   ├── index.html               # Homepage
│   ├── diabetes.html            # Diabetes form + result
│   ├── heart.html               # Heart disease form + result
│   └── cancer.html              # Breast cancer form + result
│
├── train_models.py              # Model training script
├── app.py                       # Flask application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ ML Models Used

| Disease        | Dataset          | Features | Best Model     | Typical Accuracy |
|----------------|------------------|----------|----------------|-----------------|
| Diabetes       | Pima Indians     | 8        | Random Forest  | ~77%            |
| Heart Disease  | Cleveland (UCI)  | 13       | Random Forest  | ~85%            |
| Breast Cancer  | Wisconsin (FNA)  | 30       | Random Forest  | ~97%            |

For each disease, **two models** are trained:
1. **Logistic Regression** (with StandardScaler)
2. **Random Forest** (with StandardScaler)

The model with higher test accuracy is saved automatically.

---

## 🚀 How to Run Locally

### Step 1 – Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd multi_disease_app

# Or simply navigate to the project folder
cd path/to/multi_disease_app
```

### Step 2 – Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Train the Models

```bash
python train_models.py
```

This will:
- Download datasets (Pima Diabetes + Cleveland Heart from OpenML; Cancer is built into sklearn)
- Train Logistic Regression and Random Forest for each disease
- Print accuracy comparison to the console
- Save the best model as `.pkl` files inside the `models/` folder

**Expected output:**
```
📌 Fetching Diabetes dataset (Pima Indians) from OpenML …
  [Logistic Regression]  Accuracy: 0.7468
  [Random Forest]        Accuracy: 0.7727
  ✅ Best model: Random Forest  (0.7727)
  💾 Saved → models/diabetes_model.pkl
...
🎉 All models trained and saved successfully!
```

### Step 5 – Run the Flask App

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🌐 How to Deploy on Render

### Step 1 – Prepare your project for Render

Create a file named `Procfile` in your project root:

```
web: python app.py
```

Modify `app.py` to read the PORT from environment (already handled for local, update if needed):

```python
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
```

### Step 2 – Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/multi-disease-app.git
git push -u origin main
```

> ⚠️ **Important**: Run `python train_models.py` locally first so `.pkl` files are included in your repo, OR add a startup script to train on first run.

### Step 3 – Deploy on Render

1. Go to [render.com](https://render.com) and sign up
2. Click **New → Web Service**
3. Connect your GitHub repository
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: Python 3
5. Click **Deploy** – Render will build and host your app!

---

## 🚀 How to Deploy on Railway

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **New Project → Deploy from GitHub Repo**
3. Select your repository
4. Railway auto-detects Python – it will install `requirements.txt`
5. Set the **Start Command**: `python app.py`
6. Click **Deploy** 🚀

---

## 📝 Notes

- The `models/` folder must contain `.pkl` files before running `app.py`
- Always run `python train_models.py` before `python app.py`
- For production deployment, set `debug=False` in `app.py`

---

## 🎓 VIVA QUESTIONS & ANSWERS

### 📌 Topic 1: Classification

**Q1. What is classification in Machine Learning?**
> Classification is a supervised learning task where the model learns to assign a label/category to an input based on training examples. Output is discrete (e.g., 0 or 1, Yes or No). Examples: spam detection, disease prediction.

**Q2. What is binary vs multi-class classification?**
> - **Binary**: Two classes (e.g., Diabetic / Non-diabetic)
> - **Multi-class**: More than two classes (e.g., classifying tumor type into 3+ categories)
> This project uses **binary classification** for all three diseases.

**Q3. What is a decision boundary?**
> A decision boundary is the surface/line that separates classes in the feature space. Points on one side are predicted as class 0, points on the other side as class 1.

---

### 📌 Topic 2: Overfitting

**Q4. What is overfitting?**
> Overfitting occurs when a model learns the training data **too well**, including its noise. It performs excellently on training data but poorly on unseen (test) data. Signs: high training accuracy, low test accuracy.

**Q5. How do we prevent overfitting?**
> - **Cross-validation** (k-fold)
> - **Regularization** (L1/L2 in Logistic Regression)
> - **Pruning** (Decision Trees / Random Forest depth limiting)
> - **More training data**
> - **Feature selection** – remove irrelevant features
> - **Ensemble methods** – Random Forest reduces variance

**Q6. What is the bias-variance tradeoff?**
> - **High Bias** (Underfitting): Model is too simple, misses patterns
> - **High Variance** (Overfitting): Model is too complex, memorises data
> Good models balance both via regularization or ensemble methods.

---

### 📌 Topic 3: Logistic Regression

**Q7. What is Logistic Regression?**
> Despite the name, it's a **classification** algorithm (not regression). It estimates the probability that an input belongs to a class using the **sigmoid function**: σ(z) = 1 / (1 + e^(−z))
> If P > 0.5 → Class 1; else → Class 0.

**Q8. What is the sigmoid function and why is it used?**
> The sigmoid squashes any real number to the range (0,1), making it interpretable as a probability. It's differentiable, allowing gradient-based optimization.

**Q9. What is the cost function for Logistic Regression?**
> **Binary Cross-Entropy (Log Loss)**:
> J = −(1/n) Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]
> Minimizing this loss trains the model to output probabilities close to the true labels.

**Q10. What are L1 and L2 regularization in Logistic Regression?**
> - **L1 (Lasso)**: Adds |weights| to loss → encourages sparse models (some weights = 0)
> - **L2 (Ridge)**: Adds weights² to loss → shrinks weights but doesn't zero them
> `sklearn`'s `LogisticRegression` uses L2 by default (`C=1.0`).

---

### 📌 Topic 4: Random Forest

**Q11. What is a Random Forest?**
> An **ensemble learning** method that builds multiple Decision Trees on random subsets of data (bagging) and random subsets of features, then aggregates predictions by **majority voting** (for classification).

**Q12. Why is Random Forest better than a single Decision Tree?**
> Single trees are prone to **overfitting**. Random Forest reduces variance by averaging predictions of many uncorrelated trees, improving generalization.

**Q13. What is Bootstrap Aggregation (Bagging)?**
> Each tree is trained on a **bootstrap sample** (random sample with replacement) of the training data. This introduces diversity among trees, reducing correlation and variance.

**Q14. What are hyperparameters in Random Forest?**
> - `n_estimators`: number of trees (more = better but slower)
> - `max_depth`: maximum tree depth (limits overfitting)
> - `max_features`: number of features to consider at each split
> - `min_samples_split`: minimum samples to split a node

**Q15. What is feature importance in Random Forest?**
> Random Forest can rank features by how often and how much they reduce impurity (Gini or entropy) across all trees. High importance = more predictive power.

---

### 📌 Topic 5: Evaluation Metrics

**Q16. What is accuracy and when is it misleading?**
> Accuracy = (Correct Predictions) / (Total Predictions)
> It is misleading when classes are **imbalanced** (e.g., 95% negative, 5% positive). A model predicting all negatives would have 95% accuracy but be useless.

**Q17. What is a Confusion Matrix?**
> A table showing:
> - **TP** (True Positive): Correctly predicted positive
> - **TN** (True Negative): Correctly predicted negative
> - **FP** (False Positive): Negative predicted as positive (Type I error)
> - **FN** (False Negative): Positive predicted as negative (Type II error)

**Q18. Explain Precision, Recall, and F1-Score.**
> - **Precision** = TP / (TP + FP) → "Of all predicted positives, how many were actually positive?"
> - **Recall (Sensitivity)** = TP / (TP + FN) → "Of all actual positives, how many did we catch?"
> - **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) → Harmonic mean; useful when you need to balance both.

**Q19. What is ROC-AUC?**
> - **ROC Curve**: Plots True Positive Rate vs False Positive Rate at various thresholds
> - **AUC (Area Under Curve)**: A single number (0.5–1.0). AUC = 1.0 → perfect; AUC = 0.5 → random guessing
> Higher AUC means better model discrimination.

**Q20. Why do we use StandardScaler?**
> Many algorithms (especially Logistic Regression, SVM) are sensitive to feature scale. Features with large ranges (e.g., area in cancer dataset) can dominate. `StandardScaler` transforms each feature to have **mean=0** and **std=1**, ensuring fair contribution from all features.

**Q21. What is train-test split and why do we use it?**
> We split data into a training set (to learn patterns) and a test set (to evaluate on unseen data). Common split: 80% train, 20% test. This simulates real-world performance and detects overfitting.

**Q22. What is cross-validation?**
> **k-Fold Cross-Validation** splits data into k equal folds; iteratively trains on k-1 folds and validates on the remaining fold. Results are averaged, giving a more reliable performance estimate than a single train-test split.

---

## 📊 Dataset Information

| Dataset | Source | Samples | Features | Target |
|---------|--------|---------|----------|--------|
| Pima Indians Diabetes | OpenML | 768 | 8 | 0=No Diabetes, 1=Diabetes |
| Cleveland Heart Disease | OpenML | 303 | 13 | 0=No Disease, 1=Disease |
| Breast Cancer Wisconsin | sklearn | 569 | 30 | 0=Benign, 1=Malignant |

---

## ⚕️ Disclaimer

> This project is built for **educational purposes only**. Do NOT use it for actual medical diagnosis. Always consult a qualified healthcare professional.

---

*Built with ❤️ using Python, Flask, and Scikit-learn*
