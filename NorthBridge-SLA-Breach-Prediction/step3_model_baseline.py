"""
Step 3 — Baseline Model: Logistic Regression
NorthBridge Health Services — SLA Breach Prediction
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import joblib

print("=" * 60)
print("STEP 3: BASELINE MODEL — LOGISTIC REGRESSION")
print("=" * 60)

# ---- LOAD DATA ----
X = pd.read_csv("features.csv")
y = pd.read_csv("target.csv").values.ravel()

# ---- SPLIT 80/20 ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ---- SCALE FEATURES ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- TRAIN MODEL ----
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ---- PREDICT & EVALUATE ----
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 BASELINE MODEL PERFORMANCE:")
print(f"   Precision: {precision_score(y_test, y_pred):.3f}")
print(f"   Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, zero_division=0))

# ---- SAVE ----
joblib.dump(model, "baseline_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ STEP 3 COMPLETE → baseline_model.pkl + scaler.pkl")