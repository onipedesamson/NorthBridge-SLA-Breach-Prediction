"""
Step 4 — Production Model: XGBoost Classifier
NorthBridge Health Services — SLA Breach Prediction
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import joblib

print("=" * 60)
print("STEP 4: PRODUCTION MODEL — XGBOOST")
print("=" * 60)

# ---- LOAD DATA ----
X = pd.read_csv("features.csv")
y = pd.read_csv("target.csv").values.ravel()

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---- TRAIN XGBOOST ----
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ---- PREDICT & EVALUATE ----
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n🚀 XGBOOST MODEL PERFORMANCE:")
print(f"   Precision: {precision_score(y_test, y_pred):.3f}")
print(f"   Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, zero_division=0))

# ---- FEATURE IMPORTANCE ----
print("\n🔝 TOP FEATURES:")
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).head(8)
for _, row in importance.iterrows():
    print(f"   {row['Feature']:25} → {row['Importance']:.1%}")

# ---- SAVE PRODUCTION MODEL ----
joblib.dump(model, "production_model.pkl")
print("\n✅ STEP 4 COMPLETE → production_model.pkl")