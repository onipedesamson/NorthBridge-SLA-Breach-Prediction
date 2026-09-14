"""
Step 4 — PRODUCTION MODEL: XGBoost (FINAL FIXED VERSION)
NorthBridge Health Services — SLA Breach Prediction
✅ Handles multi-class targets safely
✅ Auto-converts text columns
✅ No more 'binary' error!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("STEP 4: PRODUCTION MODEL — XGBOOST (FINAL FIXED)")
print("=" * 70)

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv("features_data.csv")
print(f"\n✅ Loaded: {len(df)} rows, {len(df.columns)} columns")

# ==================================================
# SHOW ALL COLUMN NAMES
# ==================================================
print("\n" + "=" * 70)
print("📋 ALL COLUMN NAMES:")
print("=" * 70)
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. '{col}'")

# ==================================================
# FIND OR CREATE SLA BREACH TARGET
# ==================================================
sla_col = None
keywords = ['breach', 'slabreach', 'sla_breach', 'breached', 'isbreach', 
            'breachflag', 'status', 'slastatus', 'result', 'outcome']

for col in df.columns:
    cl = col.lower()
    for kw in keywords:
        if kw in cl:
            sla_col = col
            print(f"\n✅ Found target column: '{col}'")
            break
    if sla_col:
        break

# If not found → CALCULATE from dates (P1 = 8hr SLA)
if sla_col is None:
    print("\n⚠️ Calculating SLA Breach from date columns...")
    due_col, created_col = None, None
    for c in df.columns:
        cl = c.lower()
        if 'sladue' in cl or 'duedate' in cl or 'due_at' in cl or 'sla_due' in cl:
            due_col = c
        if 'created' in cl or 'report' in cl or 'ticketdate' in cl:
            created_col = c
    
    if due_col and created_col:
        print(f"   ✅ Due date: '{due_col}'")
        print(f"   ✅ Created:  '{created_col}'")
        df[due_col] = pd.to_datetime(df[due_col], errors='coerce')
        df[created_col] = pd.to_datetime(df[created_col], errors='coerce')
        hours_available = (df[due_col] - df[created_col]).dt.total_seconds() / 3600
        df['SLA_BREACH'] = (hours_available < 8).fillna(0).astype(int)
        sla_col = 'SLA_BREACH'
        print(f"✅ Created BINARY target: 'SLA_BREACH'")
        print(f"   Breaches: {df[sla_col].sum()} | On-Time: {len(df)-df[sla_col].sum()}")
    else:
        print("\n❌ Cannot identify target column!")
        exit(1)

# ==================================================
# ✅ ENSURE TARGET IS BINARY (0 = OK, 1 = BREACH)
# ==================================================
print(f"\n🔄 Checking target column '{sla_col}' values...")
unique_vals = df[sla_col].dropna().unique()
print(f"   Unique values found: {list(unique_vals)}")

# If target has text/many values → convert to BINARY
if len(unique_vals) > 2 or not all(v in [0,1,True,False] for v in unique_vals):
    print(f"⚠️ Target is NOT binary → converting to BREACH (1) vs OK (0)...")
    val_counts = df[sla_col].value_counts()
    print(f"   Value distribution:\n{val_counts}")
    
    # Try to detect breach value automatically
    breach_patterns = ['breach', 'late', 'violation', 'yes', 'true', '1', 'failed']
    is_breach = df[sla_col].astype(str).str.lower().apply(
        lambda x: any(p in x for p in breach_patterns)
    )
    df['SLA_BREACH'] = is_breach.astype(int)
    sla_col = 'SLA_BREACH'
    print(f"✅ Converted to BINARY: {df[sla_col].sum()} breaches, {len(df)-df[sla_col].sum()} on-time")

# ==================================================
# SELECT & PREPARE FEATURES
# ==================================================
feature_cols = []
exclude = [sla_col, 'TicketID', 'TicketReference', 'ClientID', 'AgentID',
           'AssignedAgentID', 'CreatedAt', 'SLADueAt', 'UpdatedAt',
           'ClientName', 'AgentName', 'Hub', 'ContractTier', 'Channel', 'DayName']

for col in df.columns:
    if col in exclude:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        feature_cols.append(col)
    else:
        print(f"🔄 Encoding text column: '{col}'")
        df[col + '_NUM'] = pd.factorize(df[col].astype(str))[0]
        feature_cols.append(col + '_NUM')

print(f"\n✅ Using {len(feature_cols)} features")

# ==================================================
# PREPARE DATA
# ==================================================
X = df[feature_cols].fillna(0)
y = df[sla_col].fillna(0).astype(int)

print(f"✅ Target: {y.sum()} breaches, {len(y)-y.sum()} on-time")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)} rows")

# ==================================================
# TRAIN XGBOOST
# ==================================================
print("\n🔄 Training XGBoost Model...")
model = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ==================================================
# ✅ FIXED METRICS — Use 'weighted' for safety
# ==================================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
roc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 70)
print("📊 XGBOOST — FINAL RESULTS")
print("=" * 70)
print(f"✅ Accuracy:  {acc:.3f} ({acc*100:.1f}%)")
print(f"✅ Precision: {prec:.3f}")
print(f"✅ Recall:    {recall:.3f}")
print(f"✅ ROC-AUC:   {roc:.3f}")

with open("step4_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {recall:.4f}\nROC-AUC: {roc:.4f}\n")
print("✅ Metrics saved → step4_metrics.txt")

# ==================================================
# 📊 GRAPHS
# ==================================================
# GRAPH 1 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=['On Time', 'BREACHED']).plot(cmap='Greens', ax=ax)
plt.title('📊 CONFUSION MATRIX — XGBoost Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('step4_graph1_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 1 saved")

# GRAPH 2 — Top 10 Feature Importance
fi = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})\
     .sort_values('Importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi, palette='Greens_d')
plt.title('📈 TOP 10 BREACH RISK FACTORS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('step4_graph2_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 2 saved")

# GRAPH 3 — Performance Summary
metrics_df = pd.DataFrame({'Metric': ['Accuracy','Precision','Recall','ROC-AUC'], 'Score': [acc,prec,recall,roc]})
plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Score', data=metrics_df, palette='Greens_d')
plt.ylim(0, 1.05)
plt.title('📊 MODEL PERFORMANCE SUMMARY', fontsize=14, fontweight='bold')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (50%)')
for i, v in enumerate(metrics_df['Score']):
    plt.text(i, v+0.03, f"{v:.1%}", ha='center', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('step4_graph3_performance.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 3 saved")

# ==================================================
# SAVE FINAL MODEL
# ==================================================
with open("production_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("feature_list.txt", "w") as f:
    f.write("\n".join(feature_cols))

print("\n" + "=" * 70)
print("✅ STEP 4 — PRODUCTION MODEL COMPLETE!")
print(f"✅ FINAL MODEL SAVED → production_model.pkl")
print(f"✅ ALL GRAPHS & METRICS SAVED!")
print("=" * 70)