"""
Step 3 — BASELINE MODEL: Logistic Regression
NorthBridge Health Services — SLA Breach Prediction
Purpose: Establish minimum performance benchmark
Outputs: Metrics + 3 Graphs + Benchmark Score
All files saved → Visible on GitHub when pushed!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Set professional graph style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("STEP 3: BASELINE MODEL — LOGISTIC REGRESSION")
print("=" * 70)

# ==================================================
# LOAD DATA FROM STEP 2
# ==================================================
df = pd.read_csv("features_data.csv")
print(f"\n✅ Loaded: {len(df)} rows, {len(df.columns)} columns")

# ==================================================
# AUTO-DETECT OR CREATE SLA BREACH TARGET
# ==================================================
sla_col = None
for col in df.columns:
    cl = col.lower()
    if cl in ['breach', 'slabreach', 'sla_breach', 'breached', 'isbreach', 'breachflag']:
        sla_col = col
        print(f"✅ Found breach column: '{col}'")
        break

# If no column found → CREATE IT from dates
if sla_col is None:
    print("\n⚠️ No Breach column found — auto-calculating from dates...")
    due_col, created_col = None, None
    for c in df.columns:
        cl = c.lower()
        if 'sladue' in cl or 'dueat' in cl or 'sla_due' in cl:
            due_col = c
        if 'created' in cl or 'report' in cl and not due_col:
            created_col = c
    if due_col and created_col:
        df[due_col] = pd.to_datetime(df[due_col], errors='coerce')
        df[created_col] = pd.to_datetime(df[created_col], errors='coerce')
        hours_available = (df[due_col] - df[created_col]).dt.total_seconds() / 3600
        df['SLA_BREACH'] = (hours_available < 8).fillna(0).astype(int)
        sla_col = 'SLA_BREACH'
        print(f"✅ Created target: 'SLA_BREACH'")
        print(f"   Total breaches: {df[sla_col].sum()} ({df[sla_col].mean()*100:.1f}%)")
    else:
        print("\n❌ Cannot find SLA dates — please check your data!")
        exit(1)

# ==================================================
# SELECT FEATURES (NUMERIC COLUMNS ONLY)
# ==================================================
feature_cols = []
exclude = [sla_col, 'TicketID', 'TicketReference', 'ClientID', 'AgentID',
           'AssignedAgentID', 'CreatedAt', 'SLADueAt', 'UpdatedAt',
           'ClientName', 'AgentName', 'Hub', 'ContractTier', 'Channel', 'DayName']

for col in df.columns:
    if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
        feature_cols.append(col)

print(f"\n✅ Using {len(feature_cols)} features for training")

# ==================================================
# PREPARE DATA FOR MODELING
# ==================================================
X = df[feature_cols].fillna(0)
y = df[sla_col].fillna(0).astype(int)

# Split: 70% Train / 30% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"✅ Train set: {len(X_train)} rows")
print(f"✅ Test set:  {len(X_test)} rows")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# TRAIN BASELINE MODEL
# ==================================================
print("\n🔄 Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ==================================================
# EVALUATE PERFORMANCE
# ==================================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
roc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 70)
print("📊 BASELINE MODEL PERFORMANCE")
print("=" * 70)
print(f"✅ Accuracy:  {acc:.3f}  ({acc*100:.1f}%)")
print(f"✅ Precision: {prec:.3f}  — flagged = actually breached")
print(f"✅ Recall:    {recall:.3f}  — real breaches caught")
print(f"✅ ROC-AUC:   {roc:.3f}  — overall predictive power")

with open("step3_metrics.txt", "w") as f:
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"ROC-AUC:   {roc:.4f}\n")
print("✅ Metrics saved → step3_metrics.txt")

# ==================================================
# 📊 GRAPH 1 — CONFUSION MATRIX
# ==================================================
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=['On Time', 'BREACHED']).plot(cmap='Blues', ax=ax)
plt.title('📊 CONFUSION MATRIX — Baseline Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('step3_graph1_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 1 SAVED → step3_graph1_confusion_matrix.png")

# ==================================================
# 📊 GRAPH 2 — TOP 10 FEATURE IMPORTANCE
# ==================================================
coef = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=coef, palette='Blues_d')
plt.title('📈 TOP 10 INFLUENTIAL FACTORS', fontsize=14, fontweight='bold')
plt.xlabel('Impact on Breach Prediction', fontsize=12)
plt.tight_layout()
plt.savefig('step3_graph2_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 2 SAVED → step3_graph2_feature_importance.png")

# ==================================================
# 📊 GRAPH 3 — PERFORMANCE COMPARISON
# ==================================================
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'ROC-AUC'],
    'Score': [acc, prec, recall, roc]
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Score', data=metrics_df, palette='Greens_d')
plt.ylim(0, 1.05)
plt.title('📊 MODEL PERFORMANCE SUMMARY', fontsize=14, fontweight='bold')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (50%)')
for i, v in enumerate(metrics_df['Score']):
    plt.text(i, v + 0.03, f"{v:.1%}", ha='center', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('step3_graph3_performance.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 3 SAVED → step3_graph3_performance.png")

# ==================================================
# SAVE BENCHMARK FOR STEP 4
# ==================================================
with open("baseline_benchmark.txt", "w") as f:
    f.write(f"{roc:.4f}")

print("\n" + "=" * 70)
print(f"✅ STEP 3 COMPLETE!")
print(f"✅ BENCHMARK SCORE: ROC-AUC = {roc:.3f}")
print(f"✅ ALL FILES SAVED → READY TO PUSH TO GITHUB!")
print("=" * 70)