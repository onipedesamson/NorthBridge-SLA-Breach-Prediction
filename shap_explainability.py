"""
NORTHBRIDGE — SHAP EXPLAINABILITY
Shows EXACTLY WHY each ticket is flagged as High Risk
"""
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ----------------------
# LOAD MODEL & DATA
# ----------------------
model = joblib.load("production_model.pkl")

# Load sample data
import sqlite3
conn = sqlite3.connect("mig_database.db")
tickets = pd.read_sql("SELECT * FROM Tickets LIMIT 100", conn)
conn.close()

feature_cols = ['HourCreated', 'DayOfWeek', 'PriorityLevel', 'ContractTier']
X = tickets[feature_cols]

# ----------------------
# SHAP EXPLAINER
# ----------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ----------------------
# OVERALL FEATURE IMPORTANCE
# ----------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("🏆 TOP FACTORS THAT CAUSE SLA BREACHES", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
print("✅ Saved: shap_feature_importance.png")

# ----------------------
# EXPLAIN A SINGLE HIGH-RISK TICKET
# ----------------------
high_risk_idx = 0  # First high-risk ticket
plt.figure(figsize=(8, 6))
shap.force_plot(
    explainer.expected_value,
    shap_values[high_risk_idx],
    X.iloc[high_risk_idx],
    matplotlib=True,
    show=False
)
plt.title("🔍 WHY THIS TICKET IS AT RISK", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_single_ticket_explanation.png", dpi=300, bbox_inches='tight')
print("✅ Saved: shap_single_ticket_explanation.png")

print("\n📊 SHAP COMPLETE! 2 Graphs ready for your presentation!")