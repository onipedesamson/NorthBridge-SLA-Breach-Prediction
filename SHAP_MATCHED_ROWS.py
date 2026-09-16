import os, joblib, shap, pandas as pd, matplotlib.pyplot as plt
import numpy as np

# ✅ Auto-find & load model
model = None
for f in os.listdir('.'):
    if f.startswith('production') and f.endswith('.pkl'):
        model = joblib.load(f)
        print(f'✅ MODEL LOADED: {f}')
        break

# ✅ Load data — USE ONLY THE FIRST 7000 ROWS (matches model!)
tickets = pd.read_csv('joined_data.csv')
tickets = tickets.head(7000)  # ✅ EXACT MATCH!
print(f'✅ DATA LOADED: {len(tickets)} rows (MATCHES MODEL!)')

# ✅ Convert feature columns to NUMBERS
feature_cols = ['HourCreated','DayOfWeek','PriorityLevel','ContractTier']
available_features = []
for col in feature_cols:
    if col in tickets.columns:
        tickets[col] = pd.to_numeric(tickets[col], errors='coerce').fillna(0).astype(int)
        available_features.append(col)
        print(f'✅ CLEANED COLUMN: {col}')

print(f'✅ USING FEATURES: {available_features}')
X = tickets[available_features]

# ✅ SHAP Calculation — NOW ROWS MATCH PERFECTLY!
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ✅ GRAPH 1 — Feature Importance
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.title('🏆 TOP FACTORS THAT CAUSE SLA BREACHES', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: shap_feature_importance.png')

# ✅ GRAPH 2 — Single Ticket Explanation
plt.figure(figsize=(8,6))
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False)
plt.title('🔍 WHY THIS TICKET IS AT RISK', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_single_ticket_explanation.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: shap_single_ticket_explanation.png')

print('\n' + '='*60)
print('🎉🎉🎉 SUCCESS! 2 GRAPHS CREATED! 🎉🎉🎉')
print('='*60)
