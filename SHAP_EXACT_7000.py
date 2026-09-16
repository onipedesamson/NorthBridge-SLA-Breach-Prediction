import os, joblib, shap, pandas as pd, matplotlib.pyplot as plt
import numpy as np

# ✅ Auto-find & load model
model = None
for f in os.listdir('.'):
    if f.startswith('production') and f.endswith('.pkl'):
        model = joblib.load(f)
        print(f'✅ MODEL LOADED: {f}')
        break

# ✅ Load data — FORCE EXACTLY 7000 ROWS!
tickets = pd.read_csv('joined_data.csv')
tickets = tickets.iloc[:7000]  # ✅ FORCE FIRST 7000 — NO EXCEPTIONS!
print(f'✅ USING EXACTLY: {len(tickets)} rows')

# ✅ Convert feature columns to NUMBERS
feature_cols = ['HourCreated','DayOfWeek','PriorityLevel','ContractTier']
available_features = []
for col in feature_cols:
    if col in tickets.columns:
        tickets[col] = pd.to_numeric(tickets[col], errors='coerce').fillna(0).astype(int)
        available_features.append(col)
        print(f'✅ CLEANED COLUMN: {col}')

print(f'✅ FEATURES: {available_features}')
X = tickets[available_features]
print(f'✅ FINAL SHAPE: {X.shape}')  # ✅ SHOULD SAY: (7000, 4)

# ✅ SHAP — NOW SHAPE MATCHES PERFECTLY!
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ✅ GRAPH 1
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.title('🏆 TOP FACTORS THAT CAUSE SLA BREACHES', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: shap_feature_importance.png')

# ✅ GRAPH 2
plt.figure(figsize=(8,6))
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False)
plt.title('🔍 WHY THIS TICKET IS AT RISK', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_single_ticket_explanation.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: shap_single_ticket_explanation.png')

print('\n' + '='*60)
print('🎉🎉🎉 SUCCESS! BOTH GRAPHS CREATED! 🎉🎉🎉')
print('='*60)
