import os, joblib, shap, pandas as pd, matplotlib.pyplot as plt

# ✅ Auto-find the model file
model = None
for f in os.listdir('.'):
    if f.startswith('production') and f.endswith('.pkl'):
        model = joblib.load(f)
        print(f'✅ MODEL LOADED: {f}')
        break

# ✅ USE YOUR CSV FILE INSTEAD OF DATABASE! NO TABLE NEEDED!
tickets = pd.read_csv('joined_data.csv')
print(f'✅ DATA LOADED: {len(tickets)} rows')
print(f'✅ COLUMNS FOUND: {list(tickets.columns)}')

# ✅ Try to find the right feature columns automatically
possible_cols = ['HourCreated','Hour_Created','DayOfWeek','Day_Of_Week','PriorityLevel','Priority','ContractTier','Contract_Tier']
feature_cols = [c for c in possible_cols if c in tickets.columns]
print(f'✅ USING FEATURES: {feature_cols}')

X = tickets[feature_cols]

# ✅ SHAP Calculation
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
print('🎉🎉🎉 ALL DONE! 2 GRAPHS CREATED SUCCESSFULLY! 🎉🎉🎉')
print('='*60)
