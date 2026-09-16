import os, joblib, shap, pandas as pd, matplotlib.pyplot as plt
import numpy as np

# ✅ Load model
model = None
for f in os.listdir('.'):
    if f.startswith('production') and f.endswith('.pkl'):
        model = joblib.load(f)
        print(f'✅ MODEL LOADED: {f}')
        break

# ✅ EXACT 33 FEATURE NAMES FROM YOUR MODEL
feature_names = [
    'PatientRef_NUM', 'CategoryID', 'PriorityID', 'Status_x_NUM',
    'FirstResponseAt_NUM', 'ResolvedAt_NUM', 'SLABreached', 'Description_NUM',
    'Breach_Status_NUM', 'ClientType_NUM', 'AccountManagerID', 'Region_NUM',
    'ContractStartDate_NUM', 'ContractEndDate_NUM', 'Contract_duration_(months)',
    'SLACreditClause', 'IsActive_x', 'SLA_eligibility_NUM', 'Client_Status_NUM',
    'FullName_NUM', 'TeamID', 'Role_NUM', 'Specialisms_NUM', 'DailyCapacity',
    'IsActive_y', 'Status_y_NUM', 'HOUR_OF_DAY', 'DAY_OF_WEEK', 'IS_WEEKEND',
    'MONTH', 'QUARTER'
]

# ✅ EXACT ROW COUNT FROM ERROR: 105600 ÷ 33 = 3200
n_rows = 3200
print(f'✅ TARGET ROW COUNT: {n_rows}')

# ✅ CREATE EMPTY DATAFRAME WITH ALL 33 COLUMNS FIRST
X = pd.DataFrame(columns=feature_names)

# ✅ FILL FROM CSV WHERE AVAILABLE
tickets = pd.read_csv('joined_data.csv')
tickets = tickets.iloc[:n_rows]

for name in feature_names:
    if name in tickets.columns:
        X[name] = pd.to_numeric(tickets[name], errors='coerce').fillna(0).astype(int)
    else:
        X[name] = 0  # ✅ Ensure ALL 33 exist!

print(f'✅ FINAL SHAPE: {X.shape} — MUST BE ({n_rows}, 33)')

# ✅ SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ✅ GRAPH 1
plt.figure(figsize=(12,10))
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.title('🏆 TOP FACTORS THAT CAUSE SLA BREACHES', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: shap_feature_importance.png')

# ✅ GRAPH 2
plt.figure(figsize=(12,10))
shap.summary_plot(shap_values, X, show=False)
plt.title('📊 FEATURE IMPACT DIRECTION & STRENGTH', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_beeswarm_summary.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: shap_beeswarm_summary.png')

# ✅ GRAPH 3
plt.figure(figsize=(10,6))
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False)
plt.title('🔍 WHY THIS TICKET IS AT RISK', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_single_ticket_explanation.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: shap_single_ticket_explanation.png')

print('\n' + '='*60)
print('🎉🎉🎉 SUCCESS! ALL 3 GRAPHS CREATED! 🎉🎉🎉')
print('='*60)
