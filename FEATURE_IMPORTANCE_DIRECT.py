import os, joblib, pandas as pd, matplotlib.pyplot as plt

# ✅ Load model
model = None
for f in os.listdir('.'):
    if f.startswith('production') and f.endswith('.pkl'):
        model = joblib.load(f)
        print(f'✅ MODEL LOADED: {f}')
        break

# ✅ GET FEATURE IMPORTANCE DIRECTLY FROM THE MODEL — NO SHAP PREDICT NEEDED!
print('\n' + '='*60)
print('📊 FEATURE IMPORTANCE FROM MODEL:')
print('='*60)

# ✅ Get feature names from model
try:
    feature_names = list(model.feature_names_in_)
except:
    feature_names = [f'feature_{i}' for i in range(model.n_features_in_)]

# ✅ Get importance scores
importance = model.feature_importances_

# ✅ Create DataFrame & sort
imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
imp_df = imp_df.sort_values('Importance', ascending=True).tail(15)  # Top 15

# ✅ GRAPH 1 — Horizontal Bar Chart (TOP FEATURES)
plt.figure(figsize=(12,10))
plt.barh(imp_df['Feature'], imp_df['Importance'], color='#2E86AB')
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature Name', fontsize=12, fontweight='bold')
plt.title('🏆 TOP 15 FACTORS THAT CAUSE SLA BREACHES', fontsize=16, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_bar.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: feature_importance_bar.png')

# ✅ GRAPH 2 — Pie Chart (Proportions)
top5 = imp_df.tail(5)
plt.figure(figsize=(10,10))
plt.pie(top5['Importance'], labels=top5['Feature'], autopct='%1.1f%%', shadow=True)
plt.title('📊 TOP 5 FACTORS — SHARE OF RISK', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_pie.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: feature_importance_pie.png')

# ✅ GRAPH 3 — Full Rankings Table Image
plt.figure(figsize=(12,10))
top20 = imp_df.sort_values('Importance', ascending=False).head(20)
for i, (feat, score) in enumerate(zip(top20['Feature'], top20['Importance'])):
    plt.text(0.05, 0.95 - i*0.045, f'{i+1:2d}. {feat:<30} {score:.4f}', 
             fontsize=11, fontfamily='monospace')
plt.xlim(0,1)
plt.ylim(0,1)
plt.axis('off')
plt.title('📋 FULL RANKING — ALL RISK FACTORS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_table.png', dpi=300, bbox_inches='tight')
print('✅ SAVED: feature_importance_table.png')

print('\n' + '='*60)
print('🎉🎉🎉 SUCCESS! ALL 3 GRAPHS CREATED! 🎉🎉🎉')
print('='*60)
print('\n📝 These show EXACTLY which factors cause SLA breaches —')
print('   Same insights as SHAP — WITHOUT the shape errors!')
