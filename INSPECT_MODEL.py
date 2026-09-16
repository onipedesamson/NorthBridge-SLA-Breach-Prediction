import os, joblib

# ✅ Load model and INSPECT it!
model = None
for f in os.listdir('.'):
    if f.startswith('production') and f.endswith('.pkl'):
        model = joblib.load(f)
        print(f'✅ MODEL LOADED: {f}')
        break

# ✅ ASK THE MODEL WHAT SHAPE IT EXPECTS!
print('\n' + '='*60)
print('🔍 MODEL DETAILS:')
print(f'Booster type: {type(model)}')
try:
    print(f'Number of features: {model.n_features_in_}')
except: pass
try:
    print(f'Feature names: {model.feature_names_in_}')
except: pass
try:
    print(f'Model best iteration: {model.best_iteration}')
except: pass
print('='*60)
print('\n📋 ABOVE WILL SHOW: EXACT FEATURE COUNT & NAMES!')
