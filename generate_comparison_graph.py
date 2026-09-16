"""
Generate: Baseline vs Production — Comparison Graph
Saves as: model_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# YOUR ACTUAL SCORES
# ----------------------
# Replace these with YOUR real numbers from step3_metrics.txt and step4_metrics.txt
baseline_scores = [0.81, 0.78, 0.75, 0.85]   # Accuracy, Precision, Recall, ROC-AUC
production_scores = [0.92, 0.89, 0.87, 0.94]  # Accuracy, Precision, Recall, ROC-AUC

# ----------------------
# SETUP CHART
# ----------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 7))

# ----------------------
# DRAW BARS
# ----------------------
bar1 = plt.bar(x - width/2, baseline_scores, width, 
               label='Baseline\n(Logistic Regression)', 
               color='#3498db', edgecolor='#2471a3', linewidth=2)

bar2 = plt.bar(x + width/2, production_scores, width, 
               label='Production\n(XGBoost)', 
               color='#27ae60', edgecolor='#1e8449', linewidth=2)

# ----------------------
# ADD LABELS & VALUES
# ----------------------
plt.ylabel('Score (0–1)', fontsize=14, fontweight='bold')
plt.title('🏆 MODEL PERFORMANCE COMPARISON\nBaseline vs Production', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, metrics, fontsize=13, fontweight='bold')
plt.ylim(0, 1.08)
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, label='Random Guess (50%)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=11)

# Show numbers ON TOP of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.025,
                 f'{height:.1%}', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold')

add_labels(bar1)
add_labels(bar2)

# ----------------------
# SAVE & SHOW
# ----------------------
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH SAVED → model_comparison.png")
plt.show()