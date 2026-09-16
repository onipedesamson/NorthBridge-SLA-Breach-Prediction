"""
Step 2 — FEATURE ENGINEERING + VISUAL GRAPHS
NorthBridge Health Services — SLA Breach Prediction
Graphs saved as PNG → Ready for PowerPoint!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional graphs
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("STEP 2: FEATURE ENGINEERING + VISUALIZATION")
print("=" * 70)

# ---- LOAD DATA ----
df = pd.read_csv("joined_data.csv")
print(f"\n✅ Loaded: {len(df)} rows, {len(df.columns)} columns")

# ---- SHOW ALL COLUMN NAMES ----
print("\n" + "=" * 70)
print("📋 YOUR COLUMN NAMES:")
print("=" * 70)
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. '{col}'")

# ---- AUTO-DETECT DATE COLUMN ----
date_col = None
for col in df.columns:
    if col.lower() in ['createdat', 'createddate', 'created', 'ticketdate', 'datecreated', 'reportdate', 'slastart']:
        date_col = col
        break

if date_col is None:
    print("\n⚠️ Could not auto-find date column. Looking for ANY date-like column...")
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'create' in col.lower() or 'report' in col.lower():
            date_col = col
            print(f"✅ Found likely date column: '{date_col}'")
            break

if date_col is None:
    print("\n❌ NO DATE COLUMN FOUND — please check the list above & tell me the name!")
    date_col = input("\n👉 Type your date column name: ")

# ---- CONVERT DATE & CREATE TIME FEATURES ----
print(f"\n⏳ Using date column: '{date_col}'")
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

df['HOUR_OF_DAY'] = df[date_col].dt.hour
df['DAY_OF_WEEK'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
df['DAY_NAME'] = df[date_col].dt.day_name()
df['MONTH'] = df[date_col].dt.month
df['QUARTER'] = df[date_col].dt.quarter

print("✅ Time features created!")

# ==================================================
# 📊 GRAPH 1 — TICKETS BY HOUR OF DAY
# ==================================================
plt.figure(figsize=(12, 6))
hourly = df.groupby('HOUR_OF_DAY').size()
sns.barplot(x=hourly.index, y=hourly.values, palette='viridis')
plt.title('📈 TICKETS CREATED BY HOUR OF DAY', fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day (0 = Midnight)', fontsize=14)
plt.ylabel('Number of Tickets', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graph1_tickets_by_hour.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 1 SAVED: graph1_tickets_by_hour.png")

# ==================================================
# 📊 GRAPH 2 — TICKETS BY DAY OF WEEK
# ==================================================
plt.figure(figsize=(12, 6))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily = df.groupby('DAY_NAME').size().reindex(day_order)
sns.barplot(x=daily.index, y=daily.values, palette='coolwarm')
plt.title('📈 TICKETS CREATED BY DAY OF WEEK', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Number of Tickets', fontsize=14)
plt.xticks(rotation=30)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graph2_tickets_by_day.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 2 SAVED: graph2_tickets_by_day.png")

# ==================================================
# 📊 GRAPH 3 — WEEKDAY vs WEEKEND
# ==================================================
plt.figure(figsize=(8, 6))
weekend_counts = df['IS_WEEKEND'].value_counts()
labels = ['Weekday', 'Weekend']
colors = ['#3498db', '#e74c3c']
plt.pie(weekend_counts, labels=labels, autopct='%1.1f%%', colors=colors, 
        textprops={'fontsize': 14}, explode=(0, 0.1))
plt.title('📊 TICKETS: WEEKDAY vs WEEKEND', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('graph3_weekday_vs_weekend.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 3 SAVED: graph3_weekday_vs_weekend.png")

# ==================================================
# 📊 GRAPH 4 — TICKETS BY MONTH
# ==================================================
plt.figure(figsize=(12, 6))
monthly = df.groupby('MONTH').size()
sns.lineplot(x=monthly.index, y=monthly.values, marker='o', linewidth=3, color='#2ecc71')
plt.title('📈 TICKETS TREND BY MONTH', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Tickets', fontsize=14)
plt.xticks(range(1, 13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graph4_tickets_by_month.png', dpi=300, bbox_inches='tight')
print("✅ GRAPH 4 SAVED: graph4_tickets_by_month.png")

# ---- SAVE PROCESSED DATA ----
df.to_csv("features_data.csv", index=False)
print(f"\n" + "=" * 70)
print(f"✅ STEP 2 COMPLETE!")
print(f"✅ Total features created: {len(df.columns)}")
print(f"✅ Data saved: features_data.csv")
print(f"✅ 4 GRAPHS SAVED as PNG files → READY FOR POWERPOINT!")
print("=" * 70)