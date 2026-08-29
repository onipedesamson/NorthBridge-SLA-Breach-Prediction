"""
Step 2 — Feature Engineering
NorthBridge Health Services — SLA Breach Prediction
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

# ---- LOAD RAW DATA ----
df = pd.read_csv("joined_data.csv")
print(f"Loaded: {len(df)} rows")

# ---- 1. TIME FEATURES from CREATEDAT ----
df["CREATEDAT"] = pd.to_datetime(df["CREATEDAT"])
df["HOUR_OF_DAY"] = df["CREATEDAT"].dt.hour
df["DAY_OF_WEEK"] = df["CREATEDAT"].dt.dayofweek  # 0=Monday, 6=Sunday

# ---- 2. ENCODE CATEGORICAL FEATURES ----
# Contract Tier → map to numbers
tier_map = {"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}
df["CONTRACT_TIER_NUM"] = df["CONTRACTTIER"].map(tier_map).fillna(2)

# Channel → One-Hot Encoding
df = pd.get_dummies(df, columns=["CHANNEL"], drop_first=True)

# Hub → One-Hot Encoding
df = pd.get_dummies(df, columns=["HUB"], drop_first=True)

# SLACreditClause → ensure numeric
df["SLACREDITCLAUSE"] = df["SLACREDITCLAUSE"].astype(float).fillna(0)

# ---- 3. SELECT FINAL FEATURES ----
feature_cols = [
    "PRIORITYID", "CATEGORYID",
    "HOUR_OF_DAY", "DAY_OF_WEEK",
    "CONTRACT_TIER_NUM", "SLACREDITCLAUSE",
    "TEAMID", "DAILYCAPACITY"
] + [col for col in df.columns if col.startswith(("CHANNEL_", "HUB_"))]

# Drop rows with missing values in features
df_clean = df.dropna(subset=feature_cols + ["SLABREACHED"])

# Prepare final dataset
X = df_clean[feature_cols].astype(float)
y = df_clean["SLABREACHED"].astype(int)

# Save
X.to_csv("features.csv", index=False)
y.to_csv("target.csv", index=False)

print(f"✅ Features created: {len(feature_cols)} columns")
print(f"✅ Final dataset: {len(X)} rows")
print(f"✅ Feature list: {feature_cols}")
print("\n✅ STEP 2 COMPLETE → features.csv + target.csv")