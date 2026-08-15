# =========================================================
# STEP 3 — FEATURE ENGINEERING
# Create smart indicators: lags, rolling averages, weather factors
# =========================================================

import pandas as pd
import sqlite3

# ─── READ CLEAN DATA ──────────────────────────────────────
print("📂 Reading clean data...")
conn = sqlite3.connect("mig_database.db")
df = pd.read_sql_query("SELECT * FROM cement_usage_clean", conn)
conn.close()

# Convert date & sort properly
df["week_date"] = pd.to_datetime(df["week_date"])
df = df.sort_values(by=["site_id", "week_date"]).reset_index(drop=True)
print(f"✅ {len(df):,} rows loaded — sorted by Site & Date!\n")

# ─── CREATE WORKING COPY ──────────────────────────────────
features = df.copy()

# ─── 1. LAG FEATURES ──────────────────────────────────────
# What did usage look like LAST WEEK? (helps predict THIS week)
print("🔧 Creating LAG features (last week usage)...")
features["usage_lag_1w"] = features.groupby("site_id")["cement_used_tons"].shift(1)
features["usage_lag_2w"] = features.groupby("site_id")["cement_used_tons"].shift(2)
features["opening_lag_1w"] = features.groupby("site_id")["opening_stock_tons"].shift(1)
print("✅ Lag features created!\n")

# ─── 2. ROLLING AVERAGES ──────────────────────────────────
# What's the TREND over past 4 weeks? Smooths out noise
print("🔧 Creating ROLLING averages (4-week trend)...")
features["usage_avg_4w"] = (
    features.groupby("site_id")["cement_used_tons"]
    .transform(lambda x: x.rolling(window=4, min_periods=2).mean())
)
features["opening_avg_4w"] = (
    features.groupby("site_id")["opening_stock_tons"]
    .transform(lambda x: x.rolling(window=4, min_periods=2).mean())
)
print("✅ Rolling averages created!\n")

# ─── 3. WEATHER-ADJUSTED INDICATORS ────────────────────────
print("🔧 Creating WEATHER-ADJUSTED usage factors...")
features["weather_factor"] = 1.0
features.loc[features["rain_this_week"] == 1, "weather_factor"] = 0.6
features["usage_adjusted"] = features["cement_used_tons"] / features["weather_factor"]
print("✅ Weather-adjusted usage created!\n")

# ─── 4. INVENTORY TURNOVER ─────────────────────────────────
# How FAST are we using the silo? Higher = faster usage
print("🔧 Creating INVENTORY TURNOVER metrics...")
features["turnover_rate"] = features["cement_used_tons"] / features["silo_capacity_tons"]
features["stock_utilization_pct"] = (features["opening_stock_tons"] / features["silo_capacity_tons"] * 100).round(1)
print("✅ Turnover & utilization metrics created!\n")

# ─── 5. TIME FEATURES ──────────────────────────────────────
print("🔧 Creating TIME features (Month, Quarter)...")
features["month"] = features["week_date"].dt.month
features["quarter"] = features["week_date"].dt.quarter
features["week_of_year"] = features["week_date"].dt.isocalendar().week
print("✅ Time features created!\n")

# ─── 6. FILL ANY GAPS ──────────────────────────────────────
# First rows have no lag → fill with average
features["usage_lag_1w"] = features.groupby("site_id")["usage_lag_1w"].transform(lambda x: x.fillna(x.mean()))
features["usage_lag_2w"] = features.groupby("site_id")["usage_lag_2w"].transform(lambda x: x.fillna(x.mean()))
features["usage_avg_4w"] = features.groupby("site_id")["usage_avg_4w"].transform(lambda x: x.fillna(x.mean()))
features["opening_lag_1w"] = features.groupby("site_id")["opening_lag_1w"].transform(lambda x: x.fillna(x.mean()))
features["opening_avg_4w"] = features.groupby("site_id")["opening_avg_4w"].transform(lambda x: x.fillna(x.mean()))

# ─── SHOW SUMMARY ──────────────────────────────────────────
print("=" * 60)
print("✅ ALL FEATURES CREATED SUCCESSFULLY!")
print("=" * 60)
print(f"📊 Original columns:  {len(df.columns)}")
print(f"🧩 Total columns NOW: {len(features.columns)}")
print(f"➕ NEW Features added: {len(features.columns) - len(df.columns)}")
print("\n📋 NEW FEATURES LIST:")
new_cols = sorted(set(features.columns) - set(df.columns))
for col in new_cols:
    print(f"   → {col}")

# ─── SAVE TO DATABASE ──────────────────────────────────────
conn = sqlite3.connect("mig_database.db")
features.to_sql("features_ready", conn, if_exists="replace", index=False)
conn.close()

print("\n💾 FEATURES SAVED → Table: features_ready")
print("🎉 STEP 3 COMPLETE! Data is READY for MODELING! 🤖")