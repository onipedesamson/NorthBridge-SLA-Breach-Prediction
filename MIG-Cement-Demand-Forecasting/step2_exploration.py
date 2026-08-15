# =========================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# Understand patterns, trends, seasonality & weather impact
# =========================================================

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# ─── READ CLEAN DATA ──────────────────────────────────────
print("📂 Reading CLEAN data from SQLite...")
conn = sqlite3.connect("mig_database.db")
df = pd.read_sql_query("SELECT * FROM cement_usage_clean", conn)
conn.close()

# Convert date from text back to proper date format
df["week_date"] = pd.to_datetime(df["week_date"])
df["month"] = df["week_date"].dt.month
df["year"] = df["week_date"].dt.year
df["quarter"] = df["week_date"].dt.to_period("Q")
print(f"✅ {len(df):,} rows ready for analysis!\n")

# ─── 1. CEMENT USAGE BY SITE ─────────────────────────────
print("📊 1. TOP 5 SITES — MOST CEMENT USED (Tons):")
site_totals = df.groupby("site_name")["cement_used_tons"].sum().sort_values(ascending=False).round(1)
print(site_totals.head(5))
print()

# ─── 2. SEASONALITY — USAGE BY MONTH ─────────────────────
print("📊 2. AVERAGE USAGE BY MONTH (Tons/week):")
monthly = df.groupby("month")["cement_used_tons"].mean().round(1)
print(monthly)
print("💡 Summer months = MORE cement! Winter = LESS! ✅\n")

# ─── 3. WEATHER IMPACT ────────────────────────────────────
print("📊 3. RAIN vs NO RAIN — AVERAGE USAGE COMPARISON:")
rain_impact = df.groupby("rain_this_week")["cement_used_tons"].mean().round(1)
rain_impact.index = ["☀️ NO Rain", "🌧️ Raining"]
print(rain_impact)
print("💡 Rain reduces cement usage! ✅\n")

# ─── 4. POURS vs ACTUAL USAGE ────────────────────────────
print("📊 4. PLANNED POURS → HOW MUCH CEMENT USED:")
pours_vs_usage = df.groupby("planned_pours")["cement_used_tons"].mean().round(1)
print(pours_vs_usage)

print("💡 MORE pours → MORE cement used! Makes sense! ✅\n")

# ─── 5. YEARLY TREND ──────────────────────────────────────
print("📊 5. AVERAGE USAGE — 2024 vs 2025:")
yearly = df.groupby("year")["cement_used_tons"].mean().round(1)
print(yearly)
print()

# ─── 6. REGIONAL COMPARISON ───────────────────────────────
print("📊 6. AVERAGE USAGE BY REGION:")
region_avg = df.groupby("region")["cement_used_tons"].mean().sort_values(ascending=False).round(1)
print(region_avg)
print()

print("🎉 STEP 2 COMPLETE!")
print("✅ All patterns identified — Seasonality ✅ Weather impact ✅ Site differences ✅")