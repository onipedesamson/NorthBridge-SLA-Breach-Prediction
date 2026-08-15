# =========================================================
# STEP 5 — INTERACTIVE DASHBOARD
# View forecasts, charts, reorder recommendations
# =========================================================

import pandas as pd
import sqlite3
from datetime import datetime

# ─── READ ALL DATA ────────────────────────────────────────
print("📂 Loading data from database...")
conn = sqlite3.connect("mig_database.db")

# Read ALL tables
df_clean = pd.read_sql_query("SELECT * FROM cement_usage_clean", conn)
df_forecast = pd.read_sql_query("SELECT * FROM demand_forecast", conn)
df_importance = pd.read_sql_query("SELECT * FROM feature_importance", conn)

conn.close()

df_clean["week_date"] = pd.to_datetime(df_clean["week_date"])
df_forecast["forecast_week_end"] = pd.to_datetime(df_forecast["forecast_week_end"])

print(f"✅ Historical data: {len(df_clean):,} rows")
print(f"✅ Forecasts: {len(df_forecast):,} predictions")
print(f"✅ Sites: {df_forecast['site_id'].nunique()}\n")

# ─── CALCULATE SUMMARY ──────────────────────────────────────
total_forecast = df_forecast["predicted_tons"].sum()
avg_per_week = df_forecast.groupby("forecast_week_end")["predicted_tons"].sum().mean()
top_site = df_forecast.groupby("site_name")["predicted_tons"].sum().idxmax()
top_site_total = df_forecast.groupby("site_name")["predicted_tons"].sum().max()

# ─── PRINT DASHBOARD ────────────────────────────────────────
print("=" * 70)
print("🏗️  MIG — CEMENT DEMAND FORECASTING DASHBOARD")
print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)
print()

print("📊 8-WEEK NATIONAL SUMMARY")
print("-" * 40)
print(f"   📦 TOTAL Forecasted Demand:  {total_forecast:,.1f} tons")
print(f"   📅 Average per WEEK:          {avg_per_week:,.1f} tons")
print(f"   🏆 Highest Demand Site:        {top_site} ({top_site_total:.1f} tons)")
print(f"   🌧️ Rain-expected weeks:       {df_forecast['rain_expected'].sum() // 30} weeks")
print()

print("📈 TOP 5 INFLUENCING FACTORS")
print("-" * 40)
for _, row in df_importance.head(5).iterrows():
    pct = row["Importance"] * 100
    print(f"   #{len(df_importance.head(5)) - _:d}  {row['Feature']:25s} → {pct:.1f}%")
print()

print("🔮 NEXT 8 WEEKS — TOTAL DEMAND BY WEEK")
print("-" * 40)
weekly = df_forecast.groupby("forecast_week_end").agg({
    "predicted_tons": "sum",
    "rain_expected": lambda x: "🌧️ YES" if x.sum() > len(x)/2 else "☀️ NO"
}).reset_index()
weekly.columns = ["Week Ending", "Total Tons", "Rain?"]

for _, row in weekly.iterrows():
    date_str = row["Week Ending"].date()
    bar = "█" * int(row["Total Tons"] / 100)
    print(f"   {date_str}  |  {row['Total Tons']:7.1f} tons  {row['Rain?']}  {bar}")
print()

print("📦 REORDER RECOMMENDATIONS")
print("-" * 40)
print("   Based on forecasted demand & silo capacity:\n")

top_sites = df_forecast.groupby(["site_id", "site_name", "silo_capacity_tons"])["predicted_tons"].sum().reset_index()
top_sites = top_sites.sort_values("predicted_tons", ascending=False).head(8)

print(f"   {'Site Name':25s} {'Capacity':>10s} {'8-Wk Need':>10s} {'Status':12s} {'Recommendation'}")
print("   " + "-"*65)

for _, row in top_sites.iterrows():
    cap = row["silo_capacity_tons"]
    need = row["predicted_tons"]
    ratio = need / cap
    
    if ratio > 0.75:
        status = "🔴 CRITICAL"
        action = f"🚚 DELIVER {need:.0f}t IMMEDIATELY!"
    elif ratio > 0.50:
        status = "🟡 PLANNING"
        action = f"📅 Schedule {need:.0f}t this month"
    else:
        status = "🟢 SUFFICIENT"
        action = "✅ No urgent delivery needed"
    
    print(f"   {row['site_name']:25s} {cap:10.0f} {need:10.1f}  {status:12s} {action}")

print()
print("=" * 70)
print("🎉 STEP 5 COMPLETE! PROJECT FINISHED!")
print("=" * 70)
print()
print("📋 YOUR COMPLETE PROJECT SUMMARY:")
print("   ✅ Database created & populated with realistic data")
print("   ✅ Data cleaned & validated")
print("   ✅ Patterns discovered: Seasonality, Weather impact, Site differences")
print("   ✅ Smart features engineered for prediction")
print("   ✅ AI Model trained & 8-week forecast generated")
print("   ✅ Dashboard & Reorder Recommendations delivered")
print()
print("🏆 CONGRATULATIONS! You have built a COMPLETE Cement Demand Forecasting System!")
print("=" * 70)