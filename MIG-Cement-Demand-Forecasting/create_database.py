import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

# ─── SETTINGS ───────────────────────────────────────────
NUM_SITES = 30
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
DATES = pd.date_range(START_DATE, END_DATE, freq="W-MON")

# ─── SITE NAMES ──────────────────────────────────────────
site_names = [
    "Birmingham-HQ", "Birmingham-North", "Birmingham-South",
    "London-North", "London-South", "London-East", "London-West",
    "Manchester-City", "Manchester-North", "Manchester-South",
    "Leeds-Central", "Leeds-West", "Liverpool-Docks",
    "Nottingham-City", "Nottingham-South", "Leicester-Central",
    "Coventry-East", "Wolverhampton-North", "Stoke-Central",
    "Derbyshire-A1-Route", "Derbyshire-Centre",
    "Sheffield-North", "Sheffield-South", "York-Route",
    "Newcastle-East", "Middlesbrough-Central",
    "Bristol-North", "Bristol-South", "Cardiff-Route", "Birmingham-Satellite"
]

# ─── GENERATE DATA ───────────────────────────────────────
data = []
np.random.seed(42)

for site_id, site_name in enumerate(site_names[:NUM_SITES], 1):
    base_demand = np.random.uniform(80, 300)
    silo_capacity = np.random.choice([400, 600, 800, 1000])
    region = site_name.split("-")[0]
    
    for date in DATES:
        month = date.month
        season_factor = 1.0 + 0.3 * np.sin((month - 4) * np.pi / 6)
        rainy = np.random.choice([0, 1], p=[0.7, 0.3])
        weather_factor = 0.6 if rainy else 1.0
        
        planned_pours = np.random.poisson(1.5)
        actual_usage = base_demand * season_factor * weather_factor
        actual_usage += np.random.normal(0, base_demand * 0.08)
        actual_usage = max(0, round(actual_usage, 1))
        
        opening_stock = np.random.uniform(silo_capacity * 0.3, silo_capacity * 0.8)
        delivered = np.random.uniform(0, base_demand * 1.2)
        closing_stock = opening_stock + delivered - actual_usage
        
        avg_temp = round(np.random.uniform(5, 21) + (month - 6) * 1.2, 1)
        rainfall = round(np.random.uniform(0, 45) if rainy else np.random.uniform(0, 8), 1)
        
        data.append([
            f"SITE-{site_id:03d}", site_name, region, date.date(),
            silo_capacity, planned_pours, actual_usage,
            round(opening_stock, 1), round(delivered, 1), round(closing_stock, 1),
            rainy, avg_temp, rainfall
        ])

cols = [
    "site_id", "site_name", "region", "week_date",
    "silo_capacity_tons", "planned_pours", "cement_used_tons",
    "opening_stock_tons", "delivered_tons", "closing_stock_tons",
    "rain_this_week", "avg_temp_c", "rainfall_mm"
]

df = pd.DataFrame(data, columns=cols)

# ─── SAVE TO SQLITE ──────────────────────────────────────
conn = sqlite3.connect("mig_database.db")
df.to_sql("cement_usage", conn, if_exists="replace", index=False)
conn.close()

print("✅ DATABASE CREATED: mig_database.db")
print("✅ TABLE CREATED: cement_usage")
print(f"📊 Total Rows: {len(df):,}")
print(f"📊 Total Sites: {df['site_id'].nunique()}")
print(f"📊 Date Range: {df['week_date'].min()} to {df['week_date'].max()}")
print("\n🎉 DATABASE IS READY!")
