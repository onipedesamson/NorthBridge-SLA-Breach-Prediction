# =========================================================
# STEP 1 — DATA INGESTION & CLEANING
# Read data FROM SQLite → Check quality → Fix issues → Save CLEAN data
# =========================================================

import pandas as pd
import sqlite3

# ─── 1. CONNECT TO DATABASE & READ DATA ──────────────────
print("📂 Reading data from SQLite...")
conn = sqlite3.connect("mig_database.db")
df = pd.read_sql_query("SELECT * FROM cement_usage", conn)
conn.close()
print(f"✅ READY! {len(df):,} rows loaded.\n")

# ─── 2. BASIC OVERVIEW ───────────────────────────────────
print("📋 DATA OVERVIEW:")
print(f"   Columns:  {list(df.columns)}")
print(f"   Sites:    {df['site_id'].nunique()}")
print(f"   Date min: {df['week_date'].min()}")
print(f"   Date max: {df['week_date'].max()}\n")

# ─── 3. CHECK FOR MISSING VALUES ─────────────────────────
print("🔍 CHECKING FOR MISSING VALUES:")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ NO missing values! Perfect!\n")
else:
    print("⚠️ Missing values found — will fill them!\n")
    df = df.fillna(method="ffill").fillna(0)

# ─── 4. CHECK FOR NEGATIVE NUMBERS ───────────────────────
print("🔍 CHECKING FOR NEGATIVE VALUES (inventory/usage):")
neg_usage = df[df["cement_used_tons"] < 0].shape[0]
neg_opening = df[df["opening_stock_tons"] < 0].shape[0]
neg_closing = df[df["closing_stock_tons"] < 0].shape[0]

print(f"   Negative usage:     {neg_usage}")
print(f"   Negative opening:   {neg_opening}")
print(f"   Negative closing:   {neg_closing}")

# Fix negatives → set to 0
df["cement_used_tons"] = df["cement_used_tons"].clip(lower=0)
df["opening_stock_tons"] = df["opening_stock_tons"].clip(lower=0)
df["closing_stock_tons"] = df["closing_stock_tons"].clip(lower=0)
print("✅ All values fixed to ≥ 0\n")

# ─── 5. INVENTORY BALANCE CHECK ──────────────────────────
print("🔍 INVENTORY BALANCE: Opening + Delivered − Used ≈ Closing")
df["calculated_closing"] = df["opening_stock_tons"] + df["delivered_tons"] - df["cement_used_tons"]
tolerance = 0.01 * df["silo_capacity_tons"]
mismatch = (abs(df["closing_stock_tons"] - df["calculated_closing"]) > tolerance).sum()
print(f"   Mismatched records: {mismatch}")
if mismatch == 0:
    print("✅ Inventory balance PERFECT!\n")
else:
    print("⚠️ Some mismatches — corrected!\n")
    df["closing_stock_tons"] = df["calculated_closing"]

# ─── 6. SAVE CLEAN DATA ──────────────────────────────────
df = df.drop(columns=["calculated_closing"])
conn = sqlite3.connect("mig_database.db")
df.to_sql("cement_usage_clean", conn, if_exists="replace", index=False)
conn.close()

print("🎉 STEP 1 COMPLETE!")
print("✅ CLEAN data saved to table: cement_usage_clean")
print(f"✅ Final dataset: {len(df):,} rows • {df['site_id'].nunique()} sites")