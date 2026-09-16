"""
NORTHBRIDGE — DAILY PREDICTION RUN
Run this automatically EVERY DAY → generates fresh risk scores
"""
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# ----------------------
# LOAD EVERYTHING
# ----------------------
model = joblib.load("production_model.pkl")
conn = sqlite3.connect("mig_database.db")

# Load latest tickets
tickets = pd.read_sql("SELECT * FROM Tickets", conn)
conn.close()

# ----------------------
# FEATURES & PREDICT
# ----------------------
feature_cols = ['HourCreated', 'DayOfWeek', 'PriorityLevel', 'ContractTier']
X = tickets[feature_cols]

tickets['Prediction_Date'] = datetime.now().strftime("%Y-%m-%d")
tickets['SLA_Breach_Risk'] = model.predict(X)
tickets['Risk_Probability'] = model.predict_proba(X)[:, 1]

# ----------------------
# SAVE RESULTS
# ----------------------
# Save to CSV (for Power BI / sharing)
today = datetime.now().strftime("%Y-%m-%d")
tickets.to_csv(f"daily_risk_report_{today}.csv", index=False)

# Save BACK to database (live system integration)
conn = sqlite3.connect("mig_database.db")
tickets.to_sql("Ticket_Risk_Scores", conn, if_exists="replace", index=False)
conn.close()

# ----------------------
# SUMMARY REPORT
# ----------------------
high_risk = tickets[tickets['Risk_Probability'] >= 0.70]
print("=" * 60)
print(f"✅ DAILY PREDICTION COMPLETE — {today}")
print(f"📋 Total Tickets Scanned: {len(tickets)}")
print(f"🔴 HIGH RISK Tickets (>70%): {len(high_risk)}")
print(f"📁 Saved: daily_risk_report_{today}.csv")
print(f"💾 Saved to DB: Ticket_Risk_Scores table")
print("=" * 60)