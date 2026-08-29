"""
Step 1 — Load & Join Data from Snowflake REPORTING Schema
NorthBridge Health Services — SLA Breach Prediction
"""

import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

print("=" * 60)
print("STEP 1: LOADING DATA FROM SNOWFLAKE")
print("=" * 60)

# ---- CONNECT TO SNOWFLAKE ----
conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA", "REPORTING")
)

print("✅ Connected to Snowflake successfully!")

# ---- JOIN ALL 3 TABLES ----
query = """
SELECT
    t.TICKETID,
    t.PRIORITYID,
    t.CATEGORYID,
    t.CHANNEL,
    t.CREATEDAT,
    t.SLABREACHED,
    c.CONTRACTTIER,
    c.SLACREDITCLAUSE,
    a.HUB,
    a.TEAMID,
    a.DAILYCAPACITY
FROM TICKETS t
LEFT JOIN CLIENTS c ON t.CLIENTID = c.CLIENTID
LEFT JOIN AGENTS a ON t.ASSIGNEDAGENTID = a.AGENTID
WHERE t.SLABREACHED IS NOT NULL
"""

print("📥 Fetching joined data...")
df = pd.read_sql(query, conn)
conn.close()

# ---- SAVE LOCALLY ----
df.to_csv("joined_data.csv", index=False)

print(f"✅ Loaded {len(df)} rows")
print(f"✅ Columns: {list(df.columns)}")
print(f"✅ Breach rate: {df['SLABREACHED'].mean():.1%}")
print("\n✅ STEP 1 COMPLETE → saved as joined_data.csv")