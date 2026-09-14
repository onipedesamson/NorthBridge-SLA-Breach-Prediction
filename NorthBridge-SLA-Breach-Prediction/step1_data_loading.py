"""
Step 1 — Load Data from EXCEL FILE
NorthBridge Health Services — SLA Breach Prediction
"""

import pandas as pd
import os

print("=" * 60)
print("STEP 1: LOADING DATA FROM EXCEL FILE")
print("=" * 60)

# ---- YOUR EXCEL FILE & SHEET NAMES ----
excel_file = "data/data.xlsx.xlsx"

# ✅ YOUR EXACT SHEET NAMES FROM YOUR FILE
sheet_tickets = "N-Bridge xlsx - Tickets"
sheet_clients = "N-Bridge xlsx - Clients"
sheet_agents = "N-Bridge xlsx - Agents"

# Check if file exists
if not os.path.exists(excel_file):
    print(f"❌ ERROR: Cannot find file: {excel_file}")
    exit(1)

print(f"✅ Found Excel file: {excel_file}")

# ---- READ ALL 3 SHEETS ----
tickets = pd.read_excel(excel_file, sheet_name=sheet_tickets)
clients = pd.read_excel(excel_file, sheet_name=sheet_clients)
agents = pd.read_excel(excel_file, sheet_name=sheet_agents)

print(f"✅ Tickets:   {len(tickets):>6} rows")
print(f"✅ Clients:   {len(clients):>6} rows")
print(f"✅ Agents:    {len(agents):>6} rows")

# ---- JOIN ALL 3 TABLES ----
tickets = tickets.merge(clients, on="ClientID", how="left")
tickets = tickets.merge(agents, left_on="AssignedAgentID", right_on="AgentID", how="left")

print(f"\n✅ Joined dataset: {len(tickets)} rows")
print(f"✅ Total columns: {len(tickets.columns)}")

# ---- SAVE FOR STEP 2 ----
tickets.to_csv("joined_data.csv", index=False)
print("\n✅ STEP 1 COMPLETE → saved as joined_data.csv")
print("✅ NOW RUN: python step2_feature_engineering.py")