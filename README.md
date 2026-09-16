# 🏥 NorthBridge Health Services — SLA Breach Prediction Model

> **From Reactive Tracking → Proactive Prevention**
> Machine learning classification model that predicts which customer service tickets will **BREACH** their Service Level Agreement — *before* the deadline is missed. Helps NorthBridge avoid £214,000/year in penalties.

---

## 📊 At a Glance

| Metric | Figure |
|---|---|
| 📈 Tickets Analysed | **3,500+** |
| ⚠️ SLA Breaches | **754 (~21.5%)** |
| 🎯 Problem Type | Binary Classification — Will this ticket breach? (1=Yes / 0=No) |
| 🛠️ Tech Stack | AWS S3 → Airbyte → Snowflake → Python/XGBoost → Power BI |
| 📅 Production Model | XGBoost / LightGBM Gradient Boosting |
| 📍 Location | Manchester, UK |

---

## ☁️ AWS Cloud Architecture

Your solution runs on **industry-standard AWS infrastructure** — scalable, secure, and production-ready:

| AWS Service | Role in Your Project |
|---|---|
| 📦 **Amazon S3** | Raw data landing zone — all ticket files stored securely, organised by source & date |
| ⚡ **Amazon EC2** *(Optional)* | Compute layer — where Python scripts run daily — reliable, always-on |
| 🔄 **Airbyte** *(on EC2 or local)* | Automatically syncs data from S3 → Snowflake |
| ❄️ **Snowflake** *(Cloud Data Warehouse)* | Built for cloud — stores RAW + REPORTING schemas, scales automatically |
| 🔐 **Security & IAM** | Controlled access — compliant with healthcare data standards |
| 📈 **Cloud-Native Benefit** | Grows as NorthBridge grows — thousands more tickets without slowdowns |

> **Why AWS?** Enterprise reliability → 99.99% uptime → compliant → ready for real-world deployment

---

## 🔄 Full End-to-End Pipeline
