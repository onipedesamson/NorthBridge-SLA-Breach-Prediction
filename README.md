# 🏥 NorthBridge Health Services — SLA Breach Prediction Model

## 📋 Project Overview
Machine learning classification model that predicts which customer service tickets will BREACH their Service Level Agreement — BEFORE it happens. Helps NorthBridge avoid £214,000/year in penalties.

## 🎯 Problem Type
Binary Classification → **Will this ticket breach SLA? (1=Yes / 0=No)**

## 📊 Data Source
Snowflake REPORTING Schema — joined from:
- **Tickets** → Priority, Category, Channel, CreatedAt
- **Clients** → ContractTier, SLACreditClause
- **Agents** → Hub, TeamID, DailyCapacity

## 🤖 Models
- **Baseline:** Logistic Regression
- **Production:** XGBoost / LightGBM Gradient Boosting

## 📈 Metrics
Precision, Recall, ROC-AUC

## 📁 Structure