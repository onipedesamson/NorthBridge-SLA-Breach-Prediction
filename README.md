# 🏥 NorthBridge Health Services — SLA Breach Prediction

> **Machine Learning · Cloud Pipeline · Predictive Operations**

---

## 1. Demand Is Uneven — Breach Risk Is Not Uniform

Average ticket response time is **38 hours** against an 8‑hour target. The distribution is heavily imbalanced: most tickets are resolved on time, but a significant fraction breach — and Priority 1 tickets face the highest risk.

*(Graph: `feature_importance_bar.png`)*

**Business meaning**
NorthBridge cannot treat all tickets equally. Averaged metrics hide critical concentration of risk — Priority 1 tickets, open backlog, and high‑workload agents need targeted intervention.

---

## 2. Site & Team Behaviour Is Not Uniform

Breach rates differ substantially across teams, hubs, and priority levels — a single company‑wide average would hide meaningful operational differences.

*(Graph: `graph2_tickets_by_day.png`)*

**Business meaning**
Forecasting and workload balancing must be done at the team and priority level — not globally. Risk is concentrated in specific hubs and priority tiers.

---

## 3. Key Drivers of Breach Risk

Feature importance reveals which factors most strongly predict whether a ticket will miss its deadline:

| Factor | Influence Level |
|---|---|
| 🟢 Ticket Status (Open → highest risk) | **Highest** |
| 🟡 Priority Level (P1 > P2 > P3) | High |
| 🟠 Response Time Elapsed | Medium |
| 🔴 Agent Daily Workload | Medium |
| ⚫ Client Contract Tier | Lower |

*(Graph: `step4_graph2_feature_importance.png`)*

**Business meaning**
Closing open tickets is the single most effective action to reduce breaches — more impactful than any other change. Priority 1 tickets must be routed to fastest‑response agents.

---

## 4. Baseline vs Production — Model Performance

A simple baseline (Logistic Regression) is compared against the production XGBoost model:

*(Graph: `step4_graph3_performance.png`)*

| Metric | Baseline | Production (XGBoost) |
|---|---|---|
| **Recall** (Detecting real breaches) | Low | ✅ **Greatly Improved** |
| **Precision** | — | ✅ Balanced |
| **ROC‑AUC** | 0.52 | ✅ **0.89** |

**Business meaning**
The baseline effectively guessed "no breach" — useless for decision‑making. The production model reliably identifies at‑risk tickets early enough to intervene.

---

## 5. Confusion Matrix — Production Model Results

*(Graph: `step4_graph1_confusion_matrix.png`)*

**Business meaning**
- ✅ **True Negatives** — Safe tickets correctly identified
- ✅ **True Positives** — At‑risk tickets flagged early
- ⚠️ **False Negatives** — Kept to a minimum (missed risks are costly)
- ⚠️ **False Positives** — Balanced to avoid alert fatigue

---

## 6. Temporal Patterns — When Do Breaches Occur?

*(Graph: `graph3_weekday_vs_weekend.png`)*

Notable patterns:
- 📅 **Weekdays** — Highest volume and highest breach risk
- ⏰ **Monday → Wednesday** — Peak breach probability
- 📉 **Weekends** — Lower volume but longer response delays

**Business meaning**
Staffing and escalation protocols must match weekday peaks — weekend coverage needs review for delayed resolution.

---

## 7. Cloud Architecture — AWS Native Solution
