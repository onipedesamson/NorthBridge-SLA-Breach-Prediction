"""
NorthBridge SLA Breach Prediction — PowerPoint Generator
Run this file → it creates NorthBridge_SLA_Presentation.pptx automatically!
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# ==================================================
# CREATE NEW PRESENTATION
# ==================================================
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

def add_title_slide(title, subtitle):
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    return slide

def add_content_slide(title, content, image_note=""):
    slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.text = content
    if image_note:
        p = tf.add_paragraph()
        p.text = f"\n📷 INSERT IMAGE: {image_note}"
        p.font.bold = True
        p.font.color.rgb = RGBColor(0, 102, 204)
    return slide

# ==================================================
# SLIDE 1 — TITLE
# ==================================================
add_title_slide(
    "🩺 NorthBridge Health Services\nSLA Breach Prediction",
    "Machine Learning Solution\nManchester, United Kingdom | September 2026"
)

# ==================================================
# SLIDE 2 — PROJECT OVERVIEW
# ==================================================
add_content_slide("📋 Project Overview", """
NorthBridge Health Services is a UK-based healthcare administration company
founded in 2008. Growing ticket volumes made manual SLA tracking unreliable.

✅ WHAT THIS PROJECT DELIVERS:
• End-to-end data pipeline: Load → Clean → Engineer → Model
• Baseline Model (Logistic Regression) — performance benchmark
• Production Model (XGBoost) — improved accuracy — FINAL MODEL
• 10 Visualisations — full insight into breach patterns
• All code & models documented on GitHub
""")

# ==================================================
# SLIDE 3 — BUSINESS PROBLEM
# ==================================================
add_content_slide("⚠️ Business Problem", """
❌ Manual SLA tracking — reactive, slow, error-prone
❌ 2,800+ unresolved tickets in backlog
❌ Average response = 38 hours   |   Priority 1 Target = 8 hours
❌ Data fragmented across CRM, Excel, emails
❌ Financial penalties from repeated breaches

✅ WHAT WAS NEEDED:
• Centralised data
• Real-time monitoring
• PREDICTIVE ALERTING — flag at-risk tickets EARLY
""")

# ==================================================
# SLIDE 4 — SOLUTION APPROACH
# ==================================================
add_content_slide("🏗️ Solution Approach", """
STEP 1: DATA LOADING          → Load, validate, clean, merge
       ↓
STEP 2: FEATURE ENGINEERING    → Time patterns + 4 trend graphs
       ↓
STEP 3: BASELINE MODEL         → Logistic Regression — Benchmark
       ↓
STEP 4: PRODUCTION MODEL       → XGBoost — Beat Benchmark → FINAL

🎯 PREDICTION TARGET:
• Priority 1 SLA = Response within 8 hours
• BREACH (1) = Less than 8 hours available
• ON-TIME (0) = 8+ hours available
""")

# ==================================================
# SLIDE 5 — TICKET VOLUME BY HOUR
# ==================================================
add_content_slide("📊 Ticket Volume by Hour", """
KEY INSIGHTS:
• ⏰ Peak Hours: 09:00–11:00 and 14:00–16:00
• 🌙 Low Hours: 00:00–06:00
• ⚠️ CRITICAL RISK: Tickets created AFTER 4:00 PM → less than
  8 working hours remain → HIGH probability of breach
""", "graph1_tickets_by_hour.png")

# ==================================================
# SLIDE 6 — TICKET VOLUME BY DAY
# ==================================================
add_content_slide("📊 Ticket Volume by Day of Week", """
KEY INSIGHTS:
• 📈 Monday–Wednesday: Highest volume → Weekend backlog
• 📉 Thursday–Sunday: Gradually declines
• ⚠️ FRIDAY RISK: Tickets Friday afternoon → SLA expires Monday
  → 3-day gap = HIGH RISK
""", "graph2_tickets_by_day.png")

# ==================================================
# SLIDE 7 — WEEKDAY vs WEEKEND
# ==================================================
add_content_slide("📊 Weekday vs Weekend Risk", """
KEY INSIGHTS:
• 📉 Weekend tickets = fewer in NUMBER
• ⚠️ BUT: SLA clock runs 24/7 → staff away → HIGHER breach risk
• 💡 ACTION: Weekend tickets flagged as HIGH-RISK category
""", "graph3_weekday_vs_weekend.png")

# ==================================================
# SLIDE 8 — MONTHLY TRENDS
# ==================================================
add_content_slide("📊 Monthly Volume Trends", """
KEY INSIGHTS:
• 📈 Quarter-End Peaks: Mar, Jun, Sep, Dec → Reporting pressure
• 📉 Summer Dip: July–August → then Q4 surge
• 💡 BUSINESS VALUE: Staffing forecasts — prepare for peaks
""", "graph4_tickets_by_month.png")

# ==================================================
# SLIDE 9 — BASELINE CONFUSION MATRIX
# ==================================================
add_content_slide("📊 Baseline Model Performance", """
HOW TO READ THIS MATRIX:

                  PREDICTED ON-TIME    |    PREDICTED BREACH
    ACTUAL
     On-Time           ✅ CORRECT       |       ❌ FALSE ALARM
     Breach            ❌ MISSED       |        ✅ CAUGHT

    💡 GOAL: MINIMISE BOTTOM-LEFT CELL → NEVER MISS A REAL BREACH
""", "step3_graph1_confusion_matrix.png")

# ==================================================
# SLIDE 10 — BASELINE FEATURE IMPORTANCE
# ==================================================
add_content_slide("📈 Top Influential Factors — Baseline", """
TOP RISK FACTORS IDENTIFIED:
    1. ⏰ Hour of Creation    → Late day = less time = higher risk
    2. 📅 Day of Week         → Friday/weekend = high risk
    3. 🔴 Priority Level      → P1 = tightest deadline
    4. 🏢 Contract Tier        → Higher tiers = stricter SLA
    5. 📊 Monthly Patterns     → End-of-period volume spikes
""", "step3_graph2_feature_importance.png")

# ==================================================
# SLIDE 11 — BASELINE PERFORMANCE
# ==================================================
add_content_slide("📊 Baseline Model Scorecard", """
BENCHMARK ESTABLISHED:
    • Accuracy   → Overall % correct predictions
    • Precision  → % flagged that ACTUALLY breach
    • Recall     → % of REAL breaches successfully caught
    • ROC-AUC    → Overall predictive power (higher = better)

    🎯 THIS IS THE BENCHMARK — STEP 4 MUST BEAT THESE!
""", "step3_graph3_performance.png")

# ==================================================
# SLIDE 12 — PRODUCTION CONFUSION MATRIX
# ==================================================
add_content_slide("📊 Production Model — XGBoost Performance", """
IMPROVEMENT vs BASELINE:
    • ✅ Fewer missed breaches     → Bottom-Left cell ↓
    • ✅ Fewer false alarms        → Top-Right cell ↓
    • ✅ More correct predictions  → Diagonal cells ↑
    • 🏆 XGBoost = CLEAR IMPROVEMENT across ALL metrics
""", "step4_graph1_confusion_matrix.png")

# ==================================================
# SLIDE 13 — PRODUCTION FEATURE IMPORTANCE
# ==================================================
add_content_slide("📈 Top Breach Risk Factors — Final", """
🎯 FINAL TOP 5 BREACH RISK FACTORS:

    RANK   FACTOR                  WHY IT MATTERS
    ───────────────────────────────────────────────────────
     1     ⏰ Creation Hour        Late day = less time to respond
     2     📅 Day of Week          Fri/Weekend = SLA expires while away
     3     🔴 Priority Level       P1 = 8hr deadline = NO margin for error
     4     🏢 Contract Tier        Higher tiers = stricter requirements
     5     📊 Month/Quarter        End-of-period = volume surge
""", "step4_graph2_feature_importance.png")

# ==================================================
# SLIDE 14 — PRODUCTION PERFORMANCE
# ==================================================
add_content_slide("📊 Production Model Scorecard — XGBoost", """
✅ XGBOOST RESULTS:
    • Accuracy   → ✅ HIGH
    • Precision  → ✅ RELIABLE ALERTS
    • Recall     → ✅ MOST BREACHES CAUGHT
    • ROC-AUC    → ✅ EXCELLENT PREDICTIVE POWER
""", "step4_graph3_performance.png")

# ==================================================
# SLIDE 15 — MODEL COMPARISON
# ==================================================
add_content_slide("🏆 Baseline vs Production — Comparison", """
METRIC          BASELINE (Logistic)    PRODUCTION (XGBoost)    RESULT
──────────────────────────────────────────────────────────────────────
Accuracy        Benchmark              ✅ HIGHER               ⬆️ IMPROVED
Precision       Benchmark              ✅ HIGHER               ⬆️ FEWER ALARMS
Recall          Benchmark              ✅ HIGHER               ⬆️ FEWER MISSED
ROC-AUC         Baseline Score         ✅ HIGHER               🏆 BETTER MODEL

✅ CONCLUSION: XGBoost BEATS Baseline on ALL metrics → SELECTED
   as the PRODUCTION MODEL
""")

# ==================================================
# SLIDE 16 — KEY FINDINGS
# ==================================================
add_content_slide("💡 Key Findings & Business Impact", """
🎯 TOP 3 BREACH RISK FACTORS:
    1. ⏰ Tickets created LATE IN THE DAY → insufficient working hours
    2. 📅 FRIDAY & WEEKEND tickets → SLA clock runs while staff away
    3. 🔴 PRIORITY 1 tickets → 8hr deadline = zero margin for error

💰 BUSINESS IMPACT:
    • ✅ PREDICT BEFORE BREACH → reduce financial penalties
    • ✅ SCHEDULE SMARTER → know peak hours, days, months
    • ✅ FOCUS INTERVENTION → top 3 factors = where to act
    • ✅ FULL VISIBILITY → 10 charts = complete transparency
""")

# ==================================================
# SLIDE 17 — DELIVERABLES
# ==================================================
add_content_slide("✅ Project Complete — Deliverables", """
STEP     DELIVERABLE                          STATUS
───────────────────────────────────────────────────────────
Step 1   Data Loading & Validation            ✅ DONE
Step 2   Features + 4 Trend Graphs            ✅ DONE
Step 3   Baseline Model + 3 Graphs            ✅ DONE
Step 4   XGBoost Model + 3 Graphs            ✅ DONE
💾 production_model.pkl — FINAL MODEL         ✅ SAVED
📄 All Code + Metrics — GitHub Repository     ✅ PUSHED & LIVE

🚀 NEXT STEPS:
    • Connect to Power BI → Real-time alerts
    • Integrate with Live System → Daily predictions
    • Add SHAP Explainability → Show WHY each ticket flagged
""")

# ==================================================
# SLIDE 18 — THANK YOU
# ==================================================
add_title_slide("🙏 Thank You", """
🌐 GITHUB REPOSITORY:
https://github.com/onipedesamson/NorthBridge-SLA-Breach-Prediction

📁 CONTAINS:
    • ✅ All Python Code (Step 1–4)
    • ✅ All 10 Visualisation Graphs
    • ✅ Final Production Model File
    • ✅ Metrics & Performance Scores
    • ✅ Professional Documentation

"From Reactive Monitoring → Proactive Prediction"
""")

# ==================================================
# SAVE THE PRESENTATION
# ==================================================
FILENAME = "NorthBridge_SLA_Presentation.pptx"
prs.save(FILENAME)
print("=" * 70)
print(f"✅ PRESENTATION CREATED SUCCESSFULLY!")
print(f"📁 FILE SAVED AS: {FILENAME}")
print("=" * 70)
print("\n📷 NEXT STEP: Open the file in PowerPoint → Insert your 10 graph images")
print("   where marked in BLUE TEXT!")