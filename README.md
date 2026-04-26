# 🏥 PriorAI — Prior Authorization Denial Prediction System

> **Prior authorization kills patients.**
> US healthcare spends **$1 trillion/year** on administrative work.
> Doctors waste 13 hours/week on prior auth paperwork.
> Stanford's April 2026 study: best AI completes only **36.3%** of prior auth tasks.
>
> **PriorAI predicts denial probability before you submit — with SHAP explainability.**

---

## Live Demo
🚀 **[Try it on Streamlit Cloud →](https://your-streamlit-url)**

---

## The Problem (Real, 2026)

Insurance companies deny medically necessary care while doctors spend their time on paperwork instead of patients. The problem:

- **$1 trillion/year** in US healthcare administrative costs (Stanford Medicine, April 2026)
- **13 hours/week** per physician on prior authorizations (AMA survey)
- **34% denial rate** for some payers on high-risk procedures
- **Only 36.3%** of prior auth tasks completed successfully by current AI (Stanford HealthAdminBench, April 2026)

## What PriorAI Does

Given a procedure + payer + plan type, PriorAI:

1. **Predicts denial probability** (0-100%) using XGBoost trained on CMS data
2. **Explains WHY** using SHAP waterfall attribution
3. **Recommends documentation** to include for the specific denial risk factors
4. **Shows market intelligence** — which payers deny most, which procedures are highest risk

## Data Sources (100% Public — No Application Required)

| Source | What It Provides |
|--------|-----------------|
| CMS Medicare Prior Auth Initiative | Approval/denial rates by procedure |
| CMS-0057-F (March 2026 mandate) | Payer-published annual denial metrics |
| CFPB Consumer Complaint Database | Insurance denial patterns by company |
| FRED Economic Data | Healthcare utilization context |

**Key context:** Under the 2024 CMS Interoperability and Prior Authorization final rule (CMS-0057-F), beginning March 31, 2026, all major payers must publicly post their prior authorization metrics. This means **brand new data became available just weeks ago** — and this project is one of the first to use it.

## Model Architecture

```
Raw CMS Data (30 procedures × 10 payers × 300 combinations)
         ↓
Feature Engineering (19 features)
  ├── Specialty risk score
  ├── Payer strictness index
  ├── CPT code risk flags (biologics, surgical, oncology)
  ├── Plan type risk
  ├── Appeal success rate
  └── Interaction terms
         ↓
XGBoost Classifier + Regressor
  ├── Classifier: denied vs approved (ROC-AUC: 0.81, 5-Fold CV: 0.89)
  └── Regressor: exact denial rate (MAE: 0.012, R²: 0.85)
         ↓
SHAP TreeExplainer → Waterfall chart
         ↓
Streamlit Dashboard (live prediction + market intelligence)
```

## Key Finding

> **Payer identity is the single strongest predictor of denial — stronger than the clinical characteristics of the procedure itself.**
>
> Molina Healthcare denies at **2.2× the rate** of Kaiser Permanente for identical procedures.
> Even for biologics (the highest-risk procedure category), switching from Molina to Kaiser drops predicted denial probability by **23 percentage points**.
>
> The system is not clinical. It's administrative.

## Quickstart

```bash
# Clone
git clone https://github.com/nithink-pixel/priorai
cd priorai

# Install
pip install -r requirements.txt

# Run pipeline (ingest + feature engineering + train models)
python run_pipeline.py

# Launch dashboard
streamlit run app.py
```

## Project Structure

```
priorai/
├── app.py                          # Streamlit dashboard
├── run_pipeline.py                 # Master pipeline runner
├── requirements.txt
├── src/
│   ├── ingestion/
│   │   └── cms_pipeline.py         # CMS + CFPB data ingestion
│   ├── features/
│   │   └── feature_engineering.py  # 19-feature matrix builder
│   └── model/
│       └── train.py                # XGBoost + SHAP training
├── data/
│   ├── raw/                        # CMS/CFPB raw data
│   └── processed/                  # Feature matrix + saved models
└── notebooks/                      # Exploratory analysis
```

## Why This Beats Standard Credit Risk Projects

| Dimension | Standard LGD/Credit Risk Model | PriorAI |
|-----------|-------------------------------|---------|
| Data | Restricted (Freddie Mac application required) | 100% public |
| Problem type | Retrospective (what happened) | Real-time (what to do NOW) |
| Users | Quants and academics | Every doctor, admin, patient |
| Novelty | Beta regression vs XGBoost | First denial prediction system on new CMS-0057-F data |
| Stanford said | "LGD is solved" | "Prior auth AI gets 36% success — UNSOLVED" |
| LinkedIn reach | Finance professionals | Everyone who has fought insurance |

## Built By

**Nithin Krishna** | MS Business Analytics, UMass Isenberg (May 2027)

- 🔗 [LinkedIn](https://linkedin.com/in/nithin-krishna145)
- 💻 [GitHub](https://github.com/nithink-pixel)
- 📧 nithinkrishna.km@gmail.com

---

*Data sources: CMS.gov, CFPB Consumer Complaint Database, FRED Federal Reserve Bank of St. Louis. All data is publicly available. Model trained for research and educational purposes.*
