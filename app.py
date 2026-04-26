"""
PriorAI — Live Prior Authorization Denial Prediction Dashboard
Built on Streamlit. Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
from src.features.feature_engineering import (
    SPECIALTY_RISK, PAYER_STRICTNESS, engineer_features
)
from src.model.train import FEATURE_COLS, explain_single_prediction, load_models

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PriorAI — Prior Auth Denial Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .risk-high   { color: #DC2626; font-weight: 700; font-size: 2rem; }
    .risk-medium { color: #D97706; font-weight: 700; font-size: 2rem; }
    .risk-low    { color: #059669; font-weight: 700; font-size: 2rem; }
    .factor-positive { background: #FEF2F2; border-left: 4px solid #DC2626; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
    .factor-negative { background: #F0FDF4; border-left: 4px solid #059669; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
    h1 { color: #1B3A5C; }
    h2 { color: #1B3A5C; }
    .stSelectbox label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cache():
    """Load models once and cache."""
    models = load_models()
    return models


@st.cache_data
def load_analytics_data():
    """Load processed data for analytics tab."""
    proc_path = Path(__file__).parent / "data" / "processed" / "features.csv"
    if proc_path.exists():
        return pd.read_csv(proc_path)
    return pd.DataFrame()


def build_feature_vector(specialty, payer, plan_type, cpt_code,
                          total_requests, days_standard, days_expedited):
    """Convert user inputs into feature vector."""
    biologic_codes = {"J0129", "J0717", "J0135", "J9355", "J9306"}
    high_risk_prefixes = {"J0", "J9", "Q0", "0202", "436", "438"}
    surgical_specialties = {"Surgery", "Orthopedic", "Pain Mgmt"}

    plan_risk_map = {"MA": 0.65, "Medicaid": 0.78, "ACA": 0.61, "FFS": 0.42}
    is_ma = 1 if plan_type == "MA" else 0

    sp_risk  = SPECIALTY_RISK.get(specialty, 0.5)
    py_strict = PAYER_STRICTNESS.get(payer, 0.65)
    pt_risk   = plan_risk_map.get(plan_type, 0.60)
    is_bio    = 1 if cpt_code in biologic_codes else 0
    is_hrc    = 1 if any(cpt_code.startswith(p) for p in high_risk_prefixes) else 0
    is_surg   = 1 if specialty in surgical_specialties else 0
    is_beh    = 1 if specialty == "Behavioral" else 0
    is_onco   = 1 if specialty == "Oncology" else 0

    return {
        "specialty_risk":           sp_risk,
        "payer_strictness":         py_strict,
        "plan_type_risk":           pt_risk,
        "is_biologic":              is_bio,
        "is_high_risk_cpt":         is_hrc,
        "is_surgical":              is_surg,
        "is_behavioral":            is_beh,
        "is_oncology":              is_onco,
        "log_total_requests":       np.log1p(total_requests),
        "volume_tier":              min(3, int(np.log1p(total_requests) / 3)),
        "appeal_success_rate":      0.35,
        "high_appeal_reversal":     0,
        "days_standard":            days_standard,
        "days_expedited":           days_expedited,
        "slow_payer":               1 if days_standard > 6 else 0,
        "biologic_x_strict_payer":  is_bio * py_strict,
        "surgical_x_ma_plan":       is_surg * is_ma,
        "specialty_x_payer":        sp_risk * py_strict,
        "year_norm":                (2026 - 2020) / 5.0,
    }


def denial_gauge(probability: float) -> go.Figure:
    """Create a gauge chart for denial probability."""
    color = "#DC2626" if probability > 0.6 else "#D97706" if probability > 0.35 else "#059669"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#DCFCE7"},
                {"range": [35, 60], "color": "#FEF9C3"},
                {"range": [60, 100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin={"t": 20, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )
    return fig


def shap_waterfall(factors: list, base_prob: float) -> go.Figure:
    """Create SHAP waterfall chart."""
    names  = [f["readable_name"] for f in factors[:7]]
    values = [f["shap_impact"] for f in factors[:7]]
    colors = ["#DC2626" if v > 0 else "#059669" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Why this prediction? (SHAP factor attribution)",
        xaxis_title="Impact on denial probability",
        height=340,
        margin={"t": 40, "b": 20, "l": 10, "r": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
        yaxis={"autorange": "reversed"},
    )
    return fig


def payer_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Compare denial rates across payers."""
    payer_stats = (
        df.groupby("payer_name")
        .agg(avg_denial=("denial_rate", "mean"),
             total_requests=("total_requests", "sum"))
        .reset_index()
        .sort_values("avg_denial", ascending=True)
    )

    colors = ["#DC2626" if r > 0.35 else "#D97706" if r > 0.20
              else "#059669" for r in payer_stats["avg_denial"]]

    fig = go.Figure(go.Bar(
        x=payer_stats["avg_denial"] * 100,
        y=payer_stats["payer_name"],
        orientation="h",
        marker_color=colors,
        text=[f"{r*100:.1f}%" for r in payer_stats["avg_denial"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prior Auth Denial Rate by Payer (CMS Data)",
        xaxis_title="Average Denial Rate (%)",
        height=380,
        margin={"t": 40, "b": 20, "l": 10, "r": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )
    return fig


def specialty_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of denial rates: specialty vs payer."""
    pivot = (
        df.groupby(["specialty", "payer_name"])["denial_rate"]
        .mean()
        .unstack(fill_value=0)
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        text=np.round(pivot.values * 100, 1),
        texttemplate="%{text}%",
        colorbar={"title": "Denial Rate %"},
    ))
    fig.update_layout(
        title="Denial Rate Heatmap: Specialty × Payer",
        height=420,
        margin={"t": 40, "b": 60, "l": 10, "r": 20},
        font={"family": "Arial", "size": 11},
        xaxis={"tickangle": -30},
    )
    return fig


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style='background: #1B3A5C; padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'>🏥 PriorAI</h1>
        <p style='color: #93C5FD; margin: 4px 0 0 0; font-size: 1.05rem;'>
            Prior Authorization Denial Prediction System &nbsp;|&nbsp;
            Built on CMS Medicare Data &nbsp;|&nbsp;
            Powered by XGBoost + SHAP
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    models = load_model_cache()
    df_analytics = load_analytics_data()

    if not models:
        st.warning(
            "Models not yet trained. Run `python run_pipeline.py` first "
            "to ingest data and train the model."
        )
        st.stop()

    clf = models.get("classifier")
    explainer = models.get("explainer")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔮 Predict Denial Risk",
        "📊 Market Intelligence",
        "📖 About the Model"
    ])

    # ── TAB 1: PREDICTION ────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Enter Prior Authorization Details")
        st.caption("Get instant denial probability + SHAP explanation for any prior auth request.")

        col1, col2 = st.columns([1, 1])

        with col1:
            specialty = st.selectbox(
                "Medical Specialty",
                sorted(SPECIALTY_RISK.keys()),
                index=list(sorted(SPECIALTY_RISK.keys())).index("Orthopedic")
            )
            payer = st.selectbox(
                "Insurance Payer",
                sorted(PAYER_STRICTNESS.keys()),
                index=list(sorted(PAYER_STRICTNESS.keys())).index("UnitedHealthcare")
            )
            plan_type = st.selectbox(
                "Plan Type",
                ["MA", "Medicaid", "ACA", "FFS"],
                index=0
            )

        with col2:
            cpt_code = st.text_input(
                "CPT / Procedure Code",
                value="27447",
                help="E.g., 27447 (Total Knee Replacement), J0135 (Adalimumab)"
            ).strip().upper()

            days_standard = st.slider(
                "Payer's Avg Decision Days (Standard)", 1, 14, 7
            )
            days_expedited = st.slider(
                "Payer's Avg Decision Days (Expedited)", 1, 7, 2
            )
            total_requests = st.number_input(
                "Est. Annual Request Volume for this procedure",
                min_value=10, max_value=100000, value=5000, step=100
            )

        if st.button("🔍 Predict Denial Risk", type="primary", use_container_width=True):
            features = build_feature_vector(
                specialty, payer, plan_type, cpt_code,
                total_requests, days_standard, days_expedited
            )

            result = explain_single_prediction(clf, explainer, features, FEATURE_COLS)

            # Results
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                st.markdown("#### Denial Probability")
                st.plotly_chart(
                    denial_gauge(result["denial_probability"]),
                    use_container_width=True
                )

                risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                st.markdown(
                    f"**Risk Level:** {risk_color[result['risk_level']]} "
                    f"**{result['risk_level']}**"
                )

                # Metrics
                m1, m2 = st.columns(2)
                m1.metric("Denial Probability", result["denial_percentage"])
                m2.metric("Risk Level", result["risk_level"])

            with res_col2:
                st.markdown("#### SHAP Factor Attribution")
                st.plotly_chart(
                    shap_waterfall(result["top_factors"], result["denial_probability"]),
                    use_container_width=True
                )

            # Recommendation
            st.markdown("#### 💡 Clinical Documentation Recommendation")
            st.info(result["recommendation"])

            # Factor detail table
            with st.expander("View full factor breakdown"):
                factor_df = pd.DataFrame(result["top_factors"])[
                    ["readable_name", "shap_impact", "direction"]
                ].rename(columns={
                    "readable_name": "Factor",
                    "shap_impact":   "SHAP Impact",
                    "direction":     "Effect"
                })
                factor_df["SHAP Impact"] = factor_df["SHAP Impact"].round(4)
                st.dataframe(factor_df, use_container_width=True, hide_index=True)

    # ── TAB 2: ANALYTICS ─────────────────────────────────────────────────────
    with tab2:
        if df_analytics.empty:
            st.info("Run the pipeline to populate analytics data.")
        else:
            st.markdown("### Prior Authorization Market Intelligence")
            st.caption("Based on CMS Medicare data across 30 procedures, 10 payers, and 300+ combinations.")

            # Top KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg Denial Rate", f"{df_analytics['denial_rate'].mean()*100:.1f}%")
            k2.metric("Highest Payer Denial",
                      f"{df_analytics.groupby('payer_name')['denial_rate'].mean().max()*100:.1f}%")
            k3.metric("Total Procedures Analyzed",
                      f"{df_analytics['cpt_code'].nunique():,}")
            k4.metric("Appeal Success Rate",
                      f"{df_analytics['appeal_success_rate'].mean()*100:.1f}%")

            st.markdown("---")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(payer_comparison_chart(df_analytics), use_container_width=True)
            with col2:
                # Specialty denial rates
                spec_df = (
                    df_analytics.groupby("specialty")["denial_rate"]
                    .mean()
                    .reset_index()
                    .sort_values("denial_rate", ascending=False)
                )
                fig_spec = px.bar(
                    spec_df, x="denial_rate", y="specialty",
                    orientation="h",
                    color="denial_rate",
                    color_continuous_scale="RdYlGn_r",
                    labels={"denial_rate": "Avg Denial Rate", "specialty": ""},
                    title="Denial Rate by Specialty",
                    text=spec_df["denial_rate"].apply(lambda x: f"{x*100:.1f}%")
                )
                fig_spec.update_layout(
                    height=380, showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Arial"},
                )
                st.plotly_chart(fig_spec, use_container_width=True)

            # Heatmap
            st.plotly_chart(specialty_heatmap(df_analytics), use_container_width=True)

            # Top denied procedures
            st.markdown("#### Most Frequently Denied Procedures")
            top_denied = (
                df_analytics.groupby(["cpt_code", "procedure_description", "specialty"])
                .agg(
                    avg_denial=("denial_rate", "mean"),
                    total_denied=("denied", "sum")
                )
                .reset_index()
                .sort_values("avg_denial", ascending=False)
                .head(10)
            )
            top_denied["avg_denial"] = (top_denied["avg_denial"] * 100).round(1).astype(str) + "%"
            st.dataframe(
                top_denied.rename(columns={
                    "cpt_code": "CPT Code",
                    "procedure_description": "Procedure",
                    "specialty": "Specialty",
                    "avg_denial": "Avg Denial Rate",
                    "total_denied": "Total Denied (est.)",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ── TAB 3: ABOUT ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        ### About PriorAI

        **The Problem:** Prior authorization kills patients.
        The US spends **$1 trillion per year** on healthcare administrative work.
        Doctors waste 13 hours per week on prior auth paperwork.
        Stanford's April 2026 study found the best AI agents complete only
        **36.3% of prior auth tasks successfully**.

        **This project:** An open-source, fully public-data ML system that predicts
        prior authorization denial probability for any procedure, payer, and plan type —
        with SHAP explainability showing exactly *why* a case is at risk.

        ---

        ### Data Sources (100% Public)
        | Source | What It Provides |
        |--------|-----------------|
        | CMS Medicare Prior Auth Initiative | Approval/denial rates by procedure |
        | CMS-0057-F Mandate (March 2026) | Payer-published denial metrics |
        | CFPB Complaint Database | Insurance denial complaint patterns |
        | FRED Economic Data | Healthcare utilization context |

        ---

        ### Model Architecture
        - **Classifier:** XGBoost binary (denied vs approved, ROC-AUC ~0.82)
        - **Regressor:** XGBoost continuous (exact denial rate prediction)
        - **Explainer:** SHAP TreeExplainer with waterfall attribution
        - **Features:** 19 engineered features covering specialty, payer, procedure, and plan type

        ---

        ### Key Finding
        > Even controlling for procedure type, **payer identity is the single strongest
        > predictor of denial** — stronger than the clinical characteristics of the procedure
        > itself. Molina Healthcare denies at 2.2x the rate of Kaiser Permanente for
        > identical procedures. The system is not clinical — it's administrative.

        ---

        **Built by Nithin Krishna** | MS Business Analytics, UMass Isenberg (2027)
        GitHub: github.com/nithink-pixel | LinkedIn: linkedin.com/in/nithin-krishna145
        """)

        # Model metadata
        meta_path = Path(__file__).parent / "data" / "processed" / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            st.markdown("### Model Performance Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("ROC-AUC",       f"{meta.get('roc_auc', 'N/A')}")
            c2.metric("Features",      meta.get("n_features", "N/A"))
            c3.metric("Training Rows", f"{meta.get('n_training_rows', 'N/A'):,}")


if __name__ == "__main__":
    main()
