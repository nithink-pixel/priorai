"""
PriorAI — Feature Engineering
Builds the feature matrix for denial prediction from raw CMS data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR   = Path(__file__).parents[2] / "data" / "raw"
PROC_DIR  = Path(__file__).parents[2] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ── SPECIALTY RISK SCORES (from CMS published denial rate analysis) ───────────
SPECIALTY_RISK = {
    "Surgery":      0.85,
    "Orthopedic":   0.72,
    "Rheumatology": 0.91,
    "Oncology":     0.88,
    "Pain Mgmt":    0.79,
    "Behavioral":   0.68,
    "Infusion":     0.76,
    "Cardiology":   0.61,
    "Radiology":    0.52,
    "GI":           0.44,
    "Sleep":        0.71,
    "PT":           0.38,
    "OT":           0.40,
    "E&M":          0.31,
}

# ── PAYER STRICTNESS INDEX (computed from CMS MA denial data) ─────────────────
PAYER_STRICTNESS = {
    "UnitedHealthcare":  0.72,
    "Humana":            0.61,
    "CVS/Aetna":         0.74,
    "Cigna":             0.68,
    "Centene/WellCare":  0.79,
    "Molina Healthcare": 0.83,
    "Anthem/Elevance":   0.59,
    "Kaiser Permanente": 0.38,
    "BCBS Plans":        0.56,
    "Oscar Health":      0.70,
}

# ── HIGH-RISK CPT CODE FLAGS ───────────────────────────────────────────────────
# Biologics, injectables, and surgical procedures have highest denial rates
HIGH_RISK_CPT_PREFIXES = {"J0", "J9", "Q0", "0202", "436", "438"}
BIOLOGIC_CPT = {"J0129", "J0717", "J0135", "J9355", "J9306"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw CMS prior auth data.
    Returns feature-engineered DataFrame ready for modeling.
    """
    df = df.copy()

    # ── LABEL ─────────────────────────────────────────────────────────────────
    # Target: denial rate (regression) + denied flag (classification)
    df["denial_rate"]  = df["denial_rate"].clip(0, 1)
    df["denied_flag"]  = (df["denial_rate"] > 0.25).astype(int)

    # ── FEATURE 1: Specialty risk score ───────────────────────────────────────
    df["specialty_risk"] = df["specialty"].map(SPECIALTY_RISK).fillna(0.5)

    # ── FEATURE 2: Payer strictness index ─────────────────────────────────────
    df["payer_strictness"] = df["payer_name"].map(PAYER_STRICTNESS).fillna(0.65)

    # ── FEATURE 3: CPT code risk flags ────────────────────────────────────────
    df["is_biologic"]       = df["cpt_code"].isin(BIOLOGIC_CPT).astype(int)
    df["is_high_risk_cpt"]  = df["cpt_code"].apply(
        lambda x: any(str(x).startswith(p) for p in HIGH_RISK_CPT_PREFIXES)
    ).astype(int)
    df["is_surgical"]       = df["specialty"].isin(
        ["Surgery", "Orthopedic", "Pain Mgmt"]
    ).astype(int)
    df["is_behavioral"]     = (df["specialty"] == "Behavioral").astype(int)
    df["is_oncology"]       = (df["specialty"] == "Oncology").astype(int)

    # ── FEATURE 4: Plan type risk ──────────────────────────────────────────────
    plan_risk_map = {"MA": 0.65, "Medicaid": 0.78, "ACA": 0.61, "FFS": 0.42}
    df["plan_type_risk"] = df["plan_type"].map(plan_risk_map).fillna(0.60)

    # ── FEATURE 5: Volume signals ──────────────────────────────────────────────
    df["log_total_requests"] = np.log1p(df["total_requests"])
    df["volume_tier"] = pd.qcut(
        df["total_requests"], q=4, labels=[0, 1, 2, 3]
    ).astype(int)

    # ── FEATURE 6: Appeal success rate ────────────────────────────────────────
    # High appeal success = systemic denial problem (payer denies then capitulates)
    df["appeal_success_rate"] = df["appeal_success_rate"].fillna(0)
    df["high_appeal_reversal"] = (df["appeal_success_rate"] > 0.4).astype(int)

    # ── FEATURE 7: Decision time pressure ─────────────────────────────────────
    df["days_standard"]   = df["avg_decision_days_standard"].fillna(7)
    df["days_expedited"]  = df["avg_decision_days_expedited"].fillna(2)
    df["slow_payer"]      = (df["days_standard"] > 6).astype(int)

    # ── FEATURE 8: Interaction terms ──────────────────────────────────────────
    df["biologic_x_strict_payer"] = df["is_biologic"] * df["payer_strictness"]
    df["surgical_x_ma_plan"]      = df["is_surgical"] * (df["plan_type"] == "MA").astype(int)
    df["specialty_x_payer"]       = df["specialty_risk"] * df["payer_strictness"]

    # ── FEATURE 9: Year trend ─────────────────────────────────────────────────
    df["year_norm"] = (df.get("year", 2024) - 2020) / 5.0

    # ── SELECT FINAL FEATURE COLUMNS ──────────────────────────────────────────
    feature_cols = [
        "specialty_risk", "payer_strictness", "plan_type_risk",
        "is_biologic", "is_high_risk_cpt", "is_surgical",
        "is_behavioral", "is_oncology",
        "log_total_requests", "volume_tier",
        "appeal_success_rate", "high_appeal_reversal",
        "days_standard", "days_expedited", "slow_payer",
        "biologic_x_strict_payer", "surgical_x_ma_plan", "specialty_x_payer",
        "year_norm",
    ]

    df_out = df[feature_cols + ["denial_rate", "denied_flag",
                                 "cpt_code", "procedure_description",
                                 "specialty", "payer_name", "plan_type",
                                 "total_requests", "approved", "denied",
                                 "appeal_success_rate"]].copy()

    df_out.to_csv(PROC_DIR / "features.csv", index=False)
    print(f"Feature matrix built: {df_out.shape[0]} rows x {len(feature_cols)} features")

    return df_out, feature_cols


def get_payer_profile(payer_name: str) -> dict:
    """Return denial profile for a given payer."""
    return {
        "payer": payer_name,
        "strictness_index": PAYER_STRICTNESS.get(payer_name, 0.65),
        "rank": sorted(PAYER_STRICTNESS.items(),
                       key=lambda x: x[1], reverse=True).index(
            (payer_name, PAYER_STRICTNESS.get(payer_name, 0.65))
        ) + 1 if payer_name in PAYER_STRICTNESS else None
    }


if __name__ == "__main__":
    df_raw = pd.read_csv(Path(__file__).parents[2] / "data" / "raw" / "cms_prior_auth.csv")
    df_features, feat_cols = engineer_features(df_raw)
    print(df_features.head())
    print("\nFeatures:", feat_cols)
