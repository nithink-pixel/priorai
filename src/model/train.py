"""
PriorAI — Model Training & SHAP Explanation Engine
XGBoost denial prediction with SHAP waterfall explanations.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import shap

MODEL_DIR = Path(__file__).parents[2] / "data" / "processed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "specialty_risk", "payer_strictness", "plan_type_risk",
    "is_biologic", "is_high_risk_cpt", "is_surgical",
    "is_behavioral", "is_oncology",
    "log_total_requests", "volume_tier",
    "appeal_success_rate", "high_appeal_reversal",
    "days_standard", "days_expedited", "slow_payer",
    "biologic_x_strict_payer", "surgical_x_ma_plan", "specialty_x_payer",
    "year_norm",
]


def train_denial_classifier(df: pd.DataFrame):
    """
    Train XGBoost binary classifier: will this prior auth be denied?
    Target: denied_flag (1 = high denial rate > 25%)
    """
    print("\n── Training Denial Classifier ──────────────────────────────")

    X = df[FEATURE_COLS]
    y = df["denied_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y[y==0]) / max(len(y[y==1]), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC:  {auc:.4f}")
    print(f"  Accuracy: {(y_pred == y_test).mean():.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Approved', 'Denied'])}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc-auc", n_jobs=-1)
    print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    with open(MODEL_DIR / "denial_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test, auc


def train_denial_rate_regressor(df: pd.DataFrame):
    """
    Train XGBoost regressor to predict the exact denial rate (0-1).
    """
    print("\n── Training Denial Rate Regressor ──────────────────────────")

    X = df[FEATURE_COLS]
    y = df["denial_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    with open(MODEL_DIR / "denial_rate_regressor.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def compute_shap_values(model, X_sample: pd.DataFrame):
    """
    Compute SHAP values for model explainability.
    Returns: shap_values, explainer
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Save explainer
    with open(MODEL_DIR / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    return shap_values, explainer


def explain_single_prediction(
    model,
    explainer,
    features: dict,
    feature_names: list
) -> dict:
    """
    Generate SHAP explanation for a single prior auth request.
    Returns structured dict for dashboard display.
    """
    X = pd.DataFrame([features])[feature_names]
    prob = float(model.predict_proba(X)[0][1])
    shap_vals = explainer.shap_values(X)[0]

    # Build explanation
    impacts = []
    for i, col in enumerate(feature_names):
        impacts.append({
            "feature": col,
            "value": float(X[col].iloc[0]),
            "shap_impact": float(shap_vals[i]),
            "direction": "increases_denial" if shap_vals[i] > 0 else "decreases_denial"
        })

    # Sort by absolute impact
    impacts.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
    top_factors = impacts[:5]

    # Human-readable factor names
    readable_names = {
        "payer_strictness":         "Payer denial history",
        "specialty_risk":           "Specialty denial risk",
        "is_biologic":              "Biologic/injectable drug",
        "is_surgical":              "Surgical procedure",
        "plan_type_risk":           "Insurance plan type",
        "biologic_x_strict_payer":  "Biologic + strict payer combo",
        "surgical_x_ma_plan":       "Surgery on Medicare Advantage",
        "specialty_x_payer":        "Specialty-payer combination",
        "appeal_success_rate":      "Historical appeal success",
        "high_appeal_reversal":     "Payer often reverses on appeal",
        "slow_payer":               "Slow decision maker",
        "is_high_risk_cpt":         "High-risk procedure code",
        "is_oncology":              "Oncology treatment",
        "is_behavioral":            "Behavioral health",
        "days_standard":            "Average approval timeline",
    }

    for f in top_factors:
        f["readable_name"] = readable_names.get(f["feature"], f["feature"])

    return {
        "denial_probability": round(prob, 3),
        "denial_percentage":  f"{prob * 100:.1f}%",
        "risk_level":         "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.35 else "LOW",
        "top_factors":        top_factors,
        "recommendation":     _generate_recommendation(prob, top_factors),
    }


def _generate_recommendation(prob: float, factors: list) -> str:
    """Generate actionable recommendation based on denial probability."""
    if prob < 0.25:
        return "Low denial risk. Standard documentation should be sufficient."

    recs = []
    factor_names = [f["feature"] for f in factors[:3] if f["shap_impact"] > 0]

    if "is_biologic" in factor_names or "is_high_risk_cpt" in factor_names:
        recs.append(
            "Include step therapy documentation showing prior treatment failures."
        )
    if "is_surgical" in factor_names:
        recs.append(
            "Attach conservative treatment history (PT, medications tried and failed)."
        )
    if "payer_strictness" in factor_names:
        recs.append(
            "This payer has high denial rates. Request expedited review if clinically urgent."
        )
    if "is_behavioral" in factor_names:
        recs.append(
            "Include functional impairment documentation and prior treatment response."
        )
    if "slow_payer" in factor_names:
        recs.append(
            "File at least 10 days before treatment. Consider peer-to-peer call."
        )
    if "high_appeal_reversal" in factor_names:
        recs.append(
            "This payer frequently reverses on appeal. If denied, appeal immediately."
        )

    if not recs:
        recs.append(
            "Include detailed clinical notes supporting medical necessity."
        )

    return " ".join(recs)


def load_models():
    """Load saved models and explainer."""
    models = {}
    for name, fname in [
        ("classifier",  "denial_classifier.pkl"),
        ("regressor",   "denial_rate_regressor.pkl"),
        ("explainer",   "shap_explainer.pkl"),
    ]:
        path = MODEL_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def run_training(df: pd.DataFrame):
    """Run full training pipeline."""
    print("\n" + "="*60)
    print("PriorAI — Model Training")
    print("="*60)

    clf_model, X_test_clf, y_test_clf, auc = train_denial_classifier(df)
    reg_model, X_test_reg, y_test_reg      = train_denial_rate_regressor(df)

    print("\n── Computing SHAP Values ────────────────────────────────────")
    shap_vals, explainer = compute_shap_values(clf_model, X_test_clf)
    print(f"  SHAP computed for {len(X_test_clf)} test samples")

    # Feature importance summary
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n── Top 10 Features by SHAP Importance ──────────────────────")
    print(importance_df.head(10).to_string(index=False))

    importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    # Save model metadata
    metadata = {
        "roc_auc":        round(auc, 4),
        "n_features":     len(FEATURE_COLS),
        "n_training_rows": len(df),
        "features":       FEATURE_COLS,
        "trained_at":     pd.Timestamp.now().isoformat(),
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Models saved to: {MODEL_DIR}")
    print("="*60 + "\n")

    return clf_model, reg_model, explainer


if __name__ == "__main__":
    PROC_DIR = Path(__file__).parents[2] / "data" / "processed"
    df = pd.read_csv(PROC_DIR / "features.csv")
    run_training(df)
