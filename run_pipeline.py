"""
PriorAI — Master Pipeline
Run this once to:
  1. Ingest CMS + CFPB data
  2. Engineer features
  3. Train XGBoost + SHAP models
  4. Print launch instructions

Usage:
    python run_pipeline.py

Then launch dashboard:
    streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.ingestion.cms_pipeline import run_full_pipeline
from src.features.feature_engineering import engineer_features
from src.model.train import run_training


def main():
    print("\n" + "="*60)
    print("  PriorAI — Full Pipeline")
    print("  Prior Authorization Denial Prediction System")
    print("="*60)

    # Step 1: Ingest
    cms_df, cfpb_df, fred_df = run_full_pipeline()

    # Step 2: Feature engineering
    print("\nStep 2: Engineering features...")
    df_features, feature_cols = engineer_features(cms_df)

    # Step 3: Train
    print("\nStep 3: Training models...")
    clf_model, reg_model, explainer = run_training(df_features)

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("  Launch the dashboard with:")
    print()
    print("      streamlit run app.py")
    print()
    print("  Or open: http://localhost:8501")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
