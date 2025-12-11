"""
Evaluate a saved XGBoost model on the eval split.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    eval_df = pd.read_csv(eval_path)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    # ---------------------------
    # CLEANUP: match train.py
    # ---------------------------

    # 1) Drop leftover original city columns (same as in train.py)
    cols_to_drop = ["city_norm", "city_encoded", "city"]
    X_eval = X_eval.drop(columns=cols_to_drop, errors="ignore")

    # 2) Keep only numeric + bool columns
    X_eval = X_eval.select_dtypes(include=["number", "bool"])

    # 3) Handle missing values
    X_eval = X_eval.fillna(0)

    # Load model AFTER preprocessing, so we can align columns to it
    model = load(model_path)

    # 4) (Optional but safer) Align eval columns to training columns
    #    This ensures the same features in same order.
    if hasattr(model, "feature_names_in_"):
        train_cols = list(model.feature_names_in_)

        # Add any missing columns with 0
        for col in train_cols:
            if col not in X_eval.columns:
                X_eval[col] = 0

        # Drop any extra columns not seen during training
        X_eval = X_eval[train_cols]

    y_pred = model.predict(X_eval)

    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print("📊 Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
