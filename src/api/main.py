from src.batch.run_monthly import run_monthly_predictions
from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import boto3
import os
from contextlib import asynccontextmanager

# Import inference pipeline
from src.inference_pipeline.inference import predict

# Config
S3_BUCKET = os.getenv("S3_BUCKET", "housing-regression-data31")
REGION = os.getenv("AWS_REGION", "ap-south-2")
s3 = None
MODEL_PATH: Path | None = None
TRAIN_FE_PATH: Path | None = None
TRAIN_FEATURE_COLUMNS: list[str] | None = None
HOLDOUT_DATA = None


def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# Paths


@asynccontextmanager
async def lifespan(app: FastAPI):
    global s3, MODEL_PATH, TRAIN_FE_PATH, TRAIN_FEATURE_COLUMNS, HOLDOUT_DATA

    s3 = boto3.client("s3", region_name=REGION)

    MODEL_PATH = Path(load_from_s3(
        "models/xgb_best_model.pkl",
        "models/xgb_best_model.pkl"
    ))

    TRAIN_FE_PATH = Path(load_from_s3(
        "processed/feature_engineered_train.csv",
        "data/processed/feature_engineered_train.csv"
    ))

    fe_path = Path(load_from_s3(
        "processed/feature_engineered_holdout.csv",
        "data/processed/feature_engineered_holdout.csv"
    ))
    meta_path = Path(load_from_s3(
        "processed/cleaning_holdout.csv",
        "data/processed/cleaning_holdout.csv"
    ))

    fe = pd.read_csv(fe_path)
    meta = pd.read_csv(meta_path, parse_dates=["date"])[["date", "city_full"]]

    HOLDOUT_DATA = {
        "features": fe.to_dict(orient="records"),
        "meta": meta.to_dict(orient="records"),
    }

    yield


# App
# Instantiates the FastAPI app.
app = FastAPI(
    title="Housing Regression API",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀"}

# /health → checks if model exists, returns status info


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/holdout_data")
def get_holdout_data():
    return HOLDOUT_DATA


@app.get("/ready")
def ready():
    # App started, but model not ready yet
    if MODEL_PATH is None:
        return {
            "status": "starting",
            "message": "Model not loaded yet"
        }

    # Model path exists check
    if not MODEL_PATH.exists():
        return {
            "status": "unhealthy",
            "error": f"Model not found at {MODEL_PATH}"
        }

    return {
        "status": "healthy",
        "model_path": str(MODEL_PATH),
        "n_features_expected": (
            len(TRAIN_FEATURE_COLUMNS)
            if TRAIN_FEATURE_COLUMNS is not None
            else None
        )
    }

# Prediction Endpoint: This is the core ML serving endpoint.


@app.post("/predict")
def predict_batch(data: List[dict]):
    # ⛔ Model not ready yet
    if MODEL_PATH is None:
        return {
            "status": "starting",
            "message": "Model not loaded yet"
        }

    if not MODEL_PATH.exists():
        return {
            "status": "unhealthy",
            "error": f"Model not found at {MODEL_PATH}"
        }

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    # 🔒 Align features safely
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    preds_df = predict(df, model_path=MODEL_PATH)

    resp = {
        "predictions": preds_df["predicted_price"].astype(float).tolist()
    }

    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp


# Batch runner

# Trigger a monthly batch job via API.


@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/"
    }

# Returns a preview of the most recent batch predictions.


@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }


# ============================================================================
# AWS Lambda Handler (2026 Best Practice)
# ============================================================================
# This handler allows the FastAPI app to run on AWS Lambda
# Uses Mangum adapter to convert Lambda events to ASGI format
# ============================================================================

from mangum import Mangum

# Lambda handler - entry point for AWS Lambda
# Set lifespan="off" to prevent issues with Lambda's execution model
handler = Mangum(app, lifespan="off")
