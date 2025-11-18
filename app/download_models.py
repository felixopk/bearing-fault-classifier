"""
Download ML models from S3 at container startup using boto3.
"""

import os
from pathlib import Path
import logging
import boto3

logger = logging.getLogger(__name__)

BUCKET = "opkcloudz-ml-models"

MODEL_KEY = "bearing-classifier/v1.0.0/random_forest_model.pkl"
SCALER_KEY = "bearing-classifier/v1.0.0/random_forest_scaler.pkl"


def download_file(bucket: str, key: str, local_path: Path):
    """Download a file from S3 to the container filesystem."""
    logger.info(f"📥 Downloading: s3://{bucket}/{key}")

    s3 = boto3.client("s3")

    # Ensure destination folder exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3.download_file(bucket, key, str(local_path))
        logger.info(f"✅ Downloaded {key} → {local_path}")
    except Exception as e:
        logger.error(f"❌ Failed to download {key}: {e}")
        raise


def ensure_models_exist():
    """Check and download ML models before starting the API."""
    base_path = Path("/app/models")
    base_path.mkdir(parents=True, exist_ok=True)

    model_path = base_path / "random_forest_model.pkl"
    scaler_path = base_path / "random_forest_scaler.pkl"

    logger.info("🔍 Checking model files...")

    if not model_path.exists():
        logger.info("📥 Model file missing. Downloading...")
        download_file(BUCKET, MODEL_KEY, model_path)

    if not scaler_path.exists():
        logger.info("📥 Scaler file missing. Downloading...")
        download_file(BUCKET, SCALER_KEY, scaler_path)

    logger.info("✨ Model and scaler are ready!")
