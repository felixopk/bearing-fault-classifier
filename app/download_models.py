"""
Download ML models from S3 at container startup
"""
import os
import boto3
from pathlib import Path

BUCKET = "opkcloudz-ml-models"
MODEL_KEY = "bearing-classifier/v1.0.0/random_forest_model.pkl"
SCALER_KEY = "bearing-classifier/v1.0.0/random_forest_scaler.pkl"

def download_file(bucket: str, key: str, local_path: str):
    print(f"📥 Downloading from S3: s3://{bucket}/{key}")
    s3 = boto3.client("s3")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.download_file(bucket, key, local_path)
        print(f"✅ Downloaded {key} → {local_path}")
    except Exception as e:
        print(f"❌ Failed to download {key}: {e}")
        raise

def ensure_models_exist():
    """Ensure both model + scaler exist locally, else download from S3."""
    models_dir = Path("/app/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "random_forest_model.pkl"
    scaler_path = models_dir / "random_forest_scaler.pkl"

    if not model_path.exists():
        download_file(BUCKET, MODEL_KEY, str(model_path))

    if not scaler_path.exists():
        download_file(BUCKET, SCALER_KEY, str(scaler_path))

def get_model_version():
    """Return model version from environment or default."""
    return os.getenv("MODEL_VERSION", "1.0.0")
