"""
Download ML models from S3 at container startup
"""
import os
import requests
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "random_forest_model.pkl": {
        "url": os.getenv("MODEL_URL", ""),
    },
    "random_forest_scaler.pkl": {
        "url": os.getenv("SCALER_URL", ""),
    }
}

def download_file(url: str, save_path: Path):
    """Download file from URL"""
    if not url:
        raise ValueError(f"No URL provided for {save_path.name}")
    
    logger.info(f"📥 Downloading {save_path.name}...")
    logger.info(f"   From: {url[:80]}...")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f"✅ Downloaded {save_path.name} ({downloaded / 1024 / 1024:.2f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise

def ensure_models_exist(force_download: bool = False):
    """Download models if they don't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    logger.info("🔍 Checking model files...")
    
    for filename, config in MODEL_CONFIG.items():
        model_path = models_dir / filename
        url = config["url"]
        
        should_download = force_download or not model_path.exists()
        
        if should_download:
            if not url:
                raise ValueError(f"❌ {filename} URL not set! Check MODEL_URL and SCALER_URL environment variables.")
            
            download_file(url, model_path)
        else:
            logger.info(f"✅ {filename} already exists")
    
    logger.info("✅ All model files ready!")

def get_model_version():
    """Get model version from environment"""
    return os.getenv("MODEL_VERSION", "1.0.0")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_models_exist()
