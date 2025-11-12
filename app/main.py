"""
FastAPI Application for Bearing Fault Classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bearing Fault Classifier API",
    description="Production ML API for bearing fault detection with 96.20% accuracy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount templates
templates = Jinja2Templates(directory="app/templates")

# Global variables for model
MODEL = None
SCALER = None
FEATURE_NAMES = None
CLASS_LABELS = ["Ball", "Inner_Race", "Normal", "Outer_Race"]


# Pydantic models for request/response
class PredictionInput(BaseModel):
    """Input features for prediction"""

    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Output prediction"""

    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    status: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_name: str
    version: str


# Startup event - Load model
@app.on_event("startup")
async def load_model():
    """Load trained model and scaler on startup"""
    global MODEL, SCALER, FEATURE_NAMES

    try:
        models_dir = Path("models")

        # Load best model (Random Forest)
        model_path = models_dir / "random_forest_model.pkl"
        scaler_path = models_dir / "random_forest_scaler.pkl"

        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        MODEL = joblib.load(model_path)
        SCALER = joblib.load(scaler_path)

        # Load feature names
        FEATURE_NAMES = [f"feature_{i}" for i in range(19)]

        logger.info("✓ Model and scaler loaded successfully")
        logger.info(f"✓ Model type: {type(MODEL).__name__}")
        logger.info(f"✓ Expected features: {len(FEATURE_NAMES)}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render home page"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Bearing Fault Classifier", "accuracy": "96.20%"},
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        model_name="Random Forest Classifier",
        version="1.0.0",
    )


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make prediction on input features
    """
    try:
        if MODEL is None or SCALER is None:
            raise HTTPException(
                status_code=503, detail="Model not loaded. Please restart the service."
            )

        features = input_data.features
        if len(features) != 19:
            raise HTTPException(
                status_code=400, detail=f"Expected 19 features, got {len(features)}"
            )

        X = np.array(features).reshape(1, -1)
        X_scaled = SCALER.transform(X)

        prediction = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]

        confidence = float(np.max(probabilities))

        prob_dict = {
            label: float(prob) for label, prob in zip(CLASS_LABELS, probabilities)
        }

        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

        return PredictionOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=prob_dict,
            status="success",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV file"""
    try:
        if MODEL is None or SCALER is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        if len(df.columns) != 19:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 19 feature columns, got {len(df.columns)}",
            )

        X_scaled = SCALER.transform(df.values)
        predictions = MODEL.predict(X_scaled)
        probabilities = MODEL.predict_proba(X_scaled)

        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            results.append(
                {
                    "index": i,
                    "prediction": pred,
                    "confidence": float(np.max(probs)),
                    "probabilities": {
                        label: float(prob) for label, prob in zip(CLASS_LABELS, probs)
                    },
                }
            )

        logger.info(f"Batch prediction: {len(results)} samples processed")

        return JSONResponse(
            content={"status": "success", "count": len(results), "predictions": results}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(MODEL).__name__,
        "n_features": len(FEATURE_NAMES),
        "classes": CLASS_LABELS,
        "accuracy": "96.20%",
        "training_samples": 3680,
        "test_samples": 920,
        "feature_names": FEATURE_NAMES,
    }


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    return {"model_loaded": 1 if MODEL is not None else 0, "api_status": "up"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
