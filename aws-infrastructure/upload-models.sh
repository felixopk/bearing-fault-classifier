#!/bin/bash
# Upload trained models to S3

set -e

# Load configuration
source aws-infrastructure/config.env

MODEL_VERSION="1.0.0"
S3_PREFIX="bearing-classifier/v${MODEL_VERSION}"

echo "📤 Uploading models to S3..."
echo "Bucket: $S3_MODELS_BUCKET"
echo "Prefix: $S3_PREFIX"
echo ""

# Upload Random Forest model
if [ -f "models/random_forest_model.pkl" ]; then
    echo "Uploading random_forest_model.pkl..."
    aws s3 cp models/random_forest_model.pkl \
        s3://${S3_MODELS_BUCKET}/${S3_PREFIX}/random_forest_model.pkl \
        --metadata "version=${MODEL_VERSION},accuracy=96.20,date=$(date +%Y-%m-%d)" \
        --region $AWS_REGION
    echo "✅ Model uploaded"
else
    echo "❌ random_forest_model.pkl not found!"
    exit 1
fi

# Upload scaler
if [ -f "models/random_forest_scaler.pkl" ]; then
    echo "Uploading random_forest_scaler.pkl..."
    aws s3 cp models/random_forest_scaler.pkl \
        s3://${S3_MODELS_BUCKET}/${S3_PREFIX}/random_forest_scaler.pkl \
        --metadata "version=${MODEL_VERSION},date=$(date +%Y-%m-%d)" \
        --region $AWS_REGION
    echo "✅ Scaler uploaded"
else
    echo "❌ random_forest_scaler.pkl not found!"
    exit 1
fi

echo ""
echo "✅ All models uploaded successfully!"
echo ""
echo "Model URLs:"
echo "  MODEL_URL=https://${S3_MODELS_BUCKET}.s3.${AWS_REGION}.amazonaws.com/${S3_PREFIX}/random_forest_model.pkl"
echo "  SCALER_URL=https://${S3_MODELS_BUCKET}.s3.${AWS_REGION}.amazonaws.com/${S3_PREFIX}/random_forest_scaler.pkl"
echo ""
