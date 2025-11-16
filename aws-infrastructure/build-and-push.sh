#!/bin/bash
# Build Docker image and push to ECR

set -e

# Load configuration
source aws-infrastructure/config.env

echo "🐳 Building and pushing Docker image..."
echo "ECR Repository: $ECR_URI"
echo ""

# Login to ECR
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_URI

# Build image
echo "🏗️  Building Docker image..."
docker build -t $APP_NAME:latest .

# Tag image
echo "🏷️  Tagging image..."
docker tag $APP_NAME:latest $ECR_URI:latest
docker tag $APP_NAME:latest $ECR_URI:$(git rev-parse --short HEAD)

# Push to ECR
echo "📤 Pushing to ECR..."
docker push $ECR_URI:latest
docker push $ECR_URI:$(git rev-parse --short HEAD)

echo ""
echo "✅ Image pushed successfully!"
echo "  $ECR_URI:latest"
echo "  $ECR_URI:$(git rev-parse --short HEAD)"
echo ""
