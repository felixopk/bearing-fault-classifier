#!/bin/bash
# Master deployment script - runs everything in order

set -euo pipefail

echo "🚀 Complete AWS Deployment Script"
echo "=================================="
echo ""

# -------------------------
# PRECHECKS
# -------------------------

# Check AWS CLI authentication
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ AWS CLI is not authenticated!"
    echo "Run: aws configure"
    exit 1
fi

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Start Docker and try again."
    exit 1
fi

# -------------------------
# 1. SETUP INFRASTRUCTURE
# -------------------------
echo "Step 1/6: Setting up AWS infrastructure..."
./aws-infrastructure/setup-infrastructure.sh

# -------------------------
# 2. UPLOAD MODELS
# -------------------------
echo ""
echo "Step 2/6: Uploading models to S3..."
./aws-infrastructure/upload-models.sh

# -------------------------
# 3. BUILD & PUSH IMAGE
# -------------------------
echo ""
echo "Step 3/6: Building and pushing Docker image..."
./aws-infrastructure/build-and-push.sh

# -------------------------
# 4. TASK DEFINITION
# -------------------------
echo ""
echo "Step 4/6: Creating ECS task definition..."
./aws-infrastructure/create-task-definition.sh

# -------------------------
# 5. ECS SERVICE
# -------------------------
echo ""
echo "Step 5/6: Creating ECS service..."
./aws-infrastructure/create-service.sh

# -------------------------
# 6. DNS + HTTPS
# -------------------------
echo ""
echo "Step 6/6: Configure DNS + HTTPS with Route53?"

read -p "Do you want to configure HTTPS + DNS now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./aws-infrastructure/configure-dns.sh
else
    echo "⏭️  Skipping DNS configuration."
    echo "Run later with: ./aws-infrastructure/configure-dns.sh"
fi

# Load configuration (contains ALB_DNS, DOMAIN_NAME, etc.)
source aws-infrastructure/config.env

# -------------------------
# SUCCESS OUTPUT
# -------------------------
echo ""
echo "=================================="
echo "✅ Deployment Complete!"
echo "=================================="
echo ""

echo "🔗 Your API endpoints:"
echo "  Load Balancer (HTTPS): https://$ALB_DNS"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Custom Domain (HTTPS): https://$DOMAIN_NAME"
fi

echo ""
echo "🧪 Test your API:"
echo "  curl https://$ALB_DNS/health"
echo "  curl https://$ALB_DNS/model/info"
echo "  open https://$ALB_DNS/docs"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "  curl https://$DOMAIN_NAME/health"
fi

echo ""
echo "📊 Monitor your service:"
echo "  ECS Dashboard:"
echo "  https://console.aws.amazon.com/ecs/home?region=$AWS_REGION#/clusters/$CLUSTER_NAME/services/$SERVICE_NAME"

echo ""
echo "  CloudWatch Logs:"
echo "  https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group//ecs/$TASK_FAMILY"

echo ""
echo "🔄 Update workflow:"
echo "  1. Update your code"
echo "  2. Rebuild + push image:"
echo "        ./aws-infrastructure/build-and-push.sh"
echo "  3. Restart tasks:"
echo "        aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --force-new-deployment"
echo ""

echo "💰 Estimated monthly cost: ~15–30 USD"
echo "  - Fargate tasks         ≈ $15"
echo "  - ALB                   ≈ $16"
echo "  - NAT Gateway (optional) additional cost"
echo "  - S3 + ECR Storage      <$1"
echo ""
