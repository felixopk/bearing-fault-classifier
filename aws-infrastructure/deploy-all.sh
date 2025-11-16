#!/bin/bash
# Master deployment script - runs everything in order

set -e

echo "🚀 Complete AWS Deployment Script"
echo "=================================="
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI not configured!"
    echo "Please run: aws configure"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running!"
    echo "Please start Docker and try again"
    exit 1
fi

echo "Step 1/6: Setting up AWS infrastructure..."
./aws-infrastructure/setup-infrastructure.sh

echo ""
echo "Step 2/6: Uploading models to S3..."
./aws-infrastructure/upload-models.sh

echo ""
echo "Step 3/6: Building and pushing Docker image..."
./aws-infrastructure/build-and-push.sh

echo ""
echo "Step 4/6: Creating ECS task definition..."
./aws-infrastructure/create-task-definition.sh

echo ""
echo "Step 5/6: Creating ECS service..."
./aws-infrastructure/create-service.sh

echo ""
echo "Step 6/6: Configuring DNS (Route 53)..."
read -p "Do you want to configure DNS now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./aws-infrastructure/configure-dns.sh
else
    echo "⏭️  Skipping DNS configuration"
    echo "You can run it later with: ./aws-infrastructure/configure-dns.sh"
fi

# Load config
source aws-infrastructure/config.env

echo ""
echo "=================================="
echo "✅ Deployment Complete!"
echo "=================================="
echo ""
echo "🔗 Your API is accessible at:"
echo "  Load Balancer: http://$ALB_DNS"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Custom Domain: http://$DOMAIN_NAME"
fi
echo ""
echo "🧪 Test your API:"
echo "  curl http://$ALB_DNS/health"
echo "  curl http://$ALB_DNS/model/info"
echo "  open http://$ALB_DNS/docs"
echo ""
echo "📊 Monitor your service:"
echo "  AWS Console: https://console.aws.amazon.com/ecs/home?region=$AWS_REGION#/clusters/$CLUSTER_NAME/services/$SERVICE_NAME"
echo "  CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group//ecs/$TASK_FAMILY"
echo ""
echo "🔄 To update your application:"
echo "  1. Make code changes"
echo "  2. Push to GitHub: git push origin main"
echo "  3. GitHub Actions will automatically deploy"
echo ""
echo "💰 Estimated monthly cost: ~$15-30"
echo "  - Fargate: ~$15 (1 task, 0.5 vCPU, 1GB RAM)"
echo "  - ALB: ~$16"
echo "  - Data transfer: ~$1"
echo "  - S3/ECR: <$1"
echo ""
