#!/bin/bash
# Master deployment script - runs everything in order

set -euo pipefail

echo "🚀 Complete AWS Deployment Script"
echo "=================================="
echo ""

# ------------------------------------------------
#  CHECKS
# ------------------------------------------------

# Check AWS CLI
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI not configured!"
    echo "Run: aws configure"
    exit 1
fi

# Check Docker
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running!"
    echo "Start Docker and try again."
    exit 1
fi

# ------------------------------------------------
# 1. Infrastructure
# ------------------------------------------------
echo "Step 1/6: Setting up AWS infrastructure..."
./aws-infrastructure/setup-infrastructure.sh

# config.env now exists
source aws-infrastructure/config.env

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
echo "Step 6/6: Configuring DNS (Route 53 + HTTPS)..."
read -p "Configure DNS + HTTPS now? (y/n): " -n 1 -r
echo

if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    ./aws-infrastructure/configure-dns.sh
    DNS_DONE=true
else
    echo "⏭️  Skipping DNS configuration."
    DNS_DONE=false
fi

echo ""
echo "=================================="
echo "✅ Deployment Complete!"
echo "=================================="
echo ""

# ------------------------------------------------
# OUTPUT INFO
# ------------------------------------------------

echo "🔗 Access URLs:"
echo "----------------------------------"
echo "  Load Balancer (HTTPS):  https://$ALB_DNS"

if [[ "$DNS_DONE" = true ]]; then
    echo "  Custom Domain (HTTPS):  https://$DOMAIN_NAME"
fi

echo ""
echo "🧪 Test your API:"
echo "----------------------------------"

echo "  curl https://$ALB_DNS/health"
echo "  curl https://$ALB_DNS/model/info"
echo "  open https://$ALB_DNS/docs"

if [[ "$DNS_DONE" = true ]]; then
    echo ""
    echo "  curl https://$DOMAIN_NAME/health"
    echo "  curl https://$DOMAIN_NAME/model/info"
    echo "  open https://$DOMAIN_NAME/docs"
fi

echo ""
echo "📊 Monitor your service:"
echo "----------------------------------"
echo "  ECS Console:"
echo "    https://console.aws.amazon.com/ecs/home?region=$AWS_REGION#/clusters/$CLUSTER_NAME/services/$SERVICE_NAME"

echo ""
echo "  CloudWatch Logs:"
echo "    https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group//ecs/$TASK_FAMILY"

echo ""
echo "🔄 To update your application:"
echo "----------------------------------"
echo "  1. Make your code changes"
echo "  2. Run:"
echo "       ./aws-infrastructure/build-and-push.sh"
echo "       ./aws-infrastructure/create-service.sh"
echo ""
echo "  (CI/CD pipeline can automate this — ask me to generate it!)"

echo ""
echo "💰 Estimated monthly cost:"
echo "----------------------------------"
echo "  ~\$15–30 monthly:"
echo "    - Fargate Task: ~\$15"
echo "    - ALB: ~\$16"
echo "    - NAT + Data transfer: ~\$3"
echo "    - S3 + ECR storage: < \$1"
echo ""
