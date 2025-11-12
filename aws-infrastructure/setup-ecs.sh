#!/bin/bash
# AWS ECS Setup Script

set -e

# Configuration
AWS_REGION="us-east-1"
CLUSTER_NAME="bearing-classifier-cluster"
SERVICE_NAME="bearing-classifier-service"
TASK_FAMILY="bearing-classifier-task"
ECR_REPO="bearing-fault-classifier"
CONTAINER_NAME="bearing-api"
CONTAINER_PORT=8000

echo "🚀 Setting up AWS ECS Infrastructure"
echo "======================================"

# 1. Create ECR Repository
echo "📦 Creating ECR repository..."
aws ecr create-repository \
    --repository-name $ECR_REPO \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    || echo "Repository already exists"

# 2. Create ECS Cluster
echo "🏗️  Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $CLUSTER_NAME \
    --region $AWS_REGION \
    || echo "Cluster already exists"

# 3. Create CloudWatch Log Group
echo "📊 Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name /ecs/$TASK_FAMILY \
    --region $AWS_REGION \
    || echo "Log group already exists"

# 4. Create IAM Role for ECS Task Execution
echo "🔐 Creating IAM roles..."
cat > /tmp/trust-policy.json << 'TRUST'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
TRUST

aws iam create-role \
    --role-name ecsTaskExecutionRole \
    --assume-role-policy-document file:///tmp/trust-policy.json \
    || echo "Role already exists"

aws iam attach-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"

echo ""
echo "✅ Infrastructure setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Push your Docker image to ECR:"
echo "   aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
echo "   docker tag bearing-fault-classifier:latest $ECR_URI:latest"
echo "   docker push $ECR_URI:latest"
echo ""
echo "2. Register task definition (see task-definition.json)"
echo "3. Create ECS service (see create-service.sh)"
EOF
