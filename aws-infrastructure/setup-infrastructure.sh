#!/bin/bash
# Complete AWS Infrastructure Setup for Bearing Fault Classifier

set -e

# Configuration
AWS_REGION="us-east-1"
APP_NAME="bearing-classifier"
CLUSTER_NAME="${APP_NAME}-cluster"
SERVICE_NAME="${APP_NAME}-service"
TASK_FAMILY="${APP_NAME}-task"
ECR_REPO="${APP_NAME}"
DOMAIN_NAME="api.opkcloudz.com"  # Your subdomain
S3_MODELS_BUCKET="opkcloudz-ml-models"

echo "🚀 Setting up AWS Infrastructure"
echo "=================================="
echo "Region: $AWS_REGION"
echo "App: $APP_NAME"
echo "Domain: $DOMAIN_NAME"
echo ""

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account ID: $ACCOUNT_ID"
echo ""

# 1. Create ECR Repository
echo "📦 Creating ECR repository..."
aws ecr create-repository \
    --repository-name $ECR_REPO \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    2>/dev/null || echo "  Repository already exists"

ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"
echo "  ECR URI: $ECR_URI"
echo ""

# 2. Create S3 Bucket for Models
echo "🗄️  Creating S3 bucket for models..."
aws s3 mb s3://$S3_MODELS_BUCKET --region $AWS_REGION 2>/dev/null || echo "  Bucket already exists"

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket $S3_MODELS_BUCKET \
    --versioning-configuration Status=Enabled

echo "  S3 Bucket: $S3_MODELS_BUCKET"
echo ""

# 3. Create ECS Cluster
echo "🏗️  Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $CLUSTER_NAME \
    --region $AWS_REGION \
    --capacity-providers FARGATE FARGATE_SPOT \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    2>/dev/null || echo "  Cluster already exists"
echo ""

# 4. Create CloudWatch Log Group
echo "📊 Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name /ecs/$TASK_FAMILY \
    --region $AWS_REGION \
    2>/dev/null || echo "  Log group already exists"

# Set retention
aws logs put-retention-policy \
    --log-group-name /ecs/$TASK_FAMILY \
    --retention-in-days 7 \
    --region $AWS_REGION
echo ""

# 5. Create IAM Roles
echo "🔐 Creating IAM roles..."

# Task Execution Role (for pulling images, writing logs)
cat > /tmp/ecs-task-execution-trust.json << 'TRUST'
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
    --role-name ${APP_NAME}-ecs-task-execution-role \
    --assume-role-policy-document file:///tmp/ecs-task-execution-trust.json \
    2>/dev/null || echo "  Execution role already exists"

aws iam attach-role-policy \
    --role-name ${APP_NAME}-ecs-task-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Task Role (for accessing S3, etc.)
aws iam create-role \
    --role-name ${APP_NAME}-ecs-task-role \
    --assume-role-policy-document file:///tmp/ecs-task-execution-trust.json \
    2>/dev/null || echo "  Task role already exists"

# Create S3 access policy for task role
cat > /tmp/s3-access-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${S3_MODELS_BUCKET}",
        "arn:aws:s3:::${S3_MODELS_BUCKET}/*"
      ]
    }
  ]
}
POLICY

aws iam put-role-policy \
    --role-name ${APP_NAME}-ecs-task-role \
    --policy-name S3ModelsAccess \
    --policy-document file:///tmp/s3-access-policy.json

echo ""

# 6. Create VPC and Networking (if needed)
echo "🌐 Setting up networking..."

# Check if default VPC exists
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $AWS_REGION)

if [ "$DEFAULT_VPC" = "None" ] || [ -z "$DEFAULT_VPC" ]; then
    echo "  Creating new VPC..."
    VPC_ID=$(aws ec2 create-vpc --cidr-block 10.0.0.0/16 --query Vpc.VpcId --output text --region $AWS_REGION)
    aws ec2 create-tags --resources $VPC_ID --tags Key=Name,Value=${APP_NAME}-vpc --region $AWS_REGION
    
    # Create subnets
    SUBNET1=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.1.0/24 --availability-zone ${AWS_REGION}a --query Subnet.SubnetId --output text --region $AWS_REGION)
    SUBNET2=$(aws ec2 create-subnet --vpc-id $VPC_ID --cidr-block 10.0.2.0/24 --availability-zone ${AWS_REGION}b --query Subnet.SubnetId --output text --region $AWS_REGION)
    
    # Create internet gateway
    IGW=$(aws ec2 create-internet-gateway --query InternetGateway.InternetGatewayId --output text --region $AWS_REGION)
    aws ec2 attach-internet-gateway --vpc-id $VPC_ID --internet-gateway-id $IGW --region $AWS_REGION
else
    echo "  Using default VPC: $DEFAULT_VPC"
    VPC_ID=$DEFAULT_VPC
    
    # Get default subnets
    SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[].SubnetId" --output text --region $AWS_REGION)
    SUBNET1=$(echo $SUBNETS | cut -d' ' -f1)
    SUBNET2=$(echo $SUBNETS | cut -d' ' -f2)
fi

echo "  VPC ID: $VPC_ID"
echo "  Subnet 1: $SUBNET1"
echo "  Subnet 2: $SUBNET2"
echo ""

# 7. Create Security Groups
echo "🔒 Creating security groups..."

# ALB Security Group
ALB_SG=$(aws ec2 create-security-group \
    --group-name ${APP_NAME}-alb-sg \
    --description "Security group for ${APP_NAME} ALB" \
    --vpc-id $VPC_ID \
    --output text \
    --query 'GroupId' \
    --region $AWS_REGION 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=${APP_NAME}-alb-sg" "Name=vpc-id,Values=$VPC_ID" \
        --query "SecurityGroups[0].GroupId" \
        --output text \
        --region $AWS_REGION)

# Allow HTTP and HTTPS
aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION 2>/dev/null || true

# ECS Security Group
ECS_SG=$(aws ec2 create-security-group \
    --group-name ${APP_NAME}-ecs-sg \
    --description "Security group for ${APP_NAME} ECS tasks" \
    --vpc-id $VPC_ID \
    --output text \
    --query 'GroupId' \
    --region $AWS_REGION 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=${APP_NAME}-ecs-sg" "Name=vpc-id,Values=$VPC_ID" \
        --query "SecurityGroups[0].GroupId" \
        --output text \
        --region $AWS_REGION)

# Allow traffic from ALB
aws ec2 authorize-security-group-ingress \
    --group-id $ECS_SG \
    --protocol tcp \
    --port 8000 \
    --source-group $ALB_SG \
    --region $AWS_REGION 2>/dev/null || true

echo "  ALB Security Group: $ALB_SG"
echo "  ECS Security Group: $ECS_SG"
echo ""

# 8. Create Application Load Balancer
echo "⚖️  Creating Application Load Balancer..."

ALB_ARN=$(aws elbv2 create-load-balancer \
    --name ${APP_NAME}-alb \
    --subnets $SUBNET1 $SUBNET2 \
    --security-groups $ALB_SG \
    --region $AWS_REGION \
    --output text \
    --query 'LoadBalancers[0].LoadBalancerArn' 2>/dev/null || \
    aws elbv2 describe-load-balancers \
        --names ${APP_NAME}-alb \
        --query "LoadBalancers[0].LoadBalancerArn" \
        --output text \
        --region $AWS_REGION)

# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --query "LoadBalancers[0].DNSName" \
    --output text \
    --region $AWS_REGION)

echo "  ALB ARN: $ALB_ARN"
echo "  ALB DNS: $ALB_DNS"
echo ""

# 9. Create Target Group
echo "🎯 Creating target group..."

TG_ARN=$(aws elbv2 create-target-group \
    --name ${APP_NAME}-tg \
    --protocol HTTP \
    --port 8000 \
    --vpc-id $VPC_ID \
    --target-type ip \
    --health-check-enabled \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --region $AWS_REGION \
    --output text \
    --query 'TargetGroups[0].TargetGroupArn' 2>/dev/null || \
    aws elbv2 describe-target-groups \
        --names ${APP_NAME}-tg \
        --query "TargetGroups[0].TargetGroupArn" \
        --output text \
        --region $AWS_REGION)

echo "  Target Group ARN: $TG_ARN"
echo ""

# 10. Create ALB Listener
echo "👂 Creating ALB listener..."

aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TG_ARN \
    --region $AWS_REGION 2>/dev/null || echo "  Listener already exists"
echo ""

# 11. Save configuration
cat > aws-infrastructure/config.env << CONFIG
AWS_REGION=$AWS_REGION
ACCOUNT_ID=$ACCOUNT_ID
APP_NAME=$APP_NAME
CLUSTER_NAME=$CLUSTER_NAME
SERVICE_NAME=$SERVICE_NAME
TASK_FAMILY=$TASK_FAMILY
ECR_REPO=$ECR_REPO
ECR_URI=$ECR_URI
S3_MODELS_BUCKET=$S3_MODELS_BUCKET
VPC_ID=$VPC_ID
SUBNET1=$SUBNET1
SUBNET2=$SUBNET2
ALB_SG=$ALB_SG
ECS_SG=$ECS_SG
ALB_ARN=$ALB_ARN
ALB_DNS=$ALB_DNS
TG_ARN=$TG_ARN
DOMAIN_NAME=$DOMAIN_NAME
CONFIG

echo "✅ Infrastructure setup complete!"
echo ""
echo "📝 Configuration saved to aws-infrastructure/config.env"
echo ""
echo "🔗 Important URLs:"
echo "  ECR Repository: $ECR_URI"
echo "  S3 Models Bucket: s3://$S3_MODELS_BUCKET"
echo "  Load Balancer DNS: $ALB_DNS"
echo ""
echo "📋 Next steps:"
echo "  1. Upload models to S3: ./aws-infrastructure/upload-models.sh"
echo "  2. Build and push Docker image: ./aws-infrastructure/build-and-push.sh"
echo "  3. Create ECS task definition: ./aws-infrastructure/create-task-definition.sh"
echo "  4. Create ECS service: ./aws-infrastructure/create-service.sh"
echo "  5. Configure Route 53: ./aws-infrastructure/configure-dns.sh"
echo ""