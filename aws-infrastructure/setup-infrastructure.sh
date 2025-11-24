#!/bin/bash
# Complete AWS Infrastructure Setup for Bearing Fault Classifier
# Updated: adds 2 private subnets + public/private route tables + NAT Gateway + associations
set -euo pipefail

# -------------------------
# Configuration (edit as needed)
# -------------------------
AWS_REGION="us-east-1"
APP_NAME="bearing-classifier"
CLUSTER_NAME="${APP_NAME}-cluster"
SERVICE_NAME="${APP_NAME}-service"
TASK_FAMILY="${APP_NAME}-task"
ECR_REPO="${APP_NAME}"
DOMAIN_NAME="api.opkcloudz.com"  # Your subdomain
S3_MODELS_BUCKET="opkcloudz-ml-models"

# Option A CIDRs (2 public + 2 private)
PUB_SUBNET_A_CIDR="10.0.1.0/24"
PUB_SUBNET_B_CIDR="10.0.2.0/24"
PRIV_SUBNET_A_CIDR="10.0.3.0/24"
PRIV_SUBNET_B_CIDR="10.0.4.0/24"
VPC_CIDR="10.0.0.0/16"

echo "🚀 Setting up AWS Infrastructure"
echo "=================================="
echo "Region: $AWS_REGION"
echo "App: $APP_NAME"
echo "Domain: $DOMAIN_NAME"
echo ""

# Ensure AWS CLI is available
if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found. Install and configure AWS CLI with credentials first."
  exit 1
fi

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")
echo "Account ID: $ACCOUNT_ID"
echo ""

# -------------------------
# 1. Create ECR Repository
# -------------------------
echo "📦 Creating ECR repository..."
aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    2>/dev/null || echo "  Repository already exists"

ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"
echo "  ECR URI: $ECR_URI"
echo ""

# -------------------------
# 2. Create S3 Bucket for Models
# -------------------------
echo "🗄️  Creating S3 bucket for models..."
aws s3 mb "s3://$S3_MODELS_BUCKET" --region "$AWS_REGION" 2>/dev/null || echo "  Bucket already exists"

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket "$S3_MODELS_BUCKET" \
    --versioning-configuration Status=Enabled \
    --region "$AWS_REGION"

echo "  S3 Bucket: $S3_MODELS_BUCKET"
echo ""

# -------------------------
# 3. Create ECS Cluster
# -------------------------
echo "🏗️  Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name "$CLUSTER_NAME" \
    --region "$AWS_REGION" \
    --capacity-providers FARGATE FARGATE_SPOT \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    2>/dev/null || echo "  Cluster already exists"
echo ""

# -------------------------
# 4. Create CloudWatch Log Group
# -------------------------
echo "📊 Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name "/ecs/$TASK_FAMILY" \
    --region "$AWS_REGION" 2>/dev/null || echo "  Log group already exists"

# Set retention (days)
aws logs put-retention-policy \
    --log-group-name "/ecs/$TASK_FAMILY" \
    --retention-in-days 7 \
    --region "$AWS_REGION"
echo ""

# -------------------------
# 5. Create IAM Roles
# -------------------------
echo "🔐 Creating IAM roles..."

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
    --role-name "${APP_NAME}-ecs-task-execution-role" \
    --assume-role-policy-document file:///tmp/ecs-task-execution-trust.json \
    2>/dev/null || echo "  Execution role already exists"

aws iam attach-role-policy \
    --role-name "${APP_NAME}-ecs-task-execution-role" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy || true

aws iam create-role \
    --role-name "${APP_NAME}-ecs-task-role" \
    --assume-role-policy-document file:///tmp/ecs-task-execution-trust.json \
    2>/dev/null || echo "  Task role already exists"

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
    --role-name "${APP_NAME}-ecs-task-role" \
    --policy-name S3ModelsAccess \
    --policy-document file:///tmp/s3-access-policy.json || true

echo ""

# -------------------------
# 6. Create VPC + 4 subnets (2 public + 2 private) or reuse default VPC + create private subnets
# -------------------------
echo "🌐 Setting up networking..."

# Check if default VPC exists
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region "$AWS_REGION" || true)

if [ -z "$DEFAULT_VPC" ] || [ "$DEFAULT_VPC" = "None" ]; then
    echo "  Creating new VPC..."
    VPC_ID=$(aws ec2 create-vpc --cidr-block "$VPC_CIDR" --query Vpc.VpcId --output text --region "$AWS_REGION")
    aws ec2 create-tags --resources "$VPC_ID" --tags Key=Name,Value="${APP_NAME}-vpc" --region "$AWS_REGION"

    # Create 2 public and 2 private subnets
    SUBNET_PUB_A=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$PUB_SUBNET_A_CIDR" --availability-zone "${AWS_REGION}a" --query Subnet.SubnetId --output text --region "$AWS_REGION")
    SUBNET_PUB_B=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$PUB_SUBNET_B_CIDR" --availability-zone "${AWS_REGION}b" --query Subnet.SubnetId --output text --region "$AWS_REGION")
    SUBNET_PRIV_A=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$PRIV_SUBNET_A_CIDR" --availability-zone "${AWS_REGION}a" --query Subnet.SubnetId --output text --region "$AWS_REGION")
    SUBNET_PRIV_B=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$PRIV_SUBNET_B_CIDR" --availability-zone "${AWS_REGION}b" --query Subnet.SubnetId --output text --region "$AWS_REGION")

    # create internet gateway and attach
    IGW=$(aws ec2 create-internet-gateway --query InternetGateway.InternetGatewayId --output text --region "$AWS_REGION")
    aws ec2 attach-internet-gateway --vpc-id "$VPC_ID" --internet-gateway-id "$IGW" --region "$AWS_REGION"

else
    echo "  Using default VPC: $DEFAULT_VPC"
    VPC_ID=$DEFAULT_VPC

    # Grab two default subnets (will be used as public subnets for ALB/NAT)
    SUBNETS_LIST=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[?MapPublicIpOnLaunch==\`true\`].SubnetId" --output text --region "$AWS_REGION" || true)
    # Fallback to any subnets if MapPublicIpOnLaunch filter returns empty
    if [ -z "$SUBNETS_LIST" ]; then
      SUBNETS_LIST=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[].SubnetId" --output text --region "$AWS_REGION")
    fi

    # pick first two as public subnets
    SUBNET_PUB_A=$(echo "$SUBNETS_LIST" | awk '{print $1}')
    SUBNET_PUB_B=$(echo "$SUBNETS_LIST" | awk '{print $2}')

    # create two private subnets in the same VPC (Option A CIDRs)
    SUBNET_PRIV_A=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$PRIV_SUBNET_A_CIDR" --availability-zone "${AWS_REGION}a" --query Subnet.SubnetId --output text --region "$AWS_REGION")
    SUBNET_PRIV_B=$(aws ec2 create-subnet --vpc-id "$VPC_ID" --cidr-block "$PRIV_SUBNET_B_CIDR" --availability-zone "${AWS_REGION}b" --query Subnet.SubnetId --output text --region "$AWS_REGION")

    # find or create IGW (attach if not attached)
    IGW=$(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=${VPC_ID}" --query "InternetGateways[0].InternetGatewayId" --output text --region "$AWS_REGION" || true)
    if [ -z "$IGW" ] || [ "$IGW" = "None" ]; then
        IGW=$(aws ec2 create-internet-gateway --query InternetGateway.InternetGatewayId --output text --region "$AWS_REGION")
        aws ec2 attach-internet-gateway --vpc-id "$VPC_ID" --internet-gateway-id "$IGW" --region "$AWS_REGION"
    fi
fi

echo "  VPC ID: $VPC_ID"
echo "  Public Subnet A: $SUBNET_PUB_A"
echo "  Public Subnet B: $SUBNET_PUB_B"
echo "  Private Subnet A: $SUBNET_PRIV_A"
echo "  Private Subnet B: $SUBNET_PRIV_B"
echo "  Internet Gateway: $IGW"
echo ""

# -------------------------
# 7. Route Tables + NAT Gateway + Associations (public/private)
# -------------------------
echo "🛣️ Setting up Route Tables + NAT Gateway..."
echo ""

# Create or find Public Route Table
if PUBLIC_RT=$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" "Name=association.subnet-id,Values=$SUBNET_PUB_A" --query "RouteTables[0].RouteTableId" --output text --region "$AWS_REGION" 2>/dev/null); then
  :
fi

if [ -z "$PUBLIC_RT" ] || [ "$PUBLIC_RT" = "None" ]; then
  PUBLIC_RT=$(aws ec2 create-route-table --vpc-id "$VPC_ID" --query 'RouteTable.RouteTableId' --output text --region "$AWS_REGION")
  aws ec2 create-tags --resources "$PUBLIC_RT" --tags Key=Name,Value="${APP_NAME}-public-rt" --region "$AWS_REGION"
fi
echo "  Public Route Table: $PUBLIC_RT"

# Ensure route 0.0.0.0/0 -> IGW exists on public RT
set +e
aws ec2 create-route --route-table-id "$PUBLIC_RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW" --region "$AWS_REGION" >/dev/null 2>&1
set -e
echo "  Added route: 0.0.0.0/0 → IGW (if missing)"

# Associate public subnets with public RT
aws ec2 associate-route-table --subnet-id "$SUBNET_PUB_A" --route-table-id "$PUBLIC_RT" --region "$AWS_REGION" || true
aws ec2 associate-route-table --subnet-id "$SUBNET_PUB_B" --route-table-id "$PUBLIC_RT" --region "$AWS_REGION" || true
echo "  Associated public subnets with public RT"

# Allocate Elastic IP for NAT (idempotent check)
echo "🌐 Allocating Elastic IP for NAT Gateway..."
# Try to find an existing allocation tagged for this app
NAT_EIP=$(aws ec2 describe-addresses --filters "Name=tag:Name,Values=${APP_NAME}-nat-eip" --query "Addresses[0].AllocationId" --output text --region "$AWS_REGION" 2>/dev/null || true)
if [ -z "$NAT_EIP" ] || [ "$NAT_EIP" = "None" ]; then
  NAT_EIP=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text --region "$AWS_REGION")
  # tag allocation (best-effort)
  aws ec2 create-tags --resources "$NAT_EIP" --tags Key=Name,Value="${APP_NAME}-nat-eip" --region "$AWS_REGION" || true
fi
echo "  NAT EIP Allocation ID: $NAT_EIP"

# Create NAT Gateway in Public Subnet A (idempotent attempt)
echo "🚇 Creating NAT Gateway..."
NAT_GW=$(aws ec2 describe-nat-gateways --filter Name=subnet-id,Values="$SUBNET_PUB_A" --query "NatGateways[?State=='available'] | [0].NatGatewayId" --output text --region "$AWS_REGION" 2>/dev/null || true)
if [ -z "$NAT_GW" ] || [ "$NAT_GW" = "None" ]; then
  NAT_GW=$(aws ec2 create-nat-gateway --subnet-id "$SUBNET_PUB_A" --allocation-id "$NAT_EIP" --query 'NatGateway.NatGatewayId' --output text --region "$AWS_REGION")
fi
echo "  NAT Gateway ID: $NAT_GW"

# Wait for NAT Gateway to become available
echo "⏳ Waiting for NAT Gateway to become available..."
aws ec2 wait nat-gateway-available --nat-gateway-ids "$NAT_GW" --region "$AWS_REGION"
echo "  NAT Gateway is available!"
echo ""

# Create Private Route Table
PRIVATE_RT=$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:Name,Values=${APP_NAME}-private-rt" --query "RouteTables[0].RouteTableId" --output text --region "$AWS_REGION" 2>/dev/null || true)
if [ -z "$PRIVATE_RT" ] || [ "$PRIVATE_RT" = "None" ]; then
  PRIVATE_RT=$(aws ec2 create-route-table --vpc-id "$VPC_ID" --query 'RouteTable.RouteTableId' --output text --region "$AWS_REGION")
  aws ec2 create-tags --resources "$PRIVATE_RT" --tags Key=Name,Value="${APP_NAME}-private-rt" --region "$AWS_REGION"
fi
echo "  Private Route Table: $PRIVATE_RT"

# Add route 0.0.0.0/0 -> NAT in private RT
set +e
aws ec2 create-route --route-table-id "$PRIVATE_RT" --destination-cidr-block 0.0.0.0/0 --nat-gateway-id "$NAT_GW" --region "$AWS_REGION" >/dev/null 2>&1
set -e
echo "  Added route: 0.0.0.0/0 → NAT Gateway (if missing)"

# Associate PRIVATE subnets with PRIVATE RT
aws ec2 associate-route-table --subnet-id "$SUBNET_PRIV_A" --route-table-id "$PRIVATE_RT" --region "$AWS_REGION" || true
aws ec2 associate-route-table --subnet-id "$SUBNET_PRIV_B" --route-table-id "$PRIVATE_RT" --region "$AWS_REGION" || true
echo "  Associated private subnets with private RT"
echo ""

# -------------------------
# 8. Create Security Groups
# -------------------------
echo "🔒 Creating security groups..."

# ALB Security Group
ALB_SG=$(aws ec2 create-security-group \
    --group-name "${APP_NAME}-alb-sg" \
    --description "Security group for ${APP_NAME} ALB" \
    --vpc-id "$VPC_ID" \
    --output text \
    --query 'GroupId' \
    --region "$AWS_REGION" 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=${APP_NAME}-alb-sg" "Name=vpc-id,Values=$VPC_ID" \
        --query "SecurityGroups[0].GroupId" \
        --output text \
        --region "$AWS_REGION")

# Allow HTTP and HTTPS
aws ec2 authorize-security-group-ingress \
    --group-id "$ALB_SG" \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region "$AWS_REGION" 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-id "$ALB_SG" \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 \
    --region "$AWS_REGION" 2>/dev/null || true

# ECS Security Group
ECS_SG=$(aws ec2 create-security-group \
    --group-name "${APP_NAME}-ecs-sg" \
    --description "Security group for ${APP_NAME} ECS tasks" \
    --vpc-id "$VPC_ID" \
    --output text \
    --query 'GroupId' \
    --region "$AWS_REGION" 2>/dev/null || \
    aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=${APP_NAME}-ecs-sg" "Name=vpc-id,Values=$VPC_ID" \
        --query "SecurityGroups[0].GroupId" \
        --output text \
        --region "$AWS_REGION")

# Allow traffic from ALB to ECS tasks on port 8000
aws ec2 authorize-security-group-ingress \
    --group-id "$ECS_SG" \
    --protocol tcp \
    --port 8000 \
    --source-group "$ALB_SG" \
    --region "$AWS_REGION" 2>/dev/null || true

echo "  ALB Security Group: $ALB_SG"
echo "  ECS Security Group: $ECS_SG"
echo ""

# -------------------------
# 9. Create Application Load Balancer (in public subnets)
# -------------------------
echo "⚖️  Creating Application Load Balancer..."

ALB_ARN=$(aws elbv2 create-load-balancer \
    --name "${APP_NAME}-alb" \
    --subnets "$SUBNET_PUB_A" "$SUBNET_PUB_B" \
    --security-groups "$ALB_SG" \
    --region "$AWS_REGION" \
    --output text \
    --query 'LoadBalancers[0].LoadBalancerArn' 2>/dev/null || \
    aws elbv2 describe-load-balancers \
        --names "${APP_NAME}-alb" \
        --query "LoadBalancers[0].LoadBalancerArn" \
        --output text \
        --region "$AWS_REGION")

# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns "$ALB_ARN" \
    --query "LoadBalancers[0].DNSName" \
    --output text \
    --region "$AWS_REGION")

echo "  ALB ARN: $ALB_ARN"
echo "  ALB DNS: $ALB_DNS"
echo ""

# -------------------------
# 10. Create Target Group
# -------------------------
echo "🎯 Creating target group..."

TG_ARN=$(aws elbv2 create-target-group \
    --name "${APP_NAME}-tg" \
    --protocol HTTP \
    --port 8000 \
    --vpc-id "$VPC_ID" \
    --target-type ip \
    --health-check-enabled \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --region "$AWS_REGION" \
    --output text \
    --query 'TargetGroups[0].TargetGroupArn' 2>/dev/null || \
    aws elbv2 describe-target-groups \
        --names "${APP_NAME}-tg" \
        --query "TargetGroups[0].TargetGroupArn" \
        --output text \
        --region "$AWS_REGION")

echo "  Target Group ARN: $TG_ARN"
echo ""

# -------------------------
# 11. Create ALB Listener (HTTP)
# -------------------------
echo "👂 Creating ALB listener..."
aws elbv2 create-listener \
    --load-balancer-arn "$ALB_ARN" \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn="$TG_ARN" \
    --region "$AWS_REGION" 2>/dev/null || echo "  Listener already exists"
echo ""

# -------------------------
# 12. Save configuration
# -------------------------
mkdir -p aws-infrastructure

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
SUBNET_PUB_A=$SUBNET_PUB_A
SUBNET_PUB_B=$SUBNET_PUB_B
SUBNET_PRIV_A=$SUBNET_PRIV_A
SUBNET_PRIV_B=$SUBNET_PRIV_B
PUBLIC_RT=$PUBLIC_RT
PRIVATE_RT=$PRIVATE_RT
NAT_GW=$NAT_GW
NAT_EIP=$NAT_EIP
IGW=$IGW
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
echo "  4. Create ECS service: ./aws-infrastructure/create-service.sh (ensure you use private subnets)"
echo "  5. Configure Route 53: ./aws-infrastructure/configure-dns.sh"
echo ""
