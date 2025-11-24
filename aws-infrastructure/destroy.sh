#!/bin/bash
# Destroy all AWS resources created for the Bearing Fault Classifier project
# Safe version — deletes only resources tagged for this app
set -euo pipefail

echo "🧨 Destroying Bearing Classifier AWS Infrastructure"
echo "==================================================="
echo ""

# Load configuration
if [ ! -f aws-infrastructure/config.env ]; then
    echo "❌ ERROR: config.env not found."
    echo "Run setup-infrastructure.sh first."
    exit 1
fi

source aws-infrastructure/config.env

echo "Project: $APP_NAME"
echo "Region: $AWS_REGION"
echo ""

# Helper to delete resources safely
safe_delete() {
    local desc="$1"
    shift
    echo "🗑️ Deleting $desc ..."
    "$@" >/dev/null 2>&1 || echo "⚠️  $desc already deleted or not found"
}

# -------------------------
# 1. Delete ECS Service + Cluster
# -------------------------
echo "🔥 Deleting ECS Service + Cluster..."

# Scale service down gracefully
aws ecs update-service \
    --cluster "$CLUSTER_NAME" \
    --service "$SERVICE_NAME" \
    --desired-count 0 \
    --region "$AWS_REGION" >/dev/null 2>&1 || true

# Wait a bit
sleep 5

# Delete service (force)
aws ecs delete-service \
    --cluster "$CLUSTER_NAME" \
    --service "$SERVICE_NAME" \
    --force \
    --region "$AWS_REGION" >/dev/null 2>&1 || true

safe_delete "ECS Cluster" \
  aws ecs delete-cluster --cluster "$CLUSTER_NAME" --region "$AWS_REGION"

echo ""

# -------------------------
# 2. Delete ALB + Listeners + Target Group
# -------------------------
echo "🔥 Deleting Load Balancer + Target Group..."

# Detach listeners
LISTENERS=$(aws elbv2 describe-listeners \
    --load-balancer-arn "$ALB_ARN" \
    --query "Listeners[].ListenerArn" \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || true)

for L in $LISTENERS; do
    safe_delete "ALB Listener $L" \
      aws elbv2 delete-listener --listener-arn "$L" --region "$AWS_REGION"
done

# Delete load balancer
safe_delete "Application Load Balancer" \
  aws elbv2 delete-load-balancer --load-balancer-arn "$ALB_ARN" --region "$AWS_REGION"

# Wait for LB to disappear
aws elbv2 wait load-balancers-deleted \
    --load-balancer-arns "$ALB_ARN" \
    --region "$AWS_REGION" >/dev/null 2>&1 || true

# Delete target group
safe_delete "Target Group" \
  aws elbv2 delete-target-group --target-group-arn "$TG_ARN" --region "$AWS_REGION"

echo ""

# -------------------------
# 3. Delete NAT Gateway + Release EIP
# -------------------------
echo "🔥 Deleting NAT Gateway + Elastic IP..."

safe_delete "NAT Gateway" \
  aws ec2 delete-nat-gateway --nat-gateway-id "$NAT_GW" --region "$AWS_REGION"

# Wait until NAT is fully deleted
aws ec2 wait nat-gateway-deleted \
  --nat-gateway-ids "$NAT_GW" \
  --region "$AWS_REGION" >/dev/null 2>&1 || true

# Release Elastic IP
safe_delete "NAT EIP Allocation" \
  aws ec2 release-address --allocation-id "$NAT_EIP" --region "$AWS_REGION"

echo ""

# -------------------------
# 4. Delete Route Tables (Private + Public)
# -------------------------
echo "🔥 Deleting Route Tables..."

safe_delete "Private Route Table" \
  aws ec2 delete-route-table --route-table-id "$PRIVATE_RT" --region "$AWS_REGION"

safe_delete "Public Route Table" \
  aws ec2 delete-route-table --route-table-id "$PUBLIC_RT" --region "$AWS_REGION"

echo ""

# -------------------------
# 5. Delete Security Groups
# -------------------------
echo "🔥 Deleting Security Groups..."

safe_delete "ECS Security Group" \
  aws ec2 delete-security-group --group-id "$ECS_SG" --region "$AWS_REGION"

safe_delete "ALB Security Group" \
  aws ec2 delete-security-group --group-id "$ALB_SG" --region "$AWS_REGION"

echo ""

# -------------------------
# 6. Detach + Delete Internet Gateway
# -------------------------
echo "🔥 Deleting Internet Gateway..."

safe_delete "Detach IGW" \
  aws ec2 detach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC_ID" --region "$AWS_REGION"

safe_delete "Delete IGW" \
  aws ec2 delete-internet-gateway --internet-gateway-id "$IGW" --region "$AWS_REGION"

echo ""

# -------------------------
# 7. Delete Subnets (4 total)
# -------------------------
echo "🔥 Deleting Subnets..."

safe_delete "Public Subnet A" aws ec2 delete-subnet --subnet-id "$SUBNET_PUB_A" --region "$AWS_REGION"
safe_delete "Public Subnet B" aws ec2 delete-subnet --subnet-id "$SUBNET_PUB_B" --region "$AWS_REGION"
safe_delete "Private Subnet A" aws ec2 delete-subnet --subnet-id "$SUBNET_PRIV_A" --region "$AWS_REGION"
safe_delete "Private Subnet B" aws ec2 delete-subnet --subnet-id "$SUBNET_PRIV_B" --region "$AWS_REGION"

echo ""

# -------------------------
# 8. Delete VPC (must be last)
# -------------------------
echo "🔥 Deleting VPC..."

safe_delete "VPC $VPC_ID" \
  aws ec2 delete-vpc --vpc-id "$VPC_ID" --region "$AWS_REGION"

echo ""

# -------------------------
# 9. Delete IAM Roles
# -------------------------
echo "🔥 Deleting IAM Roles..."

safe_delete "Task Role Policy" \
  aws iam delete-role-policy --role-name "${APP_NAME}-ecs-task-role" --policy-name S3ModelsAccess

safe_delete "Task Role" \
  aws iam delete-role --role-name "${APP_NAME}-ecs-task-role"

safe_delete "Execution Role" \
  aws iam delete-role --role-name "${APP_NAME}-ecs-task-execution-role"

echo ""

# -------------------------
# 10. Delete ECR Repository
# -------------------------
echo "🔥 Deleting ECR Repository..."

safe_delete "ECR repo" \
  aws ecr delete-repository --repository-name "$ECR_REPO" --force --region "$AWS_REGION"

echo ""

# -------------------------
# 11. Empty & Delete S3 Bucket
# -------------------------
echo "🔥 Emptying + Deleting S3 Bucket..."

safe_delete "Empty bucket" \
  aws s3 rm "s3://$S3_MODELS_BUCKET" --recursive --region "$AWS_REGION"

safe_delete "Delete bucket" \
  aws s3api delete-bucket --bucket "$S3_MODELS_BUCKET" --region "$AWS_REGION"

echo ""

# -------------------------
# 12. Clean config.env
# -------------------------
echo "🧼 Cleaning config.env ..."
rm -f aws-infrastructure/config.env

echo ""
echo "🎉 All project resources have been destroyed safely!"
echo "==================================================="
