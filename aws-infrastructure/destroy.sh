#!/bin/bash
set -euo pipefail

echo "⚠️  WARNING: This will delete ALL resources created by this project!"
read -p "Type 'DELETE' to continue: " confirm

if [[ "$confirm" != "DELETE" ]]; then
  echo "❌ Destroy cancelled."
  exit 1
fi

source aws-infrastructure/config.env

echo "🧹 Starting resource cleanup..."
echo ""

# -----------------------------
# Delete ECS Service
# -----------------------------
echo "🛑 Deleting ECS service..."

aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$SERVICE_NAME" \
  --desired-count 0 \
  --region "$AWS_REGION" || true

aws ecs delete-service \
  --cluster "$CLUSTER_NAME" \
  --service "$SERVICE_NAME" \
  --force \
  --region "$AWS_REGION" || true

# -----------------------------
# Delete ECS Cluster
# -----------------------------
echo "🧨 Deleting ECS cluster..."
aws ecs delete-cluster \
  --cluster "$CLUSTER_NAME" \
  --region "$AWS_REGION" || true

# -----------------------------
# Delete Task Definitions
# -----------------------------
echo "🗑 Deregistering task definitions..."
for rev in $(aws ecs list-task-definitions \
               --family-prefix "$TASK_FAMILY" \
               --query "taskDefinitionArns[]" \
               --output text \
               --region "$AWS_REGION"); do
    aws ecs deregister-task-definition \
        --task-definition "$rev" \
        --region "$AWS_REGION" || true
done

# -----------------------------
# Delete ALB + Target Group
# -----------------------------
echo "🔥 Deleting ALB + Target Group..."

aws elbv2 delete-listener \
  --listener-arn $(aws elbv2 describe-listeners --load-balancer-arn "$ALB_ARN" --query "Listeners[].ListenerArn" --output text --region "$AWS_REGION") \
  --region "$AWS_REGION" || true

aws elbv2 delete-load-balancer \
  --load-balancer-arn "$ALB_ARN" \
  --region "$AWS_REGION" || true

aws elbv2 delete-target-group \
  --target-group-arn "$TG_ARN" \
  --region "$AWS_REGION" || true

# Wait for ALB deletion
echo "⏳ Waiting for ALB to delete..."
aws elbv2 wait load-balancer-deleted \
    --load-balancer-arn "$ALB_ARN" \
    --region "$AWS_REGION" || true

# -----------------------------
# Delete NAT Gateway
# -----------------------------
echo "📡 Deleting NAT Gateway..."
aws ec2 delete-nat-gateway \
  --nat-gateway-id "$NAT_GW" \
  --region "$AWS_REGION" || true

echo "⏳ Waiting for NAT Gateway deletion..."
aws ec2 wait nat-gateway-deleted \
  --nat-gateway-ids "$NAT_GW" \
  --region "$AWS_REGION" || true

# -----------------------------
# Delete Elastic IP
# -----------------------------
echo "💥 Releasing Elastic IP..."
aws ec2 release-address \
    --allocation-id "$NAT_EIP" \
    --region "$AWS_REGION" || true

# -----------------------------
# Delete Route Tables
# -----------------------------
echo "🗺 Removing route tables..."

aws ec2 delete-route-table \
   --route-table-id "$PUBLIC_RT" \
   --region "$AWS_REGION" || true

aws ec2 delete-route-table \
   --route-table-id "$PRIVATE_RT" \
   --region "$AWS_REGION" || true

# -----------------------------
# Delete Subnets
# -----------------------------
echo "🧱 Deleting subnets..."

aws ec2 delete-subnet --subnet-id "$SUBNET_PUB_A" --region "$AWS_REGION" || true
aws ec2 delete-subnet --subnet-id "$SUBNET_PUB_B" --region "$AWS_REGION" || true
aws ec2 delete-subnet --subnet-id "$SUBNET_PRIV_A" --region "$AWS_REGION" || true
aws ec2 delete-subnet --subnet-id "$SUBNET_PRIV_B" --region "$AWS_REGION" || true

# -----------------------------
# Detach + Delete Internet Gateway
# -----------------------------
echo "🌐 Removing Internet Gateway..."

aws ec2 detach-internet-gateway \
  --internet-gateway-id "$IGW" \
  --vpc-id "$VPC_ID" \
  --region "$AWS_REGION" || true

aws ec2 delete-internet-gateway \
  --internet-gateway-id "$IGW" \
  --region "$AWS_REGION" || true

# -----------------------------
# Delete VPC
# -----------------------------
echo "🏚  Deleting VPC..."
aws ec2 delete-vpc \
  --vpc-id "$VPC_ID" \
  --region "$AWS_REGION" || true

# -----------------------------
# Delete S3 bucket contents + bucket
# -----------------------------
echo "🪣 Deleting S3 bucket and objects..."

aws s3 rm "s3://$S3_MODELS_BUCKET" --recursive || true
aws s3 rb "s3://$S3_MODELS_BUCKET" --force || true

# -----------------------------
# Delete ECR repo
# -----------------------------
echo "🐳 Deleting ECR repository..."

aws ecr delete-repository \
  --repository-name "$ECR_REPO" \
  --force \
  --region "$AWS_REGION" || true

# -----------------------------
# Delete IAM Roles
# -----------------------------
echo "🔐 Deleting IAM roles..."

aws iam delete-role-policy \
  --role-name "${APP_NAME}-ecs-task-role" \
  --policy-name S3ModelsAccess || true

aws iam delete-role \
  --role-name "${APP_NAME}-ecs-task-role" || true

aws iam detach-role-policy \
  --role-name "${APP_NAME}-ecs-task-execution-role" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy || true

aws iam delete-role \
  --role-name "${APP_NAME}-ecs-task-execution-role" || true

echo ""
echo "🎉 All infrastructure resources have been destroyed successfully!"
