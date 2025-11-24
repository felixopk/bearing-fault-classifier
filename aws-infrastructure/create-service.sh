#!/bin/bash
set -e

echo "🚀 Creating ECS Service..."

# Load config
source aws-infrastructure/config.env

CONTAINER_NAME="bearing-classifier-container"
CONTAINER_PORT=8000

echo "Cluster:    $CLUSTER_NAME"
echo "Service:    $SERVICE_NAME"
echo "Task Def:   $TASK_FAMILY"
echo "TG ARN:     $TG_ARN"
echo "Subnets:    $SUBNET_PRIV_A, $SUBNET_PRIV_B"
echo "SG:         $ECS_SG"
echo "Container:  $CONTAINER_NAME"

# Check if service exists first (avoid duplicate creation)
SERVICE_EXISTS=$(aws ecs describe-services \
  --cluster "$CLUSTER_NAME" \
  --services "$SERVICE_NAME" \
  --query "services[0].status" \
  --output text 2>/dev/null || echo "MISSING")

if [ "$SERVICE_EXISTS" != "MISSING" ]; then
    echo "⚠️ Service already exists. Skipping creation."
    exit 0
fi

echo "🛠️ Creating new ECS service..."

aws ecs create-service \
  --cluster "$CLUSTER_NAME" \
  --service-name "$SERVICE_NAME" \
  --task-definition "$TASK_FAMILY" \
  --desired-count 1 \
  --launch-type FARGATE \
  --platform-version "LATEST" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_PRIV_A,$SUBNET_PRIV_B],securityGroups=[$ECS_SG],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=$CONTAINER_NAME,containerPort=$CONTAINER_PORT" \
  --region "$AWS_REGION"

echo ""
echo "✅ ECS Service created successfully!"
echo "You can now deploy using GitHub Actions."
