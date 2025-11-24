#!/bin/bash
set -euo pipefail

source aws-infrastructure/config.env

echo "🚀 Creating ECS Service..."

aws ecs create-service \
  --cluster "$CLUSTER_NAME" \
  --service-name "$SERVICE_NAME" \
  --task-definition "$TASK_FAMILY" \
  --launch-type FARGATE \
  --desired-count 1 \
  --load-balancers "[
    {
      \"targetGroupArn\": \"${TG_ARN}\",
      \"containerName\": \"${APP_NAME}\",
      \"containerPort\": 8000
    }
  ]" \
  --network-configuration "awsvpcConfiguration={
      subnets=[\"${SUBNET_PRIV_A}\",\"${SUBNET_PRIV_B}\"],
      securityGroups=[\"${ECS_SG}\"],
      assignPublicIp=\"DISABLED\"
  }" \
  --region "$AWS_REGION"

echo "✅ ECS Service created!"
echo "🌐 Your API will be available at https://$DOMAIN_NAME"
