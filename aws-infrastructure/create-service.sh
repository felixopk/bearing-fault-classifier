#!/bin/bash
# Create ECS Service

set -e

# Load configuration
source aws-infrastructure/config.env

echo "🚀 Creating ECS service..."

# Create service
SERVICE_ARN=$(aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --desired-count 1 \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET1,$SUBNET2],securityGroups=[$ECS_SG],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=$TG_ARN,containerName=${APP_NAME}-container,containerPort=8000" \
    --health-check-grace-period-seconds 60 \
    --region $AWS_REGION \
    --query 'service.serviceArn' \
    --output text 2>/dev/null || \
    aws ecs describe-services \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --query "services[0].serviceArn" \
        --output text \
        --region $AWS_REGION)

echo "✅ Service created: $SERVICE_ARN"
echo ""
echo "⏳ Waiting for service to become stable (this may take 2-3 minutes)..."

aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION

echo "✅ Service is stable and running!"
echo ""
echo "🔗 Your API is now accessible at: http://$ALB_DNS"
echo ""
echo "Test with:"
echo "  curl http://$ALB_DNS/health"
echo ""
