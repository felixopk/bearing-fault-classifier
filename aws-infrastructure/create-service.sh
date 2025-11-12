#!/bin/bash
# Create ECS Service with Load Balancer

set -e

AWS_REGION="us-east-1"
CLUSTER_NAME="bearing-classifier-cluster"
SERVICE_NAME="bearing-classifier-service"
TASK_FAMILY="bearing-classifier-task"

# You need to create a VPC, subnets, and security group first
# Replace these with your actual IDs
SUBNET_1="subnet-xxxxxxxxx"
SUBNET_2="subnet-yyyyyyyyy"
SECURITY_GROUP="sg-zzzzzzzzz"

echo "🚀 Creating ECS Service..."

# Register task definition
TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region $AWS_REGION \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo "✅ Task definition registered: $TASK_DEFINITION_ARN"

# Create service
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}" \
    --region $AWS_REGION

echo "✅ Service created successfully!"
