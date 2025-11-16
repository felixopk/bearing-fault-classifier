#!/bin/bash
# Create ECS Task Definition

set -e

# Load configuration
source aws-infrastructure/config.env

MODEL_VERSION="1.0.0"

echo "📋 Creating ECS task definition..."

# Create task definition JSON
cat > /tmp/task-definition.json << TASKDEF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/${APP_NAME}-ecs-task-execution-role",
  "taskRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/${APP_NAME}-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "${APP_NAME}-container",
      "image": "${ECR_URI}:latest",
      "cpu": 512,
      "memory": 1024,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "PYTHONUNBUFFERED",
          "value": "1"
        },
        {
          "name": "LOG_LEVEL",
          "value": "info"
        },
        {
          "name": "MODEL_URL",
          "value": "https://${S3_MODELS_BUCKET}.s3.${AWS_REGION}.amazonaws.com/bearing-classifier/v${MODEL_VERSION}/random_forest_model.pkl"
        },
        {
          "name": "SCALER_URL",
          "value": "https://${S3_MODELS_BUCKET}.s3.${AWS_REGION}.amazonaws.com/bearing-classifier/v${MODEL_VERSION}/random_forest_scaler.pkl"
        },
        {
          "name": "MODEL_VERSION",
          "value": "${MODEL_VERSION}"
        },
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "${AWS_REGION}"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${TASK_FAMILY}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
TASKDEF

# Register task definition
TASK_DEF_ARN=$(aws ecs register-task-definition \
    --cli-input-json file:///tmp/task-definition.json \
    --region $AWS_REGION \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo "✅ Task definition registered: $TASK_DEF_ARN"
echo ""

# Save to config
echo "TASK_DEF_ARN=$TASK_DEF_ARN" >> aws-infrastructure/config.env
