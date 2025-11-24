#!/bin/bash
set -euo pipefail

source aws-infrastructure/config.env

echo "📦 Creating ECS Task Definition..."

aws ecs register-task-definition \
  --family "$TASK_FAMILY" \
  --execution-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/${APP_NAME}-ecs-task-execution-role" \
  --task-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/${APP_NAME}-ecs-task-role" \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu "512" \
  --memory "1024" \
  --container-definitions "[
     {
       \"name\": \"${APP_NAME}\",
       \"image\": \"${ECR_URI}:latest\",
       \"essential\": true,
       \"portMappings\": [
         {
           \"containerPort\": 8000,
           \"protocol\": \"tcp\"
         }
       ],
       \"logConfiguration\": {
         \"logDriver\": \"awslogs\",
         \"options\": {
           \"awslogs-group\": \"/ecs/${TASK_FAMILY}\",
           \"awslogs-region\": \"${AWS_REGION}\",
           \"awslogs-stream-prefix\": \"ecs\"
         }
       },
       \"environment\": [
         {\"name\": \"S3_BUCKET\", \"value\": \"${S3_MODELS_BUCKET}\"}
       ]
     }
  ]" \
  --region "$AWS_REGION"

echo "✅ Task definition created!"
