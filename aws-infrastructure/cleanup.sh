#!/bin/bash
# Cleanup all AWS resources

set -e

echo "🗑️  AWS Resource Cleanup"
echo "======================="
echo ""
echo "⚠️  WARNING: This will delete all resources!"
echo "This includes:"
echo "  - ECS Service and Cluster"
echo "  - Load Balancer and Target Groups"
echo "  - ECR Repository and images"
echo "  - Security Groups"
echo "  - CloudWatch Log Groups"
echo "  - S3 Models (optional)"
echo ""

read -p "Are you sure you want to continue? (yes/no) " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Cleanup cancelled"
    exit 0
fi

# Load configuration
if [ ! -f "aws-infrastructure/config.env" ]; then
    echo "❌ config.env not found. Have you run setup-infrastructure.sh?"
    exit 1
fi

source aws-infrastructure/config.env

echo ""
echo "Starting cleanup..."
echo ""

# 1. Delete ECS Service
echo "🗑️  Deleting ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --desired-count 0 \
    --region $AWS_REGION 2>/dev/null || true

aws ecs delete-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --force \
    --region $AWS_REGION 2>/dev/null || true

echo "⏳ Waiting for service deletion..."
sleep 10

# 2. Delete ECS Cluster
echo "🗑️  Deleting ECS cluster..."
aws ecs delete-cluster \
    --cluster $CLUSTER_NAME \
    --region $AWS_REGION 2>/dev/null || true

# 3. Delete ALB Listener
echo "🗑️  Deleting ALB listeners..."
LISTENER_ARNS=$(aws elbv2 describe-listeners \
    --load-balancer-arn $ALB_ARN \
    --query 'Listeners[*].ListenerArn' \
    --output text \
    --region $AWS_REGION 2>/dev/null || true)

for LISTENER_ARN in $LISTENER_ARNS; do
    aws elbv2 delete-listener \
        --listener-arn $LISTENER_ARN \
        --region $AWS_REGION 2>/dev/null || true
done

# 4. Delete Target Group
echo "🗑️  Deleting target group..."
aws elbv2 delete-target-group \
    --target-group-arn $TG_ARN \
    --region $AWS_REGION 2>/dev/null || true

# 5. Delete Load Balancer
echo "🗑️  Deleting load balancer..."
aws elbv2 delete-load-balancer \
    --load-balancer-arn $ALB_ARN \
    --region $AWS_REGION 2>/dev/null || true

echo "⏳ Waiting for load balancer deletion..."
sleep 30

# 6. Delete Security Groups
echo "🗑️  Deleting security groups..."
aws ec2 delete-security-group \
    --group-id $ECS_SG \
    --region $AWS_REGION 2>/dev/null || true

aws ec2 delete-security-group \
    --group-id $ALB_SG \
    --region $AWS_REGION 2>/dev/null || true

# 7. Delete CloudWatch Log Group
echo "��️  Deleting CloudWatch log group..."
aws logs delete-log-group \
    --log-group-name /ecs/$TASK_FAMILY \
    --region $AWS_REGION 2>/dev/null || true

# 8. Delete ECR Repository
echo "🗑️  Deleting ECR repository..."
aws ecr delete-repository \
    --repository-name $ECR_REPO \
    --force \
    --region $AWS_REGION 2>/dev/null || true

# 9. Optional: Delete S3 bucket
echo ""
read -p "Delete S3 models bucket? This will delete all your trained models! (yes/no) " -r
if [[ $REPLY =~ ^yes$ ]]; then
    echo "🗑️  Deleting S3 bucket..."
    aws s3 rb s3://$S3_MODELS_BUCKET --force --region $AWS_REGION 2>/dev/null || true
fi

# 10. Delete IAM Roles
echo "🗑️  Detaching and deleting IAM roles..."

# Detach policies from execution role
aws iam detach-role-policy \
    --role-name ${APP_NAME}-ecs-task-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
    2>/dev/null || true

aws iam delete-role \
    --role-name ${APP_NAME}-ecs-task-execution-role \
    2>/dev/null || true

# Delete inline policy and task role
aws iam delete-role-policy \
    --role-name ${APP_NAME}-ecs-task-role \
    --policy-name S3ModelsAccess \
    2>/dev/null || true

aws iam delete-role \
    --role-name ${APP_NAME}-ecs-task-role \
    2>/dev/null || true

# 11. Remove DNS record (optional)
read -p "Remove Route 53 DNS record for $DOMAIN_NAME? (yes/no) " -r
if [[ $REPLY =~ ^yes$ ]]; then
    echo "🗑️  Removing DNS record..."
    
    HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name \
        --dns-name opkcloudz.com \
        --query "HostedZones[0].Id" \
        --output text | cut -d'/' -f3)
    
    if [ ! -z "$HOSTED_ZONE_ID" ]; then
        cat > /tmp/dns-delete.json << DNS
{
  "Changes": [
    {
      "Action": "DELETE",
      "ResourceRecordSet": {
        "Name": "$DOMAIN_NAME",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "$(aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN --query "LoadBalancers[0].CanonicalHostedZoneId" --output text --region $AWS_REGION)",
          "DNSName": "$ALB_DNS",
          "EvaluateTargetHealth": true
        }
      }
    }
  ]
}
DNS
        
        aws route53 change-resource-record-sets \
            --hosted-zone-id $HOSTED_ZONE_ID \
            --change-batch file:///tmp/dns-delete.json \
            2>/dev/null || true
    fi
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Note: Some resources may take a few minutes to fully delete."
echo "You can verify in the AWS Console."
echo ""
