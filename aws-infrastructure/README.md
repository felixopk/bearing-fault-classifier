# AWS Deployment Guide

## Prerequisites

1. AWS CLI installed and configured
2. Docker installed
3. AWS account with appropriate permissions

## Quick Start

### Option 1: Manual Setup (Bash Scripts)
```bash
# 1. Set up infrastructure
cd aws-infrastructure
./setup-ecs.sh

# 2. Push Docker image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag bearing-fault-classifier:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/bearing-fault-classifier:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/bearing-fault-classifier:latest

# 3. Update task-definition.json with your account ID
# 4. Create service
./create-service.sh
```

### Option 2: Terraform (Infrastructure as Code)
```bash
cd aws-infrastructure
terraform init
terraform plan
terraform apply
```

## Environment Variables

Required GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

## Accessing the Application

After deployment, get the public IP:
```bash
aws ecs list-tasks --cluster bearing-classifier-cluster --service bearing-classifier-service
aws ecs describe-tasks --cluster bearing-classifier-cluster --tasks TASK_ARN
```

Then access: `http://PUBLIC_IP:8000`

## Monitoring

View logs:
```bash
aws logs tail /ecs/bearing-classifier-task --follow
```

## Cleanup
```bash
# Delete service
aws ecs delete-service --cluster bearing-classifier-cluster --service bearing-classifier-service --force

# Delete cluster
aws ecs delete-cluster --cluster bearing-classifier-cluster

# Delete ECR repository
aws ecr delete-repository --repository-name bearing-fault-classifier --force
```
