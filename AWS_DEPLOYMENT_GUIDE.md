# AWS Deployment Guide - Bearing Fault Classifier

Complete guide to deploy your ML API to AWS with custom domain.

## 🚀 Quick Start (One Command)
```bash
# Deploy everything automatically
./aws-infrastructure/deploy-all.sh
```

This will:
1. ✅ Create all AWS infrastructure (ECS, ALB, ECR, etc.)
2. ✅ Upload models to S3
3. ✅ Build and push Docker image to ECR
4. ✅ Deploy ECS service
5. ✅ Configure DNS (optional)

**Time:** ~10-15 minutes  
**Cost:** ~$15-30/month

---

## 📋 Prerequisites

### 1. AWS CLI Configured
```bash
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: us-east-1
# - Default output format: json

# Verify
aws sts get-caller-identity
```

### 2. Docker Running
```bash
docker --version
docker ps
```

### 3. Models Trained
```bash
# Make sure you have these files:
ls -lh models/random_forest_model.pkl
ls -lh models/random_forest_scaler.pkl
```

---

## 🔧 Manual Step-by-Step Deployment

### Step 1: Setup Infrastructure
```bash
./aws-infrastructure/setup-infrastructure.sh
```

**Creates:**
- ECR Repository
- ECS Cluster
- S3 Bucket for models
- VPC & Subnets (if needed)
- Security Groups
- Application Load Balancer
- Target Group
- IAM Roles

### Step 2: Upload Models
```bash
./aws-infrastructure/upload-models.sh
```

**Uploads to S3:**
- `random_forest_model.pkl`
- `random_forest_scaler.pkl`

### Step 3: Build & Push Docker Image
```bash
./aws-infrastructure/build-and-push.sh
```

**Actions:**
- Builds Docker image
- Pushes to ECR with tags `latest` and `<git-sha>`

### Step 4: Create Task Definition
```bash
./aws-infrastructure/create-task-definition.sh
```

**Configures:**
- Container specs (512 CPU, 1024 MB RAM)
- Environment variables
- Health checks
- Logging

### Step 5: Create ECS Service
```bash
./aws-infrastructure/create-service.sh
```

**Launches:**
- 1 Fargate task
- Connected to ALB
- Auto-scaling enabled

### Step 6: Configure DNS
```bash
./aws-infrastructure/configure-dns.sh
```

**Sets up:**
- Route 53 A record
- Points `api.opkclodz.com` to ALB

---

## 🧪 Testing Your Deployment

### Get Your URLs
```bash
# Load configuration
source aws-infrastructure/config.env

# Your ALB DNS
echo "ALB: http://$ALB_DNS"

# Your custom domain (after DNS setup)
echo "Domain: http://api.opkclodz.com"
```

### Test Endpoints
```bash
# Health check
curl http://api.opkclodz.com/health

# Model info
curl http://api.opkclodz.com/model/info

# Make prediction
curl -X POST http://api.opkclodz.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
  }'

# API Documentation
open http://api.opkclodz.com/docs
```

---

## 🔄 CI/CD with GitHub Actions

### Setup GitHub Secrets

Go to: **GitHub Repo → Settings → Secrets and variables → Actions**

Add these secrets:

1. **AWS_ACCESS_KEY_ID** - Your AWS access key
2. **AWS_SECRET_ACCESS_KEY** - Your AWS secret key

### Automated Deployment Flow
```
Push to main → GitHub Actions → Build Image → Push to ECR → Deploy to ECS → Health Check
```

**On every push to `main`:**
1. ✅ Tests run
2. ✅ Docker image builds
3. ✅ Image pushes to ECR
4. ✅ ECS service updates
5. ✅ Health check verifies deployment

---

## 📊 Monitoring & Logs

### View Logs
```bash
# Via AWS CLI
aws logs tail /ecs/bearing-classifier-task --follow --region us-east-1

# Via Console
# https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group//ecs/bearing-classifier-task
```

### Monitor Service
```bash
# Service status
aws ecs describe-services \
  --cluster bearing-classifier-cluster \
  --services bearing-classifier-service \
  --region us-east-1

# Running tasks
aws ecs list-tasks \
  --cluster bearing-classifier-cluster \
  --service bearing-classifier-service \
  --region us-east-1
```

### CloudWatch Metrics

Go to: CloudWatch → Container Insights → bearing-classifier-cluster

**Metrics available:**
- CPU Utilization
- Memory Utilization
- Network I/O
- Request Count

---

## 🔧 Updating Your Application

### Update Code Only
```bash
# Make changes
git add .
git commit -m "feat: update API endpoint"
git push origin main

# GitHub Actions automatically deploys
```

### Update Models
```bash
# 1. Train new models
python scripts/train_model.py

# 2. Upload to S3 with new version
MODEL_VERSION="1.1.0"
aws s3 cp models/random_forest_model.pkl \
  s3://opkclodz-ml-models/bearing-classifier/v${MODEL_VERSION}/random_forest_model.pkl

# 3. Update task definition environment variables
# Update MODEL_URL to point to new version

# 4. Force new deployment
aws ecs update-service \
  --cluster bearing-classifier-cluster \
  --service bearing-classifier-service \
  --force-new-deployment \
  --region us-east-1
```

### Manual Redeploy
```bash
# Force new deployment with latest image
aws ecs update-service \
  --cluster bearing-classifier-cluster \
  --service bearing-classifier-service \
  --force-new-deployment \
  --region us-east-1
```

---

## 🔐 Security Best Practices

### 1. Use HTTPS (Optional but Recommended)
```bash
# Request ACM certificate
aws acm request-certificate \
  --domain-name api.opkclodz.com \
  --validation-method DNS \
  --region us-east-1

# Add HTTPS listener to ALB
# (See AWS Console for certificate validation)
```

### 2. Restrict S3 Access
```bash
# Use IAM roles instead of public S3 URLs
# Already configured in setup script
```

### 3. Enable Container Insights
```bash
aws ecs update-cluster-settings \
  --cluster bearing-classifier-cluster \
  --settings name=containerInsights,value=enabled \
  --region us-east-1
```

---

## 💰 Cost Breakdown

**Monthly Estimate: $15-30**

| Service | Cost | Notes |
|---------|------|-------|
| **Fargate** | ~$15 | 1 task, 0.5 vCPU, 1GB RAM, 24/7 |
| **ALB** | ~$16 | Fixed cost |
| **ECR** | <$1 | Storage for images |
| **S3** | <$1 | Model storage (~50MB) |
| **Data Transfer** | ~$1 | First 1GB free |
| **Route 53** | $0.50 | Hosted zone |
| **CloudWatch Logs** | <$1 | 1GB free tier |

**Cost Optimization:**
- Use FARGATE_SPOT for 70% savings
- Scale to 0 tasks during off-hours
- Use S3 Intelligent-Tiering

---

## 🗑️ Cleanup (Delete Everything)
```bash
# Delete all resources
./aws-infrastructure/cleanup.sh

# This removes:
# - ECS Service & Cluster
# - Load Balancer & Target Groups
# - Security Groups
# - ECR Repository
# - CloudWatch Logs
# - IAM Roles
# - S3 Bucket (optional)
# - DNS Records (optional)
```

**⚠️ Warning:** This is irreversible!

---

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check service events
aws ecs describe-services \
  --cluster bearing-classifier-cluster \
  --services bearing-classifier-service \
  --query 'services[0].events[0:5]' \
  --region us-east-1

# Check task logs
aws logs tail /ecs/bearing-classifier-task --follow
```

### Health Check Failing
```bash
# Check container logs
aws logs tail /ecs/bearing-classifier-task --follow | grep -i error

# Common issues:
# - Models not downloading from S3
# - Port mismatch
# - Missing environment variables
```

### Can't Access via Domain
```bash
# Check DNS propagation
dig api.opkclodz.com

# Should show A record pointing to ALB

# Check ALB health
aws elbv2 describe-target-health \
  --target-group-arn $(cat aws-infrastructure/config.env | grep TG_ARN | cut -d'=' -f2) \
  --region us-east-1
```

---

## 📞 Support

- **AWS Documentation:** https://docs.aws.amazon.com/ecs/
- **Troubleshooting:** Check CloudWatch Logs first
- **Cost Calculator:** https://calculator.aws/

---

## 🎯 Next Steps

1. ✅ **Set up HTTPS** with ACM certificate
2. ✅ **Enable auto-scaling** based on CPU/Memory
3. ✅ **Add CloudWatch alarms** for monitoring
4. ✅ **Set up backup strategy** for models
5. ✅ **Implement blue/green deployments**

---

## 📝 Architecture Diagram
```
Internet
   ↓
Route 53 (api.opkclodz.com)
   ↓
Application Load Balancer
   ↓
ECS Fargate Tasks (Container)
   ├→ FastAPI Application
   └→ Downloads models from S3
```

**Data Flow:**
1. User makes request to api.opkclodz.com
2. Route 53 resolves to ALB
3. ALB forwards to ECS task
4. Container serves prediction
5. Logs go to CloudWatch

---

🎉 **Your ML API is now production-ready on AWS!**
