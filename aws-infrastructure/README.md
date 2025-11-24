🚀 Bearing Fault Classifier – Production AWS Deployment Guide

This repository deploys a production-grade, HTTPS-secured, auto-scalable ML API on AWS ECS Fargate using:

Custom VPC

2 public + 2 private subnets

Internet Gateway + NAT Gateway

Application Load Balancer (ALB)

Route53 DNS

ACM TLS Certificates

ECR for Docker images

S3 for ML models

IAM roles

CloudWatch logs

ECS Fargate service (private subnets)
                     Internet
                        │
                        ▼
                ┌─────────────────┐
                │  Route53 (DNS)  │
                └─────────────────┘
                        │  HTTPS
                        ▼
             ┌──────────────────────────┐
             │  Application Load Balancer│
             └──────────────────────────┘
                Public Subnets (A & B)
                        │
            forwards traffic on port 443
                        ▼
           ┌────────────────────────────┐
           │   ECS Fargate Service      │
           │   (Private Subnets A & B)  │
           └────────────────────────────┘
                        │
                        ▼
       ┌────────────────────────────────────┐
       │        S3 Bucket (ML Models)       │
       └────────────────────────────────────┘

    NAT Gateway → Internet Access for Private Subnets
📦 Folder Structure
aws-infrastructure/
│
├── setup-infrastructure.sh      # Full VPC + NAT + ALB + IAM setup
├── build-and-push.sh            # Build & push Docker image to ECR
├── create-task-definition.sh    # ECS task definition
├── create-service.sh            # ECS service (private subnets)
├── configure-dns.sh             # ACM + HTTPS + Route53 DNS setup
├── upload-models.sh             # Upload ML models to S3
├── destroy.sh                   # DELETE all resources
│
└── config.env                   # Auto-generated resource IDs
🛠️ Prerequisites

✔ AWS CLI installed
✔ AWS credentials configured
✔ Docker installed
✔ Domain hosted in Route53 (e.g. opkcloudz.com)
✔ Subdomain: api.opkcloudz.com
🚀 Deployment Steps
STEP 1 — Setup the full AWS infrastructure

Creates:

VPC

Subnets

NAT

IGW

ALB

S3

ECR

IAM

CloudWatch logs

2 public + 2 private subnets
Run:
chmod +x aws-infrastructure/setup-infrastructure.sh
./aws-infrastructure/setup-infrastructure.sh

This generates:
aws-infrastructure/config.env
STEP 2 — Upload ML Models to S3
./aws-infrastructure/upload-models.sh
STEP 3 — Build & Push Docker Image
./aws-infrastructure/build-and-push.sh
This builds and pushes:
<ACCOUNT_ID>.dkr.ecr.<region>.amazonaws.com/bearing-classifier:latest
STEP 4 — Create ECS Task Definition
./aws-infrastructure/create-task-definition.sh
Registers Fargate task with:

1 GB memory

0.5 vCPU

Port 8000

S3 access

CloudWatch logs
STEP 5 — Create ECS Fargate Service
./aws-infrastructure/create-service.sh
The service runs in:

Private subnets

Behind the ALB

No public IP
STEP 6 — Configure HTTPS + Route53 DNS
Creates:

ACM certificate

DNS validation records

HTTPS listener (port 443)

A-record for api.opkcloudz.com
Run:
./aws-infrastructure/configure-dns.sh
🎉 Your API is now LIVE

Test:

curl https://api.opkcloudz.com/health
or open in browser:

https://api.opkcloudz.com/health

🧹 Destroy All Resources

Run:

./aws-infrastructure/destroy.sh

✨ Congratulations!

You now have a fully automated production-grade AWS deployment, exactly like a senior DevOps engineer would build.

If you want, I can also generate:

✔ Terraform version
✔ GitHub Actions CI/CD pipeline
✔ Auto-scaling setup (ALB-based scaling)
✔ Logging + Monitoring dashboards (CloudWatch + Grafana)
