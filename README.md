🛠️ Bearing Fault Classifier – Production-Ready ML API (FastAPI + ECS Fargate + GitHub Actions CI/CD)

A fully containerized, cloud-native Machine Learning inference API built with FastAPI, deployed on AWS ECS Fargate, fronted by an Application Load Balancer, and powered by a secure CI/CD pipeline using GitHub Actions.

This project demonstrates real DevOps, MLOps, and cloud infrastructure skills:

🚀 CI/CD on GitHub Actions

🐳 Docker containerization

☁️ AWS ECS Fargate orchestration

🌐 Application Load Balancer (ALB)

🔐 IAM Role-based S3 model access

📦 S3-hosted ML model artifacts

📊 CloudWatch logs and metrics

Architecture Overview
               +----------------------------+
               |        GitHub Repo         |
               |  - Code                    |
               |  - Dockerfile              |
               |  - GitHub Actions          |
               +-------------+--------------+
                             |
                             | Push to main
                             v
              +--------------+----------------+
              |   GitHub Actions CI/CD        |
              |--------------------------------|
              | 1. Lint & Test                 |
              | 2. Build Docker Image          |
              | 3. Push to Amazon ECR          |
              | 4. Deploy ECS Task Revision    |
              +---------------+----------------+
                              |
                              v
         +--------------------+---------------------+
         |                AWS ECS Fargate           |
         |------------------------------------------|
         |  Cluster: bearing-classifier-cluster      |
         |  Service: bearing-classifier-service      |
         |  Task: bearing-classifier-task            |
         +--------------------+----------------------+
                              |
                              v
                  +-----------+------------+
                  |  Application Load      |
                  |       Balancer         |
                  |    (HTTP: 80)          |
                  +-----------+------------+
                              |
                              v
                     Public API Endpoint
    http://bearing-classifier-alb-xxxxxxx.us-east-1.elb.amazonaws.com

🚀 Features
1. FastAPI Backend

High-performance ML inference API

/predict → single prediction

/predict/batch → CSV file predictions

/model/info → introspection

/health → load balancer health check



🔹 2. Secure ML Model Storage on S3

  The trained model files are stored at:

s3://opkcloudz-ml-models/bearing-classifier/v1.0.0/
Downloaded at container startup via boto3 using IAM Task Role.

FROM python:3.11-slim
WORKDIR /app
COPY requirements.docker.txt .
RUN pip install -r requirements.docker.txt
COPY app/ app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
4. AWS ECS Fargate Deployment

No servers to manage

Auto-scaling ready

Isolated VPC networking

Public subnets for ALB

Private subnets + NAT optional

. Application Load Balancer (ALB)

Public entry point

Routes traffic → ECS tasks

Performs health checks on /health

Ensures zero-downtime rolling deployments

🔹 6. GitHub Actions CI/CD

Automatic deployments on every push to main:

Pipeline stages:

Lint & test

Build Docker image

Push to Amazon ECR

Render new Task Definition

Update ECS service

Wait for ALB health checks

Full file:
.github/workflows/deploy-aws.yml


📦 Project Structure

bearing-fault-classifier/
│
├── app/
│   ├── main.py                # FastAPI app
│   ├── download_models.py     # S3 model download logic
│   ├── templates/             # HTML templates
│   └── static/                # CSS / JS
│
├── aws-infrastructure/        # Deployment scripts
├── models/                    # Local model storage (ignored in Docker)
├── data/                      # Training/processed data
├── requirements.txt
├── requirements.docker.txt
├── Dockerfile
└── .github/workflows/
        └── deploy-aws.yml

🧪 Local Development
1️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run locally
uvicorn app.main:app --reload
Open:

📍 http://localhost:8000/docs

📍 http://localhost:8000/health


🐳 Build & Run Docker Locally
docker build -t bearing-classifier .
docker run -p 8000:8000 bearing-classifier
☁️ Deploying to AWS ECS

Deployment is automated using GitHub Actions.

To deploy:

Push your code to the main branch

GitHub Actions automatically:

Builds

Pushes to ECR

Waits for ALB health

Service becomes available at the ALB DNS.

Example health check:
curl http://bearing-classifier-alb-xxxx.us-east-1.elb.amazonaws.com/health
📊 Observability
🔹 CloudWatch Logs

Every ECS task streams logs to:
/ecs/bearing-classifier-task

🔹 CloudWatch Metrics

Request count

Target group health

Latency

Container CPU/Memory
🔐 Security

IAM Task Role with read-only S3 access

No credentials stored in code or container

Private ECR registry

CI/CD secrets stored in GitHub Actions

HTTPS support via ALB (optional)
🧠 ML Model Info

Algorithm: Random Forest Classifier

Accuracy: 96.20%

Features: 19 vibration signal features

Classes:

Ball fault

Inner race

Outer race

Normal
