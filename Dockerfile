# Production Dockerfile for AWS ECS deploymentS
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy application code
COPY app/ ./app/
COPY src/ ./src/

# Create models directory (will be populated at runtime from S3)
RUN mkdir -p ./models

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

EXPOSE 8000

# Health check with longer start period for model download
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start command
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}