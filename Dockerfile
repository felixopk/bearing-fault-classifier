# -----------------------------
# STAGE 1 — Builder
# -----------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials for pandas & scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements for faster builds
COPY requirements.docker.txt .

# Install dependencies into a wheel folder
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.docker.txt


# -----------------------------
# STAGE 2 — Runner (Final Image)
# -----------------------------
FROM python:3.11-slim

WORKDIR /app

# Install system-level runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY app/ ./app
COPY models/ ./models

# Expose FastAPI port
EXPOSE 8000
# Start the app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
