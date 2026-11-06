# Multi-stage Dockerfile for SentryICU (Full-Stack React + PyTorch FastAPI)
# Stage 1: Build the React UI Dashboard
FROM node:20-alpine AS frontend-builder
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# Stage 2: Python Backend with Full PyTorch Neural Models
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (Full PyTorch + FastAPI + Dependencies)
COPY requirements.txt .
# Install CPU-specific PyTorch to optimize image size and startup speed while preserving 100% PyTorch C++ kernels
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, benchmarks, and notebooks
COPY src/ ./src/
COPY data/sample/ ./data/sample/
COPY results/ ./results/
COPY notebooks/ ./notebooks/

# Copy compiled frontend from Stage 1 into ui/dist
COPY --from=frontend-builder /app/ui/dist ./ui/dist

# Expose port
EXPOSE 8000

# Launch FastAPI web service with full PyTorch models
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
