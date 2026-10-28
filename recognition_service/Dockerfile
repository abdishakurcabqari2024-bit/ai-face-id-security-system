# Multi-Stage Dockerfile for AI Face ID Recognition Service
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies required for OpenCV headless and C++ math
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY recognition_service/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY config.py detector.py feature_extractor.py liveness.py models.py ./
COPY saved_models/ ./saved_models/
COPY recognition_service/main.py ./recognition_service/main.py

WORKDIR /app/recognition_service

ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
