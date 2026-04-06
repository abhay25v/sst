# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENAI_API_KEY=${OPENAI_API_KEY:-dummy-key}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Verify critical dependencies
RUN python -c "from fastapi import FastAPI; print('✓ FastAPI verified')" || exit 1

# Copy application code
COPY . .

# Expose port (HF Spaces expects 7860 or uses the one you specify)
EXPOSE 8000

# Simple CMD - let Uvicorn handle everything
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
