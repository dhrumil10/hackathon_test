FROM python:3.12-slim

# Install system dependencies including C++ compiler
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Expose port 8080
EXPOSE 8080

# Run FastAPI
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port $PORT
