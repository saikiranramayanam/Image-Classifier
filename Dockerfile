# -----------------------------
# Base image
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy dependency file
# -----------------------------
COPY requirements.txt .

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application source code
# -----------------------------
COPY src/ ./src/

# -----------------------------
# Expose API port
# -----------------------------
EXPOSE ${API_PORT}

# -----------------------------
# Start the API server
# -----------------------------
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${API_PORT}"]
